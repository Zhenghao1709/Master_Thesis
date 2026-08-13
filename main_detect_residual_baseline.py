from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config.kelmarsh_config import FREQ_MINUTES, TARGET_COLS
from src.data.read_event_dataset import read_event_dataset
from src.detection.residuals import (
    apply_residual_baseline_detection,
    fit_residual_quantile_thresholds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run residual quantile baseline detection on 2023-2024 test predictions."
    )
    parser.add_argument("--run-id", help="Experiment run ID. Defaults to the newest experiment.")
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.95,
        help="Validation residual quantile used as the anomaly threshold.",
    )
    parser.add_argument(
        "--min-consecutive",
        type=int,
        default=6,
        help="Minimum consecutive anomaly points required to raise an alarm.",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=7,
        help="A fault is detected if an alarm occurs this many days before its start.",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def find_experiment(project_root: Path, requested_run_id: str | None) -> tuple[Path, Path, dict]:
    experiments_dir = project_root / "results" / "kelmarsh" / "experiments"
    if requested_run_id:
        metadata_path = experiments_dir / requested_run_id / "metadata.json"
    else:
        candidates = sorted(
            experiments_dir.glob("*/metadata.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError("No experiment metadata found. Run main_train_gru.py first.")
        metadata_path = candidates[0]

    if not metadata_path.exists():
        raise FileNotFoundError(f"Experiment metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata_path.parent, metadata_path, metadata


def load_prediction_file(
    project_root: Path,
    metadata: dict,
    path_key: str,
    fallback_name: str,
) -> pd.DataFrame:
    path_value = metadata.get("paths", {}).get(path_key)
    path = resolve_path(project_root, path_value) if path_value else None
    if path is None or not path.exists():
        path = project_root / "results" / "kelmarsh" / "experiments" / metadata["run_id"] / fallback_name
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    return pd.read_csv(path)


def normalise_event_turbine(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower().startswith("kelmarsh_"):
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    return f"Kelmarsh_{digits}" if digits else None


def evaluate_fault_horizon(
    detections: pd.DataFrame,
    events: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    if "Timestamp start" not in events.columns:
        raise ValueError("Event dataset must contain 'Timestamp start'.")
    if "Turbine" not in events.columns:
        raise ValueError("Event dataset must contain 'Turbine'.")

    alarms = detections.loc[detections["is_alarm"]].copy()
    alarms["Date and time"] = pd.to_datetime(alarms["Date and time"], errors="coerce")

    rows = []
    events = events.copy()
    events["event_turbine_id"] = events["Turbine"].map(normalise_event_turbine)
    events["Timestamp start"] = pd.to_datetime(events["Timestamp start"], errors="coerce")
    if "Timestamp end" in events.columns:
        events["Timestamp end"] = pd.to_datetime(events["Timestamp end"], errors="coerce")

    events = events[
        events["Timestamp start"].notna()
        & events["Timestamp start"].dt.year.isin([2023, 2024])
        & events["event_turbine_id"].notna()
    ].copy()

    horizon = pd.Timedelta(days=horizon_days)
    for event_index, event in events.reset_index(drop=True).iterrows():
        event_start = event["Timestamp start"]
        window_start = event_start - horizon
        event_alarms = alarms[
            (alarms["turbine_id"] == event["event_turbine_id"])
            & (alarms["Date and time"] >= window_start)
            & (alarms["Date and time"] < event_start)
        ].copy()

        if event_alarms.empty:
            detected = False
            first_alarm_time = pd.NaT
            lead_time_hours = pd.NA
            alarm_count = 0
            alarm_targets = ""
        else:
            detected = True
            first_alarm_time = event_alarms["Date and time"].min()
            lead_time_hours = (event_start - first_alarm_time).total_seconds() / 3600
            alarm_count = int(len(event_alarms))
            alarm_targets = "; ".join(sorted(event_alarms["target"].dropna().unique()))

        rows.append(
            {
                "event_index": event_index,
                "turbine_id": event["event_turbine_id"],
                "event_start": event_start,
                "event_end": event.get("Timestamp end", pd.NaT),
                "category": event.get("Category", ""),
                "component": event.get("Component", ""),
                "message": event.get("Message stack", ""),
                "horizon_start": window_start,
                "detected_in_horizon": detected,
                "first_alarm_time": first_alarm_time,
                "lead_time_hours": lead_time_hours,
                "alarm_count_in_horizon": alarm_count,
                "alarm_targets": alarm_targets,
            }
        )

    return pd.DataFrame(rows)


def annotate_alarm_horizon_matches(
    detections: pd.DataFrame,
    event_summary: pd.DataFrame,
) -> pd.DataFrame:
    out = detections.copy()
    out["in_fault_horizon"] = False
    if event_summary.empty:
        return out

    out["Date and time"] = pd.to_datetime(out["Date and time"], errors="coerce")
    for _, event in event_summary.iterrows():
        mask = (
            out["is_alarm"]
            & (out["turbine_id"] == event["turbine_id"])
            & (out["Date and time"] >= event["horizon_start"])
            & (out["Date and time"] < event["event_start"])
        )
        out.loc[mask, "in_fault_horizon"] = True
    return out


def build_alarm_episodes(
    detections: pd.DataFrame,
    freq_minutes: int = FREQ_MINUTES,
) -> pd.DataFrame:
    alarms = detections.loc[detections["is_alarm"]].copy()
    if alarms.empty:
        return pd.DataFrame(
            columns=[
                "episode_id",
                "turbine_id",
                "target",
                "start_time",
                "end_time",
                "alarm_points",
                "in_fault_horizon",
            ]
        )

    alarms["Date and time"] = pd.to_datetime(alarms["Date and time"], errors="coerce")
    alarms = alarms.sort_values(["turbine_id", "target", "segment_id", "Date and time"]).reset_index(drop=True)
    expected_delta = pd.Timedelta(minutes=freq_minutes)
    gap = alarms.groupby(["turbine_id", "target", "segment_id"])["Date and time"].diff().ne(expected_delta)
    alarms["episode_id"] = gap.cumsum()

    episodes = (
        alarms.groupby(["episode_id", "turbine_id", "target"], as_index=False)
        .agg(
            start_time=("Date and time", "min"),
            end_time=("Date and time", "max"),
            alarm_points=("Date and time", "size"),
            in_fault_horizon=("in_fault_horizon", "max"),
        )
        .sort_values(["turbine_id", "target", "start_time"])
        .reset_index(drop=True)
    )
    return episodes


def summarize_detection_performance(
    detections: pd.DataFrame,
    event_summary: pd.DataFrame,
    alarm_episodes: pd.DataFrame,
) -> dict:
    total_events = int(len(event_summary))
    detected_events = int(event_summary["detected_in_horizon"].sum()) if total_events else 0
    missed_events = total_events - detected_events

    alarm_points = detections.loc[detections["is_alarm"]].copy()
    total_alarm_points = int(len(alarm_points))
    true_alarm_points = int(alarm_points["in_fault_horizon"].sum()) if total_alarm_points else 0
    false_alarm_points = total_alarm_points - true_alarm_points

    total_alarm_episodes = int(len(alarm_episodes))
    true_alarm_episodes = (
        int(alarm_episodes["in_fault_horizon"].sum()) if total_alarm_episodes else 0
    )
    false_alarm_episodes = total_alarm_episodes - true_alarm_episodes

    def safe_ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    return {
        "event_level": {
            "total_events": total_events,
            "detected_events": detected_events,
            "missed_events": missed_events,
            "detection_rate": safe_ratio(detected_events, total_events),
            "miss_rate": safe_ratio(missed_events, total_events),
        },
        "alarm_point_level": {
            "total_alarm_points": total_alarm_points,
            "true_alarm_points": true_alarm_points,
            "false_alarm_points": false_alarm_points,
            "false_alarm_point_rate": safe_ratio(false_alarm_points, total_alarm_points),
        },
        "alarm_episode_level": {
            "total_alarm_episodes": total_alarm_episodes,
            "true_alarm_episodes": true_alarm_episodes,
            "false_alarm_episodes": false_alarm_episodes,
            "false_alarm_episode_rate": safe_ratio(false_alarm_episodes, total_alarm_episodes),
        },
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.quantile < 1:
        raise ValueError("--quantile must be between 0 and 1")
    if args.min_consecutive < 1:
        raise ValueError("--min-consecutive must be at least 1")
    if args.horizon_days < 1:
        raise ValueError("--horizon-days must be at least 1")

    project_root = Path(__file__).resolve().parent
    result_dir, metadata_path, metadata = find_experiment(project_root, args.run_id)

    val_predictions = load_prediction_file(
        project_root,
        metadata,
        path_key="predictions",
        fallback_name="healthy_val_predictions.csv",
    )
    test_predictions = load_prediction_file(
        project_root,
        metadata,
        path_key="test_predictions",
        fallback_name="test_2023_2024_predictions.csv",
    )

    target_cols = metadata.get("targets", TARGET_COLS)
    thresholds = fit_residual_quantile_thresholds(
        val_predictions,
        target_cols=target_cols,
        quantile=args.quantile,
    )
    detections = apply_residual_baseline_detection(
        test_predictions,
        thresholds=thresholds,
        target_cols=target_cols,
        min_consecutive=args.min_consecutive,
    )

    suffix = f"q{int(args.quantile * 1000):03d}_c{args.min_consecutive}"
    threshold_path = result_dir / f"residual_baseline_thresholds_{suffix}.csv"
    detection_path = result_dir / f"residual_baseline_detections_{suffix}.csv"
    event_summary_path = result_dir / f"residual_baseline_event_summary_{suffix}.csv"
    episode_path = result_dir / f"residual_baseline_alarm_episodes_{suffix}.csv"
    performance_path = result_dir / f"residual_baseline_performance_{suffix}.json"

    thresholds.to_csv(threshold_path, index=False, encoding="utf-8-sig")

    event_path = project_root / "data" / "raw" / "kelmarsh" / "auxiliary" / "kelmarsh_event_dataset.csv"
    event_summary = None
    alarm_episodes = None
    performance = None
    if event_path.exists():
        events = read_event_dataset(event_path)
        event_summary = evaluate_fault_horizon(detections, events, horizon_days=args.horizon_days)
        detections = annotate_alarm_horizon_matches(detections, event_summary)
        alarm_episodes = build_alarm_episodes(detections)
        performance = summarize_detection_performance(detections, event_summary, alarm_episodes)
        event_summary.to_csv(event_summary_path, index=False, encoding="utf-8-sig")
        alarm_episodes.to_csv(episode_path, index=False, encoding="utf-8-sig")
        performance_path.write_text(json.dumps(performance, indent=2), encoding="utf-8")

    detections.to_csv(detection_path, index=False, encoding="utf-8-sig")

    alarm_rate = float(detections["is_alarm"].mean())
    metadata["residual_baseline_detection"] = {
        "quantile": args.quantile,
        "min_consecutive": args.min_consecutive,
        "horizon_days": args.horizon_days,
        "thresholds_path": str(threshold_path.relative_to(project_root)),
        "detections_path": str(detection_path.relative_to(project_root)),
        "event_summary_path": str(event_summary_path.relative_to(project_root)) if event_summary is not None else None,
        "alarm_episodes_path": str(episode_path.relative_to(project_root)) if alarm_episodes is not None else None,
        "performance_path": str(performance_path.relative_to(project_root)) if performance is not None else None,
        "test_detection_rows": int(len(detections)),
        "test_alarm_rate": alarm_rate,
        "performance": performance,
    }
    metadata.setdefault("paths", {})["residual_baseline_thresholds"] = str(
        threshold_path.relative_to(project_root)
    )
    metadata["paths"]["residual_baseline_detections"] = str(detection_path.relative_to(project_root))
    if event_summary is not None:
        metadata["paths"]["residual_baseline_event_summary"] = str(
            event_summary_path.relative_to(project_root)
        )
        metadata["paths"]["residual_baseline_alarm_episodes"] = str(
            episode_path.relative_to(project_root)
        )
        metadata["paths"]["residual_baseline_performance"] = str(
            performance_path.relative_to(project_root)
        )
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Run ID:", metadata["run_id"])
    print("Thresholds saved to:", threshold_path)
    print("Detections saved to:", detection_path)
    print("Alarm rate:", f"{alarm_rate:.4%}")
    if event_summary is not None:
        detected_count = int(event_summary["detected_in_horizon"].sum())
        print("Event summary saved to:", event_summary_path)
        print("Detected events in horizon:", f"{detected_count}/{len(event_summary)}")
        print("Alarm episodes saved to:", episode_path)
        print("Performance summary saved to:", performance_path)
        print("Miss rate:", f"{performance['event_level']['miss_rate']:.4%}")
        print(
            "False alarm episode rate:",
            f"{performance['alarm_episode_level']['false_alarm_episode_rate']:.4%}",
        )


if __name__ == "__main__":
    main()
