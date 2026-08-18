from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config.kelmarsh_config import FREQ_MINUTES, TARGET_COLS
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
        default=0.995,
        help="Validation residual quantile used as the anomaly threshold.",
    )
    parser.add_argument(
        "--min-consecutive",
        type=int,
        default=12,
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


def turbine_id_from_flag_path(flag_path: Path) -> str:
    prefix = flag_path.stem.replace("_with_flags", "")
    return "_".join(part.capitalize() for part in prefix.split("_"))


def build_event_summary_from_flag_events(
    project_root: Path,
    horizon_days: int,
    event_flag_col: str = "in_event",
    start_year: int = 2023,
    end_year: int = 2024,
) -> pd.DataFrame:
    flags_dir = project_root / "data" / "interim" / "kelmarsh" / "flags"
    if not flags_dir.exists():
        raise FileNotFoundError(f"Missing flags directory: {flags_dir}")

    horizon = pd.Timedelta(days=horizon_days)
    rows = []
    for flag_path in sorted(flags_dir.glob("*_with_flags.parquet")):
        turbine_id = turbine_id_from_flag_path(flag_path)
        intervals = intervals_from_flag_file(
            flag_path,
            flag_cols=[event_flag_col],
            start_year=start_year,
            end_year=end_year,
        ).get(event_flag_col, [])

        for event_start, event_end in intervals:
            rows.append(
                {
                    "turbine_id": turbine_id,
                    "event_start": event_start,
                    "event_end": event_end,
                    "category": "status_flag_event",
                    "component": "",
                    "message": event_flag_col,
                    "event_source": f"flag:{event_flag_col}",
                    "horizon_start": event_start - horizon,
                    "detected_in_horizon": False,
                    "first_alarm_time": pd.NaT,
                    "lead_time_hours": pd.NA,
                    "alarm_count_in_horizon": 0,
                    "alarm_targets": "",
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "event_index",
                "turbine_id",
                "event_start",
                "event_end",
                "category",
                "component",
                "message",
                "event_source",
                "horizon_start",
                "detected_in_horizon",
                "first_alarm_time",
                "lead_time_hours",
                "alarm_count_in_horizon",
                "alarm_targets",
            ]
        )

    out = pd.DataFrame(rows).sort_values(["turbine_id", "event_start", "event_end"]).reset_index(drop=True)
    out.insert(0, "event_index", range(len(out)))
    return out


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


def annotate_alarm_episode_horizon_matches(
    alarm_episodes: pd.DataFrame,
    event_summary: pd.DataFrame,
    freq_minutes: int = FREQ_MINUTES,
) -> pd.DataFrame:
    out = alarm_episodes.copy()
    out["in_fault_horizon"] = False
    if out.empty or event_summary.empty:
        return out

    out["start_time"] = pd.to_datetime(out["start_time"], errors="coerce")
    out["end_time"] = pd.to_datetime(out["end_time"], errors="coerce")
    event_summary = event_summary.copy()
    event_summary["horizon_start"] = pd.to_datetime(event_summary["horizon_start"], errors="coerce")
    event_summary["event_start"] = pd.to_datetime(event_summary["event_start"], errors="coerce")

    # Treat an alarm episode as within the horizon when the episode overlaps
    # [event_start - horizon_days, event_start). This includes episodes that
    # started earlier than the horizon but end inside it or continue into it.
    episode_end_exclusive = out["end_time"] + pd.Timedelta(minutes=freq_minutes)
    for _, event in event_summary.iterrows():
        mask = (
            (out["turbine_id"] == event["turbine_id"])
            & (out["start_time"] < event["event_start"])
            & (episode_end_exclusive >= event["horizon_start"])
        )
        out.loc[mask, "in_fault_horizon"] = True

    return out


def update_event_summary_from_alarm_episodes(
    event_summary: pd.DataFrame,
    alarm_episodes: pd.DataFrame,
    freq_minutes: int = FREQ_MINUTES,
) -> pd.DataFrame:
    out = event_summary.copy()
    if out.empty:
        return out

    out["event_start"] = pd.to_datetime(out["event_start"], errors="coerce")
    out["horizon_start"] = pd.to_datetime(out["horizon_start"], errors="coerce")
    episodes = alarm_episodes.copy()
    episodes["start_time"] = pd.to_datetime(episodes["start_time"], errors="coerce")
    episodes["end_time"] = pd.to_datetime(episodes["end_time"], errors="coerce")
    episode_end_exclusive = episodes["end_time"] + pd.Timedelta(minutes=freq_minutes)

    for index, event in out.iterrows():
        overlapping = episodes[
            (episodes["turbine_id"] == event["turbine_id"])
            & (episodes["start_time"] < event["event_start"])
            & (episode_end_exclusive >= event["horizon_start"])
        ].copy()

        if overlapping.empty:
            out.loc[index, "detected_in_horizon"] = False
            out.loc[index, "first_alarm_time"] = pd.NaT
            out.loc[index, "lead_time_hours"] = pd.NA
            out.loc[index, "alarm_count_in_horizon"] = 0
            out.loc[index, "alarm_targets"] = ""
            out.loc[index, "alarm_episode_count_in_horizon"] = 0
            continue

        first_alarm_time = overlapping["start_time"].min()
        out.loc[index, "detected_in_horizon"] = True
        out.loc[index, "first_alarm_time"] = first_alarm_time
        out.loc[index, "lead_time_hours"] = (
            event["event_start"] - first_alarm_time
        ).total_seconds() / 3600
        out.loc[index, "alarm_count_in_horizon"] = int(overlapping["alarm_points"].sum())
        out.loc[index, "alarm_episode_count_in_horizon"] = int(len(overlapping))
        out.loc[index, "alarm_targets"] = "; ".join(sorted(overlapping["target"].dropna().unique()))

    return out


def intervals_from_flag_file(
    flag_path: Path,
    flag_cols: list[str],
    start_year: int = 2023,
    end_year: int = 2024,
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    if not flag_path.exists():
        return {flag_col: [] for flag_col in flag_cols}

    cols = ["Date and time"] + flag_cols
    flags = pd.read_parquet(flag_path, columns=cols)
    flags["Date and time"] = pd.to_datetime(flags["Date and time"], errors="coerce")
    flags = flags[
        flags["Date and time"].notna()
        & flags["Date and time"].dt.year.between(start_year, end_year)
    ].copy()

    intervals_by_flag = {}
    for flag_col in flag_cols:
        part = flags[["Date and time", flag_col]].copy()
        part[flag_col] = part[flag_col].fillna(False).astype(bool)
        part = part.sort_values("Date and time").reset_index(drop=True)
        groups = part[flag_col].ne(part[flag_col].shift()).cumsum()

        intervals = []
        for _, group in part.loc[part[flag_col]].groupby(groups):
            start = group["Date and time"].iloc[0]
            end = group["Date and time"].iloc[-1] + pd.Timedelta(minutes=FREQ_MINUTES)
            intervals.append((start, end))
        intervals_by_flag[flag_col] = intervals

    return intervals_by_flag


def overlaps_any_interval(
    start: pd.Timestamp,
    end: pd.Timestamp,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> bool:
    for interval_start, interval_end in intervals:
        if start < interval_end and end > interval_start:
            return True
    return False


def annotate_alarm_episode_operating_context(
    alarm_episodes: pd.DataFrame,
    project_root: Path,
    freq_minutes: int = FREQ_MINUTES,
) -> pd.DataFrame:
    out = alarm_episodes.copy()
    context_flags = {
        "in_manual_event_interval": "in_manual_event",
        "in_status_event_interval": "in_event",
        "in_maintenance_interval": "in_maintenance",
        "in_communication_interval": "in_communication",
        "in_curtailment_interval": "in_curtailment",
    }
    for output_col in context_flags:
        out[output_col] = False
    out["in_non_operational_interval"] = False
    out["is_operational_false_alarm"] = False

    if out.empty:
        return out

    out["start_time"] = pd.to_datetime(out["start_time"], errors="coerce")
    out["end_time"] = pd.to_datetime(out["end_time"], errors="coerce")
    flag_cols = list(context_flags.values())
    flags_dir = project_root / "data" / "interim" / "kelmarsh" / "flags"

    for turbine_id, idx in out.groupby("turbine_id", sort=False).groups.items():
        flag_path = flags_dir / f"{str(turbine_id).lower()}_with_flags.parquet"
        intervals_by_flag = intervals_from_flag_file(flag_path, flag_cols)

        for row_index in idx:
            start = out.loc[row_index, "start_time"]
            end = out.loc[row_index, "end_time"] + pd.Timedelta(minutes=freq_minutes)
            if pd.isna(start) or pd.isna(end):
                continue

            for output_col, flag_col in context_flags.items():
                out.loc[row_index, output_col] = overlaps_any_interval(
                    start,
                    end,
                    intervals_by_flag.get(flag_col, []),
                )

    non_operational_cols = list(context_flags.keys())
    out["in_non_operational_interval"] = out[non_operational_cols].any(axis=1)
    out["is_operational_false_alarm"] = (
        ~out["in_fault_horizon"].fillna(False).astype(bool)
        & ~out["in_non_operational_interval"].fillna(False).astype(bool)
    )
    return out


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
    non_operational_alarm_episodes = (
        int(alarm_episodes["in_non_operational_interval"].sum())
        if "in_non_operational_interval" in alarm_episodes.columns and total_alarm_episodes
        else 0
    )
    operational_false_alarm_episodes = (
        int(alarm_episodes["is_operational_false_alarm"].sum())
        if "is_operational_false_alarm" in alarm_episodes.columns and total_alarm_episodes
        else false_alarm_episodes
    )
    operational_alarm_opportunities = true_alarm_episodes + operational_false_alarm_episodes

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
            "non_operational_alarm_episodes": non_operational_alarm_episodes,
            "operational_false_alarm_episodes": operational_false_alarm_episodes,
            "operational_alarm_opportunities": operational_alarm_opportunities,
            "operational_false_alarm_rate": safe_ratio(
                operational_false_alarm_episodes,
                operational_alarm_opportunities,
            ),
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

    event_summary = build_event_summary_from_flag_events(
        project_root,
        horizon_days=args.horizon_days,
        event_flag_col="in_event",
    )
    detections = annotate_alarm_horizon_matches(detections, event_summary)
    alarm_episodes = build_alarm_episodes(detections)
    alarm_episodes = annotate_alarm_episode_horizon_matches(alarm_episodes, event_summary)
    alarm_episodes = annotate_alarm_episode_operating_context(alarm_episodes, project_root)
    event_summary = update_event_summary_from_alarm_episodes(event_summary, alarm_episodes)
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
        "event_source": "flag:in_event",
        "thresholds_path": str(threshold_path.relative_to(project_root)),
        "detections_path": str(detection_path.relative_to(project_root)),
        "event_summary_path": str(event_summary_path.relative_to(project_root)),
        "alarm_episodes_path": str(episode_path.relative_to(project_root)),
        "performance_path": str(performance_path.relative_to(project_root)),
        "test_detection_rows": int(len(detections)),
        "test_alarm_rate": alarm_rate,
        "performance": performance,
    }
    metadata.setdefault("paths", {})["residual_baseline_thresholds"] = str(
        threshold_path.relative_to(project_root)
    )
    metadata["paths"]["residual_baseline_detections"] = str(detection_path.relative_to(project_root))
    metadata["paths"]["residual_baseline_event_summary"] = str(event_summary_path.relative_to(project_root))
    metadata["paths"]["residual_baseline_alarm_episodes"] = str(episode_path.relative_to(project_root))
    metadata["paths"]["residual_baseline_performance"] = str(performance_path.relative_to(project_root))
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Run ID:", metadata["run_id"])
    print("Thresholds saved to:", threshold_path)
    print("Detections saved to:", detection_path)
    print("Alarm rate:", f"{alarm_rate:.4%}")
    detected_count = int(event_summary["detected_in_horizon"].sum())
    print("Event source: flag:in_event")
    print("Event summary saved to:", event_summary_path)
    print("Detected events in horizon:", f"{detected_count}/{len(event_summary)}")
    print("Alarm episodes saved to:", episode_path)
    print("Performance summary saved to:", performance_path)
    print("Miss rate:", f"{performance['event_level']['miss_rate']:.4%}")
    print(
        "False alarm episode rate:",
        f"{performance['alarm_episode_level']['false_alarm_episode_rate']:.4%}",
    )
    print(
        "Operational false alarm rate:",
        f"{performance['alarm_episode_level']['operational_false_alarm_rate']:.4%}",
    )


if __name__ == "__main__":
    main()
