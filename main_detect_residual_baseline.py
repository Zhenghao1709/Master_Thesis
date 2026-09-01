from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config.kelmarsh_config import FREQ_MINUTES, TARGET_COLS
from src.data.read_event_dataset import read_event_dataset
from src.data.read_status import read_status_folder
from src.detection.residuals import (
    apply_residual_baseline_detection,
    fit_residual_quantile_thresholds,
)
from src.preprocessing.clean_status import clean_status_df


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


def normalise_event_turbine(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower().startswith("kelmarsh_"):
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    return f"Kelmarsh_{digits}" if digits else None


def append_target_interval(
    rows: list[dict],
    turbine_id: object,
    event_start: object,
    event_end: object,
    category: str,
    subcategory: str = "",
    component: str = "",
    message: str = "",
    event_source: str = "",
) -> None:
    turbine = normalise_event_turbine(turbine_id)
    start = pd.to_datetime(event_start, errors="coerce")
    end = pd.to_datetime(event_end, errors="coerce")
    if turbine is None or pd.isna(start) or pd.isna(end) or end < start:
        return
    if start.year not in {2023, 2024}:
        return

    rows.append(
        {
            "turbine_id": turbine,
            "event_start": start,
            "event_end": end,
            "category": str(category),
            "subcategory": str(subcategory),
            "component": str(component),
            "message": str(message),
            "event_source": str(event_source),
        }
    )


def merge_target_event_intervals(events: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    columns = [
        "event_index",
        "turbine_id",
        "event_start",
        "event_end",
        "category",
        "subcategory",
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
    if events.empty:
        return pd.DataFrame(columns=columns)

    horizon = pd.Timedelta(days=horizon_days)
    events = events.copy()
    events["event_start"] = pd.to_datetime(events["event_start"], errors="coerce")
    events["event_end"] = pd.to_datetime(events["event_end"], errors="coerce")
    events = events.dropna(subset=["event_start", "event_end", "turbine_id"])
    events = events.sort_values(["turbine_id", "event_start", "event_end"]).reset_index(drop=True)

    merged_rows = []
    for turbine_id, group in events.groupby("turbine_id", sort=False):
        current = None
        for _, row in group.iterrows():
            if current is None:
                current = row.to_dict()
                continue

            if row["event_start"] <= current["event_end"] + pd.Timedelta(minutes=FREQ_MINUTES):
                current["event_end"] = max(current["event_end"], row["event_end"])
                for col in ["category", "subcategory", "component", "message", "event_source"]:
                    values = {
                        value
                        for value in [*str(current.get(col, "")).split("; "), str(row.get(col, ""))]
                        if value and value != "nan"
                    }
                    current[col] = "; ".join(sorted(values))
            else:
                merged_rows.append(current)
                current = row.to_dict()
        if current is not None:
            merged_rows.append(current)

    out = pd.DataFrame(merged_rows).sort_values(["turbine_id", "event_start", "event_end"]).reset_index(drop=True)
    out.insert(0, "event_index", range(len(out)))
    out["horizon_start"] = out["event_start"] - horizon
    out["detected_in_horizon"] = False
    out["first_alarm_time"] = pd.NaT
    out["lead_time_hours"] = pd.NA
    out["alarm_count_in_horizon"] = 0
    out["alarm_targets"] = ""
    return out[columns]


def build_target_event_summary(
    project_root: Path,
    horizon_days: int,
) -> pd.DataFrame:
    rows = []

    event_path = project_root / "data" / "raw" / "kelmarsh" / "auxiliary" / "kelmarsh_event_dataset.csv"
    if event_path.exists():
        events = read_event_dataset(event_path)
        events["category_lower"] = events["Category"].astype(str).str.strip().str.casefold()
        events["subcategory_lower"] = events["Subcategory"].astype(str).str.strip().str.casefold()

        auxiliary_targets = events[
            events["category_lower"].eq("fault")
            | (
                events["category_lower"].eq("maintenance")
                & events["subcategory_lower"].isin({"corrective", "corrective - merged"})
            )
        ].copy()

        for _, row in auxiliary_targets.iterrows():
            append_target_interval(
                rows,
                turbine_id=row.get("Turbine"),
                event_start=row.get("Timestamp start"),
                event_end=row.get("Timestamp end"),
                category=row.get("Category", ""),
                subcategory=row.get("Subcategory", ""),
                component=row.get("Component", ""),
                message=row.get("Message stack", ""),
                event_source="auxiliary_event_dataset",
            )

    status_root = project_root / "data" / "raw" / "kelmarsh" / "status"
    target_iec_categories = {
        "forced outage",
        "out of electrical specification",
        "out of environmental specification",
    }
    long_warning_min_duration = pd.Timedelta(days=7)
    if status_root.exists():
        for status_folder in sorted(p for p in status_root.iterdir() if p.is_dir()):
            turbine_id = status_folder.name
            status = clean_status_df(read_status_folder(status_folder, turbine_id=turbine_id))
            stop_targets = status[
                status["Status"].astype(str).str.strip().str.casefold().eq("stop")
                & status["IEC category"].astype(str).str.strip().str.casefold().isin(target_iec_categories)
            ].copy()
            warning_targets = status[
                status["Status"].astype(str).str.strip().str.casefold().eq("warning")
                & (status["Timestamp end"] - status["Timestamp start"] >= long_warning_min_duration)
            ].copy()

            for _, row in stop_targets.iterrows():
                append_target_interval(
                    rows,
                    turbine_id=row.get("turbine_id"),
                    event_start=row.get("Timestamp start"),
                    event_end=row.get("Timestamp end"),
                    category="Status Stop",
                    subcategory=row.get("IEC category", ""),
                    component=row.get("Service contract category", ""),
                    message=row.get("Message", ""),
                    event_source="status_stop_target",
                )
            for _, row in warning_targets.iterrows():
                append_target_interval(
                    rows,
                    turbine_id=row.get("turbine_id"),
                    event_start=row.get("Timestamp start"),
                    event_end=row.get("Timestamp end"),
                    category="Status Warning",
                    subcategory="duration >= 7 days",
                    component=row.get("Service contract category", ""),
                    message=row.get("Message", ""),
                    event_source="status_long_warning_target",
                )

    return merge_target_event_intervals(pd.DataFrame(rows), horizon_days=horizon_days)


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


def non_full_performance_intervals_from_status(
    project_root: Path,
    turbine_id: str,
    start_year: int = 2023,
    end_year: int = 2024,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    status_folder = project_root / "data" / "raw" / "kelmarsh" / "status" / turbine_id
    if not status_folder.exists():
        return []

    status = read_status_folder(status_folder, turbine_id=turbine_id)
    status["Timestamp start"] = pd.to_datetime(status["Timestamp start"], errors="coerce")
    status["Timestamp end"] = pd.to_datetime(status["Timestamp end"], errors="coerce")
    iec = status["IEC category"].astype(str).str.strip().str.casefold()
    status = status[
        status["Timestamp start"].notna()
        & status["Timestamp end"].notna()
        & status["Timestamp start"].dt.year.between(start_year, end_year)
        & ~iec.eq("full performance")
    ].copy()

    return list(zip(status["Timestamp start"], status["Timestamp end"]))


def informational_environmental_spec_intervals_from_status(
    project_root: Path,
    turbine_id: str,
    start_year: int = 2023,
    end_year: int = 2024,
    buffer_hours: int = 1,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    status_folder = project_root / "data" / "raw" / "kelmarsh" / "status" / turbine_id
    if not status_folder.exists():
        return []

    status = read_status_folder(status_folder, turbine_id=turbine_id)
    status["Timestamp start"] = pd.to_datetime(status["Timestamp start"], errors="coerce")
    status["Timestamp end"] = pd.to_datetime(status["Timestamp end"], errors="coerce")
    status_text = status["Status"].astype(str).str.strip().str.casefold()
    iec = status["IEC category"].astype(str).str.strip().str.casefold()
    status = status[
        status["Timestamp start"].notna()
        & status["Timestamp end"].notna()
        & status["Timestamp start"].dt.year.between(start_year, end_year)
        & status_text.isin(["information", "informational"])
        & iec.eq("out of environmental specification")
    ].copy()

    buffer = pd.Timedelta(hours=buffer_hours)
    status["Timestamp start"] = status["Timestamp start"] - buffer
    status["Timestamp end"] = status["Timestamp end"] + buffer
    return list(zip(status["Timestamp start"], status["Timestamp end"]))


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
    out["in_non_full_performance_interval"] = False
    out["in_information_environmental_buffer_interval"] = False
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
        non_full_performance_intervals = non_full_performance_intervals_from_status(
            project_root,
            turbine_id=str(turbine_id),
        )
        information_environmental_intervals = informational_environmental_spec_intervals_from_status(
            project_root,
            turbine_id=str(turbine_id),
        )

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
            out.loc[row_index, "in_non_full_performance_interval"] = overlaps_any_interval(
                start,
                end,
                non_full_performance_intervals,
            )
            out.loc[row_index, "in_information_environmental_buffer_interval"] = overlaps_any_interval(
                start,
                end,
                information_environmental_intervals,
            )

    non_operational_cols = list(context_flags.keys()) + [
        "in_non_full_performance_interval",
        "in_information_environmental_buffer_interval",
    ]
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

    event_summary = build_target_event_summary(
        project_root,
        horizon_days=args.horizon_days,
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
        "event_source": (
            "auxiliary Fault; auxiliary Maintenance Corrective/Corrective - merged; "
            "status Stop with IEC Forced outage/Out of Electrical Specification/"
            "Out of Environmental Specification; status Warning with duration >= 7 days"
        ),
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
    print("Event source: target event definition")
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
