from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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
    evaluation_start: object = "2023-01-01",
    evaluation_end_exclusive: object = "2025-01-01",
) -> None:
    turbine = normalise_event_turbine(turbine_id)
    start = pd.to_datetime(event_start, errors="coerce")
    end = pd.to_datetime(event_end, errors="coerce")
    if turbine is None or pd.isna(start) or pd.isna(end) or end < start:
        return
    period_start = pd.Timestamp(evaluation_start)
    period_end = pd.Timestamp(evaluation_end_exclusive)
    if not (period_start <= start < period_end):
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
    evaluation_start: object = "2023-01-01",
    evaluation_end_exclusive: object = "2025-01-01",
    include_long_warnings: bool = False,
) -> pd.DataFrame:
    """Build target events without status Warnings by default.

    ``include_long_warnings=True`` reproduces the earlier evaluation definition;
    operating-context status flags are constructed independently.
    """
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
                evaluation_start=evaluation_start,
                evaluation_end_exclusive=evaluation_end_exclusive,
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
                    evaluation_start=evaluation_start,
                    evaluation_end_exclusive=evaluation_end_exclusive,
                )
            if include_long_warnings:
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
                        evaluation_start=evaluation_start,
                        evaluation_end_exclusive=evaluation_end_exclusive,
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
    merge_gap_hours: float = 0.0,
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
    if merge_gap_hours > 0 and not episodes.empty:
        merge_delta = pd.Timedelta(hours=merge_gap_hours)
        merged_rows = []
        for (_, target), group in episodes.groupby(["turbine_id", "target"], sort=False):
            current = None
            for _, row in group.sort_values("start_time").iterrows():
                if current is None:
                    current = row.to_dict()
                    continue

                if row["start_time"] <= current["end_time"] + merge_delta:
                    current["end_time"] = max(current["end_time"], row["end_time"])
                    current["alarm_points"] += int(row["alarm_points"])
                    current["in_fault_horizon"] = bool(current["in_fault_horizon"]) or bool(
                        row["in_fault_horizon"]
                    )
                else:
                    merged_rows.append(current)
                    current = row.to_dict()
            if current is not None:
                merged_rows.append(current)

        episodes = pd.DataFrame(merged_rows).sort_values(
            ["turbine_id", "target", "start_time"]
        ).reset_index(drop=True)
        episodes["episode_id"] = range(len(episodes))
    return episodes


def fuse_alarm_episodes(
    episodes: pd.DataFrame,
    freq_minutes: int = FREQ_MINUTES,
) -> pd.DataFrame:
    """Union simultaneous or adjacent per-target episodes into turbine-level episodes."""
    columns = ["episode_id", "turbine_id", "start_time", "end_time", "alarm_points", "in_fault_horizon"]
    if episodes.empty:
        return pd.DataFrame(columns=columns)

    source = episodes.copy()
    source["start_time"] = pd.to_datetime(source["start_time"], errors="coerce")
    source["end_time"] = pd.to_datetime(source["end_time"], errors="coerce")
    source = source.dropna(subset=["turbine_id", "start_time", "end_time"])
    step = pd.Timedelta(minutes=freq_minutes)
    rows = []
    for turbine, group in source.groupby("turbine_id", sort=False):
        current_start = current_end = None
        for row in group.sort_values(["start_time", "end_time"]).itertuples(index=False):
            if current_start is None:
                current_start, current_end = row.start_time, row.end_time
            elif row.start_time <= current_end + step:
                current_end = max(current_end, row.end_time)
            else:
                rows.append((turbine, current_start, current_end))
                current_start, current_end = row.start_time, row.end_time
        if current_start is not None:
            rows.append((turbine, current_start, current_end))

    out = pd.DataFrame(rows, columns=["turbine_id", "start_time", "end_time"])
    out["episode_id"] = range(len(out))
    out["alarm_points"] = ((out["end_time"] - out["start_time"]) / step).astype(int) + 1
    out["in_fault_horizon"] = False
    return out[columns]


def build_turbine_alarm_episodes(
    detections: pd.DataFrame,
    freq_minutes: int = FREQ_MINUTES,
    merge_gap_hours: float = 0.0,
) -> pd.DataFrame:
    return fuse_alarm_episodes(
        build_alarm_episodes(detections, freq_minutes=freq_minutes, merge_gap_hours=merge_gap_hours),
        freq_minutes=freq_minutes,
    )


def annotate_alarm_episode_horizon_matches(
    alarm_episodes: pd.DataFrame,
    event_summary: pd.DataFrame,
    freq_minutes: int = FREQ_MINUTES,
    trigger_time_col: str | None = None,
) -> pd.DataFrame:
    out = alarm_episodes.copy()
    out["in_fault_horizon"] = False
    if out.empty or event_summary.empty:
        return out

    out["start_time"] = pd.to_datetime(out["start_time"], errors="coerce")
    out["end_time"] = pd.to_datetime(out["end_time"], errors="coerce")
    if trigger_time_col is not None:
        if trigger_time_col not in out.columns:
            raise ValueError(f"Missing episode trigger time column: {trigger_time_col}")
        out[trigger_time_col] = pd.to_datetime(out[trigger_time_col], errors="coerce")
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
        if trigger_time_col is not None:
            mask &= out[trigger_time_col] < event["event_start"]
        out.loc[mask, "in_fault_horizon"] = True

    return out


def update_event_summary_from_alarm_episodes(
    event_summary: pd.DataFrame,
    alarm_episodes: pd.DataFrame,
    freq_minutes: int = FREQ_MINUTES,
    trigger_time_col: str = "start_time",
) -> pd.DataFrame:
    out = event_summary.copy()
    if out.empty:
        return out

    out["event_start"] = pd.to_datetime(out["event_start"], errors="coerce")
    out["horizon_start"] = pd.to_datetime(out["horizon_start"], errors="coerce")
    episodes = alarm_episodes.copy()
    if "target" not in episodes.columns and "in_non_operational_interval" in episodes.columns:
        episodes = episodes.loc[~episodes["in_non_operational_interval"].fillna(False).astype(bool)]
    episodes["start_time"] = pd.to_datetime(episodes["start_time"], errors="coerce")
    episodes["end_time"] = pd.to_datetime(episodes["end_time"], errors="coerce")
    if trigger_time_col not in episodes.columns:
        raise ValueError(f"Missing episode trigger time column: {trigger_time_col}")
    episodes[trigger_time_col] = pd.to_datetime(episodes[trigger_time_col], errors="coerce")
    episode_end_exclusive = episodes["end_time"] + pd.Timedelta(minutes=freq_minutes)

    for index, event in out.iterrows():
        overlapping = episodes[
            (episodes["turbine_id"] == event["turbine_id"])
            & (episodes["start_time"] < event["event_start"])
            & (episode_end_exclusive >= event["horizon_start"])
            & (episodes[trigger_time_col] < event["event_start"])
        ].copy()

        if overlapping.empty:
            out.loc[index, "detected_in_horizon"] = False
            out.loc[index, "first_alarm_time"] = pd.NaT
            out.loc[index, "lead_time_hours"] = pd.NA
            out.loc[index, "alarm_count_in_horizon"] = 0
            out.loc[index, "alarm_targets"] = ""
            out.loc[index, "alarm_episode_count_in_horizon"] = 0
            continue

        first_alarm_time = overlapping[trigger_time_col].min()
        out.loc[index, "detected_in_horizon"] = True
        out.loc[index, "first_alarm_time"] = first_alarm_time
        out.loc[index, "lead_time_hours"] = (
            event["event_start"] - first_alarm_time
        ).total_seconds() / 3600
        out.loc[index, "alarm_count_in_horizon"] = int(overlapping["alarm_points"].sum())
        out.loc[index, "alarm_episode_count_in_horizon"] = int(len(overlapping))
        out.loc[index, "alarm_targets"] = (
            "; ".join(sorted(overlapping["target"].dropna().unique()))
            if "target" in overlapping.columns else ""
        )

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


def prepare_interval_bounds(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[np.ndarray, np.ndarray]:
    valid_intervals = [
        (pd.Timestamp(interval_start), pd.Timestamp(interval_end))
        for interval_start, interval_end in intervals
        if pd.notna(interval_start) and pd.notna(interval_end)
    ]
    if not valid_intervals:
        return np.array([], dtype="datetime64[ns]"), np.array([], dtype="datetime64[ns]")

    valid_intervals.sort(key=lambda item: item[0])
    starts = np.array([item[0].to_datetime64() for item in valid_intervals], dtype="datetime64[ns]")
    ends = np.array([item[1].to_datetime64() for item in valid_intervals], dtype="datetime64[ns]")
    return starts, np.maximum.accumulate(ends)


def prepare_continuous_interval_bounds(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[np.ndarray, np.ndarray]:
    valid = sorted(
        (pd.Timestamp(start), pd.Timestamp(end)) for start, end in intervals
        if pd.notna(start) and pd.notna(end) and end > start
    )
    merged = []
    for start, end in valid:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return prepare_interval_bounds(merged)


def overlaps_prepared_intervals(
    start: pd.Timestamp,
    end: pd.Timestamp,
    bounds: tuple[np.ndarray, np.ndarray],
) -> bool:
    starts, prefix_max_ends = bounds
    if len(starts) == 0 or pd.isna(start) or pd.isna(end):
        return False

    end_value = np.datetime64(pd.Timestamp(end).to_datetime64())
    start_value = np.datetime64(pd.Timestamp(start).to_datetime64())
    candidate_count = np.searchsorted(starts, end_value, side="left")
    if candidate_count == 0:
        return False
    return bool(prefix_max_ends[candidate_count - 1] > start_value)


def overlaps_many_prepared_intervals(
    starts_to_check: pd.Series,
    ends_to_check: pd.Series,
    bounds: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    interval_starts, prefix_max_ends = bounds
    result = np.zeros(len(starts_to_check), dtype=bool)
    if len(interval_starts) == 0 or len(starts_to_check) == 0:
        return result

    valid = starts_to_check.notna().to_numpy() & ends_to_check.notna().to_numpy()
    if not valid.any():
        return result

    starts = starts_to_check.to_numpy(dtype="datetime64[ns]")
    ends = ends_to_check.to_numpy(dtype="datetime64[ns]")
    candidate_counts = np.searchsorted(interval_starts, ends[valid], side="left")
    valid_positions = np.flatnonzero(valid)
    has_candidate = candidate_counts > 0
    if not has_candidate.any():
        return result

    positions = valid_positions[has_candidate]
    counts = candidate_counts[has_candidate]
    result[positions] = prefix_max_ends[counts - 1] > starts[positions]
    return result


def contained_many_prepared_intervals(
    starts_to_check: pd.Series,
    ends_to_check: pd.Series,
    bounds: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    interval_starts, prefix_max_ends = bounds
    result = np.zeros(len(starts_to_check), dtype=bool)
    if len(interval_starts) == 0 or len(starts_to_check) == 0:
        return result

    valid = starts_to_check.notna().to_numpy() & ends_to_check.notna().to_numpy()
    if not valid.any():
        return result

    starts = starts_to_check.to_numpy(dtype="datetime64[ns]")
    ends = ends_to_check.to_numpy(dtype="datetime64[ns]")
    positions = np.flatnonzero(valid)
    candidates = np.searchsorted(interval_starts, starts[valid], side="right") - 1
    has_candidate = candidates >= 0
    result[positions[has_candidate]] = (
        prefix_max_ends[candidates[has_candidate]] >= ends[valid][has_candidate]
    )
    return result


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
    buffer_hours: int = 6,
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


def auxiliary_maintenance_intervals(
    project_root: Path,
    turbine_id: str,
    start_year: int = 2023,
    end_year: int = 2024,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    path = project_root / "data" / "raw" / "kelmarsh" / "auxiliary" / "kelmarsh_event_dataset.csv"
    if not path.exists():
        return []
    events = read_event_dataset(path)
    start = pd.Timestamp(start_year, 1, 1)
    end = pd.Timestamp(end_year + 1, 1, 1)
    turbine_number = turbine_id.rsplit("_", 1)[-1]
    selected = events[
        events["Turbine"].astype(str).str.strip().eq(turbine_number)
        & events["Category"].astype(str).str.strip().str.casefold().eq("maintenance")
        & events["Timestamp start"].lt(end)
        & events["Timestamp end"].gt(start)
    ]
    return list(zip(selected["Timestamp start"], selected["Timestamp end"]))


def operating_context_intervals(
    project_root: Path,
    turbine_id: str,
    buffer_hours: int = 6,
    start_year: int = 2023,
    end_year: int = 2024,
    target_events: pd.DataFrame | None = None,
    evaluation_cache: dict | None = None,
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    if buffer_hours not in (0, 1, 3, 6):
        raise ValueError("Environment buffer must be 0, 1, 3, or 6 hours")
    cache_key = (str(project_root), turbine_id, buffer_hours, start_year, end_year,
                 id(target_events))
    if evaluation_cache is not None and cache_key in evaluation_cache:
        return evaluation_cache[cache_key]
    flag_path = project_root / "data" / "interim" / "kelmarsh" / "flags" / f"{turbine_id.lower()}_with_flags.parquet"
    flags = intervals_from_flag_file(
        flag_path, ["in_maintenance", "in_communication", "in_curtailment"],
        start_year=start_year, end_year=end_year,
    )
    if target_events is None:
        target_events = build_target_event_summary(
            project_root, horizon_days=7, include_long_warnings=False,
        )
    selected = target_events.loc[target_events["turbine_id"].eq(turbine_id)]
    intervals = {
        **flags,
        "auxiliary_maintenance": auxiliary_maintenance_intervals(
            project_root, turbine_id, start_year, end_year,
        ),
        "non_full_performance": non_full_performance_intervals_from_status(
            project_root, turbine_id, start_year, end_year,
        ),
        "target_event": list(zip(selected["event_start"], selected["event_end"])),
        "environmental_buffer": [],
    }
    if buffer_hours:
        intervals["environmental_buffer"] = informational_environmental_spec_intervals_from_status(
            project_root, turbine_id, start_year, end_year, buffer_hours,
        )
    if evaluation_cache is not None:
        evaluation_cache[cache_key] = intervals
    return intervals


def merge_operating_intervals(
    intervals_by_source: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    intervals = sorted(
        (pd.Timestamp(start), pd.Timestamp(end))
        for group in intervals_by_source.values() for start, end in group
        if pd.notna(start) and pd.notna(end) and end > start
    )
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def longest_operational_alarm_run(
    start: pd.Timestamp,
    end: pd.Timestamp,
    non_operational_bounds: tuple[np.ndarray, np.ndarray],
    freq_minutes: int = FREQ_MINUTES,
) -> int:
    """Count the longest 10-minute alarm-point run left after non-op points are removed."""
    if pd.isna(start) or pd.isna(end) or end < start:
        return 0
    step = pd.Timedelta(minutes=freq_minutes)
    points = pd.Series(pd.date_range(start, end, freq=step))
    blocked = overlaps_many_prepared_intervals(points, points + step, non_operational_bounds)
    edges = np.diff(np.r_[0, (~blocked).astype(np.int8), 0])
    run_starts = np.flatnonzero(edges == 1)
    run_ends = np.flatnonzero(edges == -1)
    return int((run_ends - run_starts).max(initial=0))


def split_alarm_episodes_at_non_operational(
    alarm_episodes: pd.DataFrame,
    project_root: Path,
    buffer_hours: int = 6,
    min_consecutive: int = 1,
    freq_minutes: int = FREQ_MINUTES,
    evaluation_cache: dict | None = None,
) -> pd.DataFrame:
    """Keep true alarms; split other episodes into qualifying operational runs."""
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be positive")
    out = alarm_episodes.copy()
    if "alarm_points" not in out:
        out["alarm_points"] = pd.Series(dtype="int64")
    if "episode_id" not in out:
        out["episode_id"] = pd.Series(dtype="int64")
    out["source_episode_id"] = pd.Series(dtype="int64")
    if out.empty:
        out.attrs["suppressed_source_episodes"] = 0
        return out

    step = pd.Timedelta(minutes=freq_minutes)
    if evaluation_cache is not None and "target_events" in evaluation_cache:
        target_events = evaluation_cache["target_events"]
    else:
        target_events = build_target_event_summary(project_root, horizon_days=7, include_long_warnings=False)
        if evaluation_cache is not None:
            evaluation_cache["target_events"] = target_events
    out = annotate_alarm_episode_horizon_matches(out, target_events, freq_minutes=freq_minutes)
    rows = []
    for turbine_id, group in out.groupby("turbine_id", sort=False):
        sources = operating_context_intervals(
            project_root, str(turbine_id), buffer_hours=buffer_hours, target_events=target_events,
            evaluation_cache=evaluation_cache,
        )
        bounds = prepare_interval_bounds(merge_operating_intervals(sources))
        singleton_mask = (
            ~group["in_fault_horizon"].fillna(False).astype(bool)
            & group["alarm_points"].eq(1)
        )
        if min_consecutive == 1 and singleton_mask.any():
            singletons = group.loc[singleton_mask]
            blocked = overlaps_many_prepared_intervals(
                singletons["start_time"], singletons["end_time"] + step, bounds,
            )
            kept = singletons.loc[~blocked].copy()
            kept["source_episode_id"] = kept.index.astype(int)
            rows.extend(kept.to_dict("records"))
        for source_id, row in group.loc[~singleton_mask].iterrows():
            if bool(row["in_fault_horizon"]):
                preserved = row.to_dict()
                preserved["source_episode_id"] = int(source_id)
                rows.append(preserved)
                continue
            points = pd.date_range(row["start_time"], row["end_time"], freq=step)
            if points.empty:
                continue
            starts = pd.Series(points)
            blocked = overlaps_many_prepared_intervals(starts, starts + step, bounds)
            edges = np.diff(np.r_[0, (~blocked).astype(np.int8), 0])
            for first, last in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if last - first < min_consecutive:
                    continue
                fragment = row.to_dict()
                fragment.update(start_time=points[first], end_time=points[last - 1],
                                alarm_points=int(last - first), source_episode_id=int(source_id))
                rows.append(fragment)

    result = pd.DataFrame(rows, columns=out.columns)
    if not result.empty:
        result["episode_id"] = range(len(result))
    result.attrs["suppressed_source_episodes"] = len(out) - result["source_episode_id"].nunique()
    return result


def annotate_alarm_episode_operating_context(
    alarm_episodes: pd.DataFrame,
    project_root: Path,
    freq_minutes: int = FREQ_MINUTES,
    buffer_hours: int = 6,
    non_operational_rule: str = "overlap",
    min_operational_consecutive: int = 1,
    evaluation_cache: dict | None = None,
) -> pd.DataFrame:
    if non_operational_rule not in {"overlap", "contained", "remaining_consecutive"}:
        raise ValueError("Unknown non_operational_rule")
    if min_operational_consecutive < 1:
        raise ValueError("min_operational_consecutive must be positive")
    if non_operational_rule == "remaining_consecutive":
        fragments = split_alarm_episodes_at_non_operational(
            alarm_episodes, project_root, buffer_hours=buffer_hours,
            min_consecutive=min_operational_consecutive, freq_minutes=freq_minutes,
            evaluation_cache=evaluation_cache,
        )
        suppressed = fragments.attrs["suppressed_source_episodes"]
        events = (evaluation_cache["target_events"] if evaluation_cache is not None
                  else build_target_event_summary(project_root, horizon_days=7, include_long_warnings=False))
        fragments = annotate_alarm_episode_horizon_matches(fragments, events, freq_minutes=freq_minutes)
        result = annotate_alarm_episode_operating_context(
            fragments, project_root, freq_minutes=freq_minutes, buffer_hours=buffer_hours,
            non_operational_rule="overlap",
            evaluation_cache=evaluation_cache,
        )
        result.attrs["suppressed_source_episodes"] = suppressed
        return result
    out = alarm_episodes.copy()
    context_flags = {
        "in_maintenance_interval": "in_maintenance",
        "in_communication_interval": "in_communication",
        "in_curtailment_interval": "in_curtailment",
        "in_auxiliary_maintenance_interval": "auxiliary_maintenance",
        "in_target_event_interval": "target_event",
        "in_non_full_performance_interval": "non_full_performance",
        "in_information_environmental_buffer_interval": "environmental_buffer",
    }
    for output_col in context_flags:
        out[output_col] = False
    out["overlaps_non_operational_interval"] = False
    out["matches_non_operational_rule"] = False
    out["max_operational_consecutive"] = pd.NA
    out["in_non_operational_interval"] = False
    out["is_operational_false_alarm"] = False

    if out.empty:
        return out

    out["start_time"] = pd.to_datetime(out["start_time"], errors="coerce")
    out["end_time"] = pd.to_datetime(out["end_time"], errors="coerce")
    if evaluation_cache is not None and "target_events" in evaluation_cache:
        target_events = evaluation_cache["target_events"]
    else:
        target_events = build_target_event_summary(project_root, horizon_days=7, include_long_warnings=False)
        if evaluation_cache is not None:
            evaluation_cache["target_events"] = target_events

    for turbine_id, idx in out.groupby("turbine_id", sort=False).groups.items():
        intervals_by_source = operating_context_intervals(
            project_root, str(turbine_id), buffer_hours=buffer_hours, target_events=target_events,
            evaluation_cache=evaluation_cache,
        )
        bounds_by_source = {key: prepare_interval_bounds(value) for key, value in intervals_by_source.items()}
        all_non_operational_bounds = prepare_interval_bounds(merge_operating_intervals(intervals_by_source))

        idx_list = list(idx)
        starts = out.loc[idx_list, "start_time"]
        ends = out.loc[idx_list, "end_time"] + pd.Timedelta(minutes=freq_minutes)

        for output_col, source in context_flags.items():
            out.loc[idx_list, output_col] = overlaps_many_prepared_intervals(
                starts, ends, bounds_by_source[source],
            )
        overlaps = overlaps_many_prepared_intervals(
            starts, ends, all_non_operational_bounds,
        )
        out.loc[idx_list, "overlaps_non_operational_interval"] = overlaps
        classify = (
            overlaps_many_prepared_intervals
            if non_operational_rule == "overlap"
            else contained_many_prepared_intervals
        )
        out.loc[idx_list, "matches_non_operational_rule"] = classify(
            starts, ends, all_non_operational_bounds,
        )

    out["in_non_operational_interval"] = (
        out["matches_non_operational_rule"]
        & ~out["in_fault_horizon"].fillna(False).astype(bool)
    )
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
    if "target" not in alarm_episodes.columns and not alarm_points.empty:
        alarm_points = alarm_points.drop_duplicates(["turbine_id", "Date and time"])
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

    suffix = f"q{int(args.quantile * 1000):03d}_c{args.min_consecutive}_turbine"
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
    alarm_episodes = build_turbine_alarm_episodes(detections)
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
