"""Load the no-Warning thesis results used by the detection viewer."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main_detect_residual_baseline import (
    annotate_alarm_episode_horizon_matches,
    annotate_alarm_episode_operating_context,
    build_target_event_summary,
    fuse_alarm_episodes,
    informational_environmental_spec_intervals_from_status,
    merge_operating_intervals,
    operating_context_intervals,
    update_event_summary_from_alarm_episodes,
)


EXPECTED_EVENTS = 459
METHOD_LABELS = {
    "scc": "SCC",
    "cusum_two_sided": "Two-sided CUSUM",
    "cusum_absolute": "One-sided CUSUM",
    "bcad": "BCAD",
}
SCC_GRID = "scc_grid_abs_meanstd_c6_k0-10_step0p5_split6h_no_warning_turbine.csv"
CUSUM_GRIDS = {
    "cusum_two_sided": "cusum_grid_fixed_meanstd_no_consecutive_k0-1_h20-200_split6h_no_warning_turbine.csv",
    "cusum_absolute": "cusum_grid_absolute_fixed_meanstd_no_consecutive_k0-1_h20-200_split6h_no_warning_turbine.csv",
}
SCC_SETTINGS = {"abs_meanstd_k3_c6": 3.0, "abs_meanstd_k3p5_c6": 3.5,
                "abs_meanstd_k4_c6": 4.0, "abs_meanstd_k6_c6": 6.0,
                "abs_meanstd_k5_c6_split6h_no_warning_turbine": 5.0}
CUSUM_SETTINGS = {
    "cusum_two_sided": {"twosided_std_k0_h140_reset": (0.0, 140.0),
                        "twosided_std_k0_h180_reset_split6h_no_warning_turbine": (0.0, 180.0)},
    "cusum_absolute": {"absolute_fixed_meanstd_k0p8_h80_reset": (0.8, 80.0),
                       "absolute_fixed_meanstd_k0p8_h180_reset_split6h_no_warning_turbine": (0.8, 180.0)},
}
REVISED_BCAD = {
    "foldedmle_samecenterabs_w72_q990_vm5_c72": "folded",
    "abslog_w72_q990_vm5_c72": "gaussian",
    "weibullmle_foldedcalib2021_2022_w72_q990_c72": "maximum",
    "weibulljointmle_foldedcalib2021_2022_w72_q990_c72": "joint",
}
DEFAULT_SETTINGS = {
    "scc": "abs_meanstd_k5_c6_split6h_no_warning_turbine",
    "cusum_two_sided": "twosided_std_k0_h180_reset_split6h_no_warning_turbine",
    "cusum_absolute": "absolute_fixed_meanstd_k0p8_h180_reset_split6h_no_warning_turbine",
    "bcad": "weibullmle_foldedcalib2021_2022_w72_q990_c72",
}


def min_consecutive_for_setting(method: str, setting: str) -> int:
    if method.startswith("cusum"):
        return 1  # CUSUM has no consecutive filter; a single surviving alarm point is enough.
    match = re.search(r"(?:^|_)c(\d+)(?:_|$)", setting)
    if match is None:
        raise ValueError(f"Cannot determine min_consecutive for {method}/{setting}")
    return int(match.group(1))


def experiment_dir(project_root: Path, run_id: str) -> Path:
    path = project_root / "results" / "kelmarsh" / "experiments" / run_id
    if not (path / "metadata.json").is_file():
        raise FileNotFoundError(f"Experiment metadata not found: {path}")
    return path


def list_runs(project_root: Path) -> list[str]:
    base = project_root / "results" / "kelmarsh" / "experiments"
    return sorted(p.name for p in base.iterdir() if (p / "metadata.json").is_file()) if base.exists() else []


def _episode_path(run_dir: Path, method: str, setting: str) -> Path:
    if method == "scc":
        return run_dir / f"scc_alarm_episodes_{setting}.csv"
    if method == "cusum_two_sided":
        if setting.endswith("_split6h_no_warning_turbine"):
            return run_dir / f"cusum_alarm_episodes_{setting}.csv"
        return run_dir / "no_warning_lead_time" / f"cusum_alarm_episodes_{setting}.csv"
    if method == "cusum_absolute":
        return run_dir / f"cusum_alarm_episodes_{setting}.csv"
    source_setting = setting.replace("_c72", "_c6") if setting in REVISED_BCAD else setting
    return run_dir / "no_warning_bcad" / f"bcad_alarm_episodes_{source_setting}.csv"


def available_settings(project_root: Path, run_id: str, method: str) -> list[str]:
    run_dir = experiment_dir(project_root, run_id)
    if method == "bcad":
        folder = run_dir / "no_warning_bcad"
        old = {
            setting for path in folder.glob("bcad_performance_*.json")
            if (setting := path.stem.removeprefix("bcad_performance_"))
            and (folder / f"bcad_event_summary_{setting}.csv").is_file()
            and _episode_path(run_dir, method, setting).is_file()
        } if folder.exists() else set()
        scan_dir = project_root / "results" / "kelmarsh" / "thesis_revision_split6h" / "bcad_window_scan"
        revised = {setting for setting, mode in REVISED_BCAD.items()
                   if (scan_dir / f"{mode}_w36-1080_c72.csv").is_file()
                   and _episode_path(run_dir, method, setting).is_file()}
        return sorted(old | revised)
    settings = SCC_SETTINGS if method == "scc" else CUSUM_SETTINGS.get(method, {})
    grid = SCC_GRID if method == "scc" else CUSUM_GRIDS.get(method)
    if grid is None or not (run_dir / grid).is_file():
        return []
    return [setting for setting in settings if _episode_path(run_dir, method, setting).is_file()]


def _grid_row(run_dir: Path, method: str, setting: str) -> pd.Series:
    if method == "scc":
        grid = pd.read_csv(run_dir / SCC_GRID)
        rows = grid.loc[grid["k"].eq(SCC_SETTINGS[setting])]
    else:
        grid = pd.read_csv(run_dir / CUSUM_GRIDS[method])
        k, h = CUSUM_SETTINGS[method][setting]
        rows = grid.loc[grid["k"].eq(k) & grid["h"].eq(h)]
    if len(rows) != 1:
        raise ValueError(f"Expected one no-Warning grid row for {method}/{setting}, found {len(rows)}")
    return rows.iloc[0]


def load_result(project_root: Path, run_id: str, method: str, setting: str,
                buffer_hours: int = 6, non_operational_rule: str = "remaining_consecutive") -> tuple[dict, dict, pd.DataFrame]:
    if setting not in available_settings(project_root, run_id, method):
        raise ValueError(f"No complete no-Warning result for {method}/{setting}")
    run_dir = experiment_dir(project_root, run_id)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if method == "bcad" and setting in REVISED_BCAD:
        mode = REVISED_BCAD[setting]
        scan = project_root / "results" / "kelmarsh" / "thesis_revision_split6h" / "bcad_window_scan" / f"{mode}_w36-1080_c72.csv"
        row = pd.read_csv(scan).loc[lambda frame: frame["window_size"].eq(72)].iloc[0]
        metrics = {
            "total_events": int(row["total_events"]),
            "detected_events": int(row["detected_events"]),
            "missed_events": int(row["missed_events"]),
        }
        threshold_path = run_dir / "no_warning_bcad" / f"bcad_thresholds_{setting.replace('_c72', '_c6')}.csv"
    elif method == "bcad":
        folder = run_dir / "no_warning_bcad"
        result = json.loads((folder / f"bcad_performance_{setting}.json").read_text(encoding="utf-8"))
        if result.get("target_event_definition") != "fault_corrective_maintenance_selected_stop_no_warning":
            raise ValueError(f"BCAD result does not declare the no-Warning target definition: {setting}")
        event = result["event_level"]
        episode = result["alarm_episode_level"]
        metrics = {
            "total_events": event["total_events"],
            "detected_events": event["detected_events"],
            "missed_events": event["missed_events"],
            "event_recall": event["detection_rate"],
            "total_alarm_episodes": episode["total_alarm_episodes"],
            "true_alarm_episodes": episode["true_alarm_episodes"],
            "operational_false_alarm_episodes": episode["operational_false_alarm_episodes"],
            "operational_precision": result["precision"],
            "operational_false_alarm_rate": episode["operational_false_alarm_rate"],
            "f1": result["f1"],
        }
        threshold_path = folder / f"bcad_thresholds_{setting}.csv"
    else:
        row = _grid_row(run_dir, method, setting)
        metrics = {
            "total_events": int(row["total_events"]),
            "detected_events": int(row["detected_events"]),
            "missed_events": int(row["total_events"] - row["detected_events"]),
            "event_recall": float(row["recall"]),
            "total_alarm_points": int(row["total_alarm_points"]),
            "total_alarm_episodes": int(row["total_alarm_episodes"]),
            "true_alarm_episodes": int(row["true_alarm_episodes"]),
            "operational_false_alarm_episodes": int(row["operational_false_alarm_episodes"]),
            "operational_precision": float(row["precision"]),
            "operational_false_alarm_rate": float(row["operational_false_alarm_rate"]),
            "f1": float(row["f1_score"]),
        }
        prefix = "scc" if method == "scc" else "cusum"
        threshold_path = run_dir / f"{prefix}_thresholds_{setting}.csv"
    if metrics["total_events"] != EXPECTED_EVENTS:
        raise ValueError(f"Expected {EXPECTED_EVENTS} no-Warning events, found {metrics['total_events']}")
    episodes = _fused_episodes(project_root, run_id, method, setting, buffer_hours, non_operational_rule)
    events = update_event_summary_from_alarm_episodes(_target_events(project_root), episodes)
    detected = int(events["detected_in_horizon"].sum())
    true_count = int(episodes["in_fault_horizon"].sum())
    false_count = int(episodes["is_operational_false_alarm"].sum())
    ignored_count = int(episodes["in_non_operational_interval"].sum())
    if non_operational_rule == "remaining_consecutive":
        ignored_count = int(episodes.attrs.get("suppressed_source_episodes", 0))
    precision = true_count / (true_count + false_count) if true_count + false_count else 0.0
    recall = detected / EXPECTED_EVENTS
    metrics.update({
        "detected_events": detected,
        "missed_events": EXPECTED_EVENTS - detected,
        "event_recall": recall,
        "total_alarm_episodes": len(episodes),
        "true_alarm_episodes": true_count,
        "operational_false_alarm_episodes": false_count,
        "ignored_non_operational_episodes": ignored_count,
        "operational_precision": precision,
        "operational_false_alarm_rate": false_count / (true_count + false_count) if true_count + false_count else 0.0,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    })
    if non_operational_rule == "remaining_consecutive":
        metrics["min_operational_consecutive"] = min_consecutive_for_setting(method, setting)
        metrics["suppressed_source_episodes"] = ignored_count
    metrics.pop("total_alarm_points", None)
    thresholds = pd.read_csv(threshold_path) if threshold_path.is_file() else pd.DataFrame()
    return metadata, metrics, thresholds


@lru_cache(maxsize=4)
def _target_events(project_root: Path) -> pd.DataFrame:
    events = build_target_event_summary(project_root, horizon_days=7, include_long_warnings=False)
    if len(events) != EXPECTED_EVENTS:
        raise ValueError(f"Expected {EXPECTED_EVENTS} no-Warning target events, found {len(events)}")
    if events["event_source"].fillna("").str.contains("status_long_warning_target").any():
        raise ValueError("Status Warning is present among prediction target events")
    return events


@lru_cache(maxsize=32)
def _fused_episodes(project_root: Path, run_id: str, method: str, setting: str,
                    buffer_hours: int = 6, non_operational_rule: str = "remaining_consecutive") -> pd.DataFrame:
    columns = ["turbine_id", "target", "start_time", "end_time", "alarm_points"] if (
        method == "bcad" and setting in REVISED_BCAD) else ["turbine_id", "start_time", "end_time"]
    source = pd.read_csv(
        _episode_path(experiment_dir(project_root, run_id), method, setting),
        usecols=columns,
        parse_dates=["start_time", "end_time"],
    )
    if method == "bcad" and setting in REVISED_BCAD:
        source = source.loc[source["alarm_points"].ge(67)].copy()
        source["start_time"] += pd.Timedelta(minutes=660)
    episodes = fuse_alarm_episodes(source)
    episodes = annotate_alarm_episode_horizon_matches(episodes, _target_events(project_root))
    episodes = annotate_alarm_episode_operating_context(
        episodes, project_root, buffer_hours=buffer_hours,
        non_operational_rule=non_operational_rule,
        min_operational_consecutive=min_consecutive_for_setting(method, setting),
    )
    return episodes


def load_timeline(project_root: Path, run_id: str, method: str, setting: str,
                  turbine: str, year: int, buffer_hours: int = 6,
                  non_operational_rule: str = "remaining_consecutive") -> tuple[pd.DataFrame, pd.DataFrame]:
    if year not in (2023, 2024) or turbine not in {f"Kelmarsh_{i}" for i in range(1, 7)}:
        raise ValueError("Select one Kelmarsh turbine and either 2023 or 2024")
    if setting not in available_settings(project_root, run_id, method):
        raise ValueError(f"No complete no-Warning result for {method}/{setting}")
    start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
    turbine_events = _target_events(project_root)
    turbine_events = turbine_events.loc[turbine_events["turbine_id"].eq(turbine)].copy()
    events = turbine_events.loc[turbine_events["event_start"].lt(end)
                                & turbine_events["event_end"].gt(start)].copy()
    episodes = _fused_episodes(project_root, run_id, method, setting, buffer_hours, non_operational_rule)
    episodes = episodes.loc[episodes["turbine_id"].eq(turbine)
                            & episodes["start_time"].lt(end)
                            & (episodes["end_time"] + pd.Timedelta(minutes=10)).gt(start)].copy()
    return events, episodes


def merge_alarm_intervals(episodes: pd.DataFrame, start: pd.Timestamp,
                          end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    intervals = sorted(
        (max(row.start_time, start), min(row.end_time + pd.Timedelta(minutes=10), end))
        for row in episodes.itertuples(index=False)
        if pd.notna(row.start_time) and pd.notna(row.end_time)
        and max(row.start_time, start) < min(row.end_time + pd.Timedelta(minutes=10), end)
    )
    merged = []
    for interval_start, interval_end in intervals:
        if merged and interval_start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval_end))
        else:
            merged.append((interval_start, interval_end))
    return merged


def merge_intervals(intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
                    start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    clipped = sorted((max(a, start), min(b, end)) for a, b in intervals
                     if pd.notna(a) and pd.notna(b) and max(a, start) < min(b, end))
    merged = []
    for a, b in clipped:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def load_non_operational_intervals(project_root: Path, turbine: str, year: int,
                                   buffer_hours: int = 6
                                   ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
    sources = operating_context_intervals(
        project_root, turbine, buffer_hours=buffer_hours, target_events=_target_events(project_root),
    )
    return merge_intervals(merge_operating_intervals(sources), start, end)


@lru_cache(maxsize=2)
def load_coverage_comparison(project_root: Path) -> pd.DataFrame:
    """Share of observed 2023-2024 SCADA timestamps excluded at each buffer width."""
    counts = {hours: 0 for hours in (0, 1, 3, 6)}
    total = 0
    events = _target_events(project_root)
    for turbine in (f"Kelmarsh_{i}" for i in range(1, 7)):
        flag_path = project_root / "data" / "interim" / "kelmarsh" / "flags" / f"{turbine.lower()}_with_flags.parquet"
        timestamps = pd.read_parquet(flag_path, columns=["Date and time"])["Date and time"]
        timestamps = pd.to_datetime(timestamps, errors="coerce")
        timestamps = timestamps[timestamps.ge("2023-01-01") & timestamps.lt("2025-01-01")].dropna()
        values = timestamps.to_numpy(dtype="datetime64[ns]")
        total += len(values)
        base = operating_context_intervals(
            project_root, turbine, buffer_hours=0, target_events=events,
        )
        for hours in counts:
            sources = base.copy()
            if hours:
                sources["environmental_buffer"] = informational_environmental_spec_intervals_from_status(
                    project_root, turbine, buffer_hours=hours,
                )
            intervals = merge_operating_intervals(sources)
            starts = np.array([start.to_datetime64() for start, _ in intervals], dtype="datetime64[ns]")
            ends = np.array([end.to_datetime64() for _, end in intervals], dtype="datetime64[ns]")
            if len(starts):
                positions = np.searchsorted(starts, values, side="right") - 1
                valid = positions >= 0
                counts[hours] += int(np.count_nonzero(values[valid] < ends[positions[valid]]))
    return pd.DataFrame({
        "buffer_hours": list(counts),
        "non_operational_points": list(counts.values()),
        "observed_points": total,
        "coverage": [count / total for count in counts.values()],
    })


def plot_coverage_comparison(comparison: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor="white")
    labels = ["None", "±1 h", "±3 h", "±6 h"]
    colors = ["#b9c3cb", "#4c91a8", "#287d83", "#165b60"]
    bars = ax.bar(labels, 100 * comparison["coverage"], color=colors, width=0.65)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylim(0, max(65, 100 * comparison["coverage"].max() + 7))
    ax.set_ylabel("Non-operational SCADA timestamps (%)")
    ax.set_xlabel("Informational / environmental buffer")
    ax.set_title("2023–2024 | All six Kelmarsh turbines")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def visible_alarm_intervals(episodes: pd.DataFrame,
                            start: pd.Timestamp, end: pd.Timestamp
                            ) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]],
                                       list[tuple[pd.Timestamp, pd.Timestamp]]]:
    if not {"in_fault_horizon", "in_non_operational_interval"}.issubset(episodes.columns):
        raise ValueError("Timeline episodes need turbine-level TA and operating-context labels")
    operational = episodes.loc[~episodes["in_non_operational_interval"].fillna(False).astype(bool)]
    is_ta = operational["in_fault_horizon"].fillna(False).astype(bool)
    lanes = []
    for selected in (operational.loc[is_ta], operational.loc[~is_ta]):
        lanes.append(merge_alarm_intervals(selected, start, end))
    return lanes[0], lanes[1]


def plot_timeline(events: pd.DataFrame, episodes: pd.DataFrame, turbine: str,
                  year: int, method: str, setting: str,
                  non_operational: list[tuple[pd.Timestamp, pd.Timestamp]],
                  show_non_operational: bool = True, buffer_hours: int = 6,
                  non_operational_rule: str = "overlap") -> plt.Figure:
    start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
    fig, ax = plt.subplots(figsize=(16, 4.2), facecolor="white")
    ax.set_facecolor("white")

    def bars(frame: pd.DataFrame, start_col: str, end_col: str) -> list[tuple[float, float]]:
        spans = []
        for interval_start, interval_end in zip(frame[start_col], frame[end_col]):
            if pd.isna(interval_start) or pd.isna(interval_end):
                continue
            clipped_start = max(interval_start, start)
            clipped_end = min(interval_end, end)
            if clipped_end > clipped_start:
                spans.append((mdates.date2num(clipped_start),
                              (clipped_end - clipped_start).total_seconds() / 86400))
        return spans

    target_bars = bars(events, "event_start", "event_end")
    if target_bars:
        ax.broken_barh(target_bars, (2.14, 0.72), facecolors="#7046a4", alpha=0.9)
    if show_non_operational:
        shadow_bars = [
            (mdates.date2num(a), (b - a).total_seconds() / 86400)
            for a, b in merge_intervals(non_operational, start, end)
        ]
        if shadow_bars:
            ax.broken_barh(shadow_bars, (2.08, 0.84), facecolors="#606b74", alpha=0.38)
    ta_intervals, fa_intervals = visible_alarm_intervals(episodes, start, end)
    for intervals, y, color in [(ta_intervals, 1.14, "#25805a"), (fa_intervals, 0.14, "#c84b45")]:
        alarm_bars = [
            (mdates.date2num(interval_start), (interval_end - interval_start).total_seconds() / 86400)
            for interval_start, interval_end in intervals
        ]
        if alarm_bars:
            ax.broken_barh(alarm_bars, (y, 0.72), facecolors=color, alpha=0.9)

    ax.set_yticks([2.5, 1.5, 0.5])
    ax.set_yticklabels(["Target event", "TA", "FA"])
    ax.set_ylim(0, 3)
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlabel("Time")
    ax.set_title(
        f"{turbine} | {year} | {METHOD_LABELS[method]} | {setting} | "
        f"env buffer ±{buffer_hours} h | {non_operational_rule}"
        + (f" | split c={min_consecutive_for_setting(method, setting)}"
           if non_operational_rule == "remaining_consecutive" else ""), fontsize=11,
    )
    fig.tight_layout()
    return fig
