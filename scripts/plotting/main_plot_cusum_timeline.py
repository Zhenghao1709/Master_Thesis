from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main_detect_residual_baseline import (
    build_target_event_summary,
    find_experiment,
    informational_environmental_spec_intervals_from_status,
    intervals_from_flag_file,
    non_full_performance_intervals_from_status,
)
from src.config.kelmarsh_config import TARGET_COLS
from src.detection.residuals import resolve_prediction_column


DEFAULT_RUN_ID = "all6_multi3_seq12_h64_l1_bs64_lr1e-03_wd0_do0_e60_p8_seed42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a CUSUM accumulation timeline for one selected setting."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--turbine", default="Kelmarsh_1")
    parser.add_argument("--target", default=TARGET_COLS[0])
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--month", type=int, default=1, help="Month to plot. Use 0 for the full year.")
    parser.add_argument("--reference-value", "--k", dest="k", type=float, default=0.0)
    parser.add_argument("--decision-threshold", "--h", dest="h", type=float, default=140.0)
    parser.add_argument("--scale-type", choices=["std"], default="std")
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument(
        "--hide-non-operation-alarms",
        action="store_true",
        help="Hide alarm markers that occur inside non-operation intervals. CUSUM computation is unchanged.",
    )
    parser.add_argument(
        "--informational-env-buffer-hours",
        type=int,
        default=6,
        help=(
            "Buffer in hours around Informational/Information + Out of Environmental Specification "
            "status intervals that should be treated as non-operation."
        ),
    )
    return parser.parse_args()


def format_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return label[:48] or "target"


def load_metadata(project_root: Path, run_id: str) -> tuple[Path, dict]:
    result_dir, _, metadata = find_experiment(project_root, run_id)
    return result_dir, metadata


def resolve_path(project_root: Path, rel_path: str) -> Path:
    return project_root / rel_path


def fit_reference_from_validation(project_root: Path, metadata: dict, target: str) -> tuple[float, float, str]:
    val_path = resolve_path(project_root, metadata["paths"]["predictions"])
    header = pd.read_csv(val_path, nrows=0).columns.tolist()
    error_col = resolve_prediction_column(pd.DataFrame(columns=header), "error", target)
    values = pd.to_numeric(pd.read_csv(val_path, usecols=[error_col])[error_col], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"No validation error values found for target: {target}")
    center = float(values.mean())
    scale = float(values.std(ddof=0))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError(f"Validation error std is zero or invalid for target: {target}")
    return center, scale, error_col


def load_test_prediction_columns(project_root: Path, metadata: dict, target: str) -> pd.DataFrame:
    test_path = resolve_path(project_root, metadata["paths"]["test_predictions"])
    header = pd.read_csv(test_path, nrows=0).columns.tolist()
    header_df = pd.DataFrame(columns=header)
    true_col = resolve_prediction_column(header_df, "y_true", target)
    pred_col = resolve_prediction_column(header_df, "y_pred", target)
    error_col = resolve_prediction_column(header_df, "error", target)
    usecols = ["turbine_id", "segment_id", "Date and time", true_col, pred_col, error_col]
    out = pd.read_csv(test_path, usecols=usecols)
    out = out.rename(columns={true_col: "measured", pred_col: "predicted", error_col: "error"})
    out["Date and time"] = pd.to_datetime(out["Date and time"], errors="coerce")
    return out


def compute_cusum_trajectory(
    predictions: pd.DataFrame,
    turbine: str,
    center: float,
    scale: float,
    k: float,
    h: float,
    reset_on_alarm: bool = True,
) -> pd.DataFrame:
    part = predictions.loc[predictions["turbine_id"] == turbine].copy()
    if part.empty:
        raise ValueError(f"No test prediction rows found for turbine: {turbine}")

    part["standardized_error"] = (pd.to_numeric(part["error"], errors="coerce") - center) / scale
    part = part.sort_values(["segment_id", "Date and time"]).reset_index(drop=True)

    frames = []
    for _, group in part.groupby("segment_id", sort=False):
        group = group.copy()
        positive_sum = 0.0
        negative_sum = 0.0
        pos_values = np.zeros(len(group), dtype=float)
        neg_values = np.zeros(len(group), dtype=float)
        anomaly_values = np.zeros(len(group), dtype=bool)
        reset_values = np.zeros(len(group), dtype=bool)

        for i, z in enumerate(group["standardized_error"].to_numpy()):
            if pd.isna(z):
                positive_sum = 0.0
                negative_sum = 0.0
                reset_values[i] = True
                continue

            positive_sum = max(0.0, positive_sum + float(z) - k)
            negative_sum = min(0.0, negative_sum + float(z) + k)
            is_anomaly = positive_sum > h or negative_sum < -h

            pos_values[i] = positive_sum
            neg_values[i] = negative_sum
            anomaly_values[i] = is_anomaly

            if is_anomaly and reset_on_alarm:
                positive_sum = 0.0
                negative_sum = 0.0
                reset_values[i] = True

        group["cusum_positive"] = pos_values
        group["cusum_negative"] = neg_values
        group["is_anomaly"] = anomaly_values
        group["reset_after_point"] = reset_values
        frames.append(group)

    return pd.concat(frames, ignore_index=True).sort_values("Date and time").reset_index(drop=True)


def filter_period(trajectory: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    start = pd.Timestamp(year=year, month=1 if month == 0 else month, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1) if month == 0 else start + pd.offsets.MonthBegin(1)
    out = trajectory.loc[
        (trajectory["Date and time"] >= start) & (trajectory["Date and time"] < end)
    ].copy()
    if out.empty:
        period_label = str(year) if month == 0 else f"{year}-{month:02d}"
        raise ValueError(f"No trajectory rows found for {period_label}")
    return out


def load_target_horizon_intervals(
    project_root: Path,
    turbine: str,
    year: int,
    month: int,
    horizon_days: int = 7,
) -> pd.DataFrame:
    plot_start = pd.Timestamp(year=year, month=1 if month == 0 else month, day=1)
    plot_end = pd.Timestamp(year=year + 1, month=1, day=1) if month == 0 else plot_start + pd.offsets.MonthBegin(1)
    events = build_target_event_summary(project_root, horizon_days=horizon_days)
    if events.empty:
        return pd.DataFrame(columns=["horizon_start", "event_start", "event_end", "event_type"])

    events = events.loc[events["turbine_id"] == turbine].copy()
    for col in ["horizon_start", "event_start", "event_end"]:
        events[col] = pd.to_datetime(events[col], errors="coerce")

    events = events.loc[
        events["horizon_start"].notna()
        & events["event_start"].notna()
        & (events["horizon_start"] < plot_end)
        & (events["event_start"] >= plot_start)
    ].copy()
    return events.sort_values("event_start").reset_index(drop=True)


def merge_and_clip_intervals(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    plot_start: pd.Timestamp,
    plot_end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    clipped = []
    for start, end in intervals:
        start = pd.to_datetime(start, errors="coerce")
        end = pd.to_datetime(end, errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        start = max(start, plot_start)
        end = min(end, plot_end)
        if start < end:
            clipped.append((start, end))

    if not clipped:
        return []

    clipped.sort(key=lambda item: item[0])
    merged = [clipped[0]]
    for start, end in clipped[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def load_non_operation_intervals(
    project_root: Path,
    turbine: str,
    year: int,
    month: int,
    informational_env_buffer_hours: int = 6,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    plot_start = pd.Timestamp(year=year, month=1 if month == 0 else month, day=1)
    plot_end = pd.Timestamp(year=year + 1, month=1, day=1) if month == 0 else plot_start + pd.offsets.MonthBegin(1)

    flag_cols = [
        "in_manual_event",
        "in_event",
        "in_maintenance",
        "in_communication",
        "in_curtailment",
    ]
    flag_path = project_root / "data" / "interim" / "kelmarsh" / "flags" / f"{turbine.lower()}_with_flags.parquet"
    intervals_by_flag = intervals_from_flag_file(flag_path, flag_cols, start_year=year, end_year=year)

    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for flag_col in flag_cols:
        intervals.extend(intervals_by_flag.get(flag_col, []))
    intervals.extend(non_full_performance_intervals_from_status(project_root, turbine, start_year=year, end_year=year))
    intervals.extend(
        informational_environmental_spec_intervals_from_status(
            project_root,
            turbine,
            start_year=year,
            end_year=year,
            buffer_hours=informational_env_buffer_hours,
        )
    )
    return merge_and_clip_intervals(intervals, plot_start, plot_end)


def shade_non_operation_intervals(
    axes: list[plt.Axes],
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> None:
    for start, end in intervals:
        for ax in axes:
            ax.axvspan(
                start,
                end,
                color="#8d99ae",
                alpha=0.18,
                linewidth=0,
                label="non-operation interval" if ax is axes[0] else None,
            )


def mark_non_operation_points(
    period: pd.DataFrame,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    out = period.copy()
    out["in_non_operation_interval"] = False
    if out.empty or not intervals:
        return out

    time = out["Date and time"]
    mask = pd.Series(False, index=out.index)
    for start, end in intervals:
        mask |= (time >= start) & (time < end)
    out["in_non_operation_interval"] = mask
    return out


def mark_target_horizon_points(
    period: pd.DataFrame,
    horizon_intervals: pd.DataFrame,
) -> pd.DataFrame:
    out = period.copy()
    out["in_target_horizon"] = False
    if out.empty or horizon_intervals.empty:
        return out

    time = out["Date and time"]
    mask = pd.Series(False, index=out.index)
    for _, row in horizon_intervals.iterrows():
        start = pd.to_datetime(row.get("horizon_start"), errors="coerce")
        end = pd.to_datetime(row.get("event_start"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        mask |= (time >= start) & (time < end)
    out["in_target_horizon"] = mask
    return out


def add_alarm_type(period: pd.DataFrame) -> pd.DataFrame:
    out = period.copy()
    out["alarm_type"] = "not_alarm"
    alarm_mask = out["is_anomaly"].fillna(False).astype(bool)
    target_mask = out.get("in_target_horizon", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    nonop_mask = out.get("in_non_operation_interval", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    out.loc[alarm_mask & target_mask, "alarm_type"] = "TA"
    out.loc[alarm_mask & ~target_mask & nonop_mask, "alarm_type"] = "ignored_non_operation"
    out.loc[alarm_mask & ~target_mask & ~nonop_mask, "alarm_type"] = "FA"
    return out


def shade_target_horizons(
    axes: list[plt.Axes],
    horizon_intervals: pd.DataFrame,
    plot_start: pd.Timestamp,
    plot_end: pd.Timestamp,
) -> None:
    if horizon_intervals.empty:
        return

    for _, row in horizon_intervals.iterrows():
        horizon_start = max(row["horizon_start"], plot_start)
        event_start = min(row["event_start"], plot_end)
        if pd.isna(horizon_start) or pd.isna(event_start) or horizon_start >= event_start:
            continue
        for ax in axes:
            ax.axvspan(
                horizon_start,
                event_start,
                color="#2ca02c",
                alpha=0.13,
                linewidth=0,
                label="target prediction horizon" if ax is axes[0] else None,
            )
            ax.axvline(event_start, color="#1b7837", linestyle="--", linewidth=1.0, alpha=0.85)


def plot_cusum_timeline(
    period: pd.DataFrame,
    horizon_intervals: pd.DataFrame,
    non_operation_intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    output_path: Path,
    run_id: str,
    turbine: str,
    target: str,
    year: int,
    month: int,
    k: float,
    h: float,
    center: float,
    scale: float,
    reset_on_alarm: bool,
    hide_non_operation_alarms: bool,
    informational_env_buffer_hours: int,
) -> None:
    plot_start = pd.Timestamp(year=year, month=1 if month == 0 else month, day=1)
    plot_end = pd.Timestamp(year=year + 1, month=1, day=1) if month == 0 else plot_start + pd.offsets.MonthBegin(1)
    period_label = str(year) if month == 0 else f"{year}-{month:02d}"
    fig, ax = plt.subplots(figsize=(16, 6.4), constrained_layout=True, facecolor="white")

    time = period["Date and time"]
    alarm = period.loc[period["is_anomaly"]]
    reset = period.loc[period["reset_after_point"]]
    visible_alarm = alarm
    visible_reset = reset
    if hide_non_operation_alarms and "in_non_operation_interval" in period.columns:
        visible_alarm = alarm.loc[~alarm["in_non_operation_interval"].fillna(False).astype(bool)]
        visible_reset = reset.loc[~reset["in_non_operation_interval"].fillna(False).astype(bool)]
    alarm_styles = {
        "TA": {"color": "#2ca02c", "label": "TA: alarm in target horizon"},
        "FA": {"color": "#d62728", "label": "FA: operational alarm outside target horizon"},
        "ignored_non_operation": {"color": "#8d99ae", "label": "ignored alarm in non-operation"},
    }
    shade_non_operation_intervals([ax], non_operation_intervals)
    shade_target_horizons([ax], horizon_intervals, plot_start, plot_end)

    ax.plot(time, period["cusum_positive"], color="#2ca02c", linewidth=1.1, label="positive_sum")
    ax.plot(time, period["cusum_negative"], color="#9467bd", linewidth=1.1, label="negative_sum")
    ax.axhline(h, color="#d62728", linestyle="--", linewidth=1.2, label="+h / -h threshold")
    ax.axhline(-h, color="#d62728", linestyle="--", linewidth=1.2)
    for alarm_type, style in alarm_styles.items():
        part = visible_alarm.loc[visible_alarm["alarm_type"].eq(alarm_type)]
        if part.empty:
            continue
        ax.scatter(
            part["Date and time"],
            part["cusum_positive"],
            color=style["color"],
            s=24,
            zorder=5,
            label=style["label"],
        )
        ax.scatter(part["Date and time"], part["cusum_negative"], color=style["color"], s=24, zorder=5)
    for ts in visible_reset["Date and time"]:
        ax.axvline(ts, color="#d62728", alpha=0.18, linewidth=0.8)
    ax.set_ylabel("CUSUM")
    ax.set_title("CUSUM accumulation and reset")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=True,
        fontsize=9,
    )

    reset_text = "reset on alarm" if reset_on_alarm else "no reset"
    hidden_text = ", non-operation alarms hidden" if hide_non_operation_alarms else ""
    fig.suptitle(
        (
            f"CUSUM timeline | {turbine} | {target} | {period_label}\n"
            f"k={k:g}, h={h:g}, {reset_text}{hidden_text}, "
            f"info-env buffer=±{informational_env_buffer_hours}h, "
            f"center={center:.4g}, std={scale:.4g}, run={run_id}"
        ),
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.k < 0:
        raise ValueError("k must be non-negative")
    if args.h <= 0:
        raise ValueError("h must be positive")
    if not 0 <= args.month <= 12:
        raise ValueError("month must be between 0 and 12, where 0 means full year")

    project_root = PROJECT_ROOT
    result_dir, metadata = load_metadata(project_root, args.run_id)
    center, scale, _ = fit_reference_from_validation(project_root, metadata, args.target)
    test_predictions = load_test_prediction_columns(project_root, metadata, args.target)
    trajectory = compute_cusum_trajectory(
        test_predictions,
        turbine=args.turbine,
        center=center,
        scale=scale,
        k=args.k,
        h=args.h,
        reset_on_alarm=not args.no_reset,
    )
    period = filter_period(trajectory, args.year, args.month)
    horizon_intervals = load_target_horizon_intervals(
        project_root,
        turbine=args.turbine,
        year=args.year,
        month=args.month,
        horizon_days=7,
    )
    non_operation_intervals = load_non_operation_intervals(
        project_root,
        turbine=args.turbine,
        year=args.year,
        month=args.month,
        informational_env_buffer_hours=args.informational_env_buffer_hours,
    )
    period = mark_non_operation_points(period, non_operation_intervals)
    period = mark_target_horizon_points(period, horizon_intervals)
    period = add_alarm_type(period)

    setting = f"k{format_number(args.k)}_h{format_number(args.h)}_{'reset' if not args.no_reset else 'noreset'}"
    target_slug = safe_label(args.target)
    period_slug = "all_year" if args.month == 0 else f"{args.month:02d}"
    visibility_label = "hide_nonop" if args.hide_non_operation_alarms else "all_alarms"
    buffer_label = f"infoenvbuf{args.informational_env_buffer_hours}h"
    base_name = f"{setting}_{buffer_label}_{visibility_label}_{args.turbine}_{args.year}_{period_slug}_{target_slug}"
    csv_dir = result_dir / "cusum_timelines"
    fig_dir = project_root / "results" / "kelmarsh" / "figures" / "cusum_timelines"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{base_name}.csv"
    fig_path = fig_dir / f"{base_name}.png"
    period.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_cusum_timeline(
        period,
        horizon_intervals=horizon_intervals,
        non_operation_intervals=non_operation_intervals,
        output_path=fig_path,
        run_id=args.run_id,
        turbine=args.turbine,
        target=args.target,
        year=args.year,
        month=args.month,
        k=args.k,
        h=args.h,
        center=center,
        scale=scale,
        reset_on_alarm=not args.no_reset,
        hide_non_operation_alarms=args.hide_non_operation_alarms,
        informational_env_buffer_hours=args.informational_env_buffer_hours,
    )

    print("Timeline CSV saved to:", csv_path)
    print("Timeline figure saved to:", fig_path)
    print("Rows plotted:", len(period))
    print("Alarm/reset points in selected period:", int(period["is_anomaly"].sum()))
    print(
        "Alarm/reset points hidden in non-operation intervals:",
        int((period["is_anomaly"] & period["in_non_operation_interval"]).sum())
        if args.hide_non_operation_alarms
        else 0,
    )
    alarm_counts = period.loc[period["is_anomaly"], "alarm_type"].value_counts().to_dict()
    print("Alarm type counts:", alarm_counts)
    print("Target prediction horizons in selected period:", len(horizon_intervals))
    print("Non-operation intervals in selected period:", len(non_operation_intervals))
    print("Informational environmental status buffer hours:", args.informational_env_buffer_hours)


if __name__ == "__main__":
    main()
