from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from main_detect_residual_baseline import find_experiment
from scripts.analysis.main_analyze_alarm_lead_time import (
    DEFAULT_RUN_ID, load_events, load_operational_turbine_episodes,
)


PROFILES = {
    "previous": {
        "SCC": "signed_meanstd_abslimit_k3p5_c6",
        "BCAD": "foldedmle_samecenterabs_w72_q990_vm5_c6",
        "CUSUM": "twosided_std_k0_h100_reset",
    },
    "thesis_best": {
        "SCC": "abs_meanstd_k6_c6",
        "BCAD": "weibullmle_foldedcalib2021_2022_w936_q990_c6",
        "CUSUM": "twosided_std_k0_h100_reset",
    },
    "thesis_best_no_warning": {
        "SCC": "abs_meanstd_k6_c6",
        "BCAD": "no_warning_bcad/bcad_alarm_episodes_weibullmle_foldedcalib2021_2022_w936_q990_c6",
        "CUSUM": "no_warning_lead_time/cusum_alarm_episodes_twosided_std_k0_h140_reset",
    },
}
def union_coverage(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    horizon_start: pd.Timestamp,
    event_start: pd.Timestamp,
) -> float:
    clipped = sorted(
        (max(start, horizon_start), min(end, event_start))
        for start, end in intervals
        if min(end, event_start) > max(start, horizon_start)
    )
    if not clipped:
        return 0.0

    merged = [list(clipped[0])]
    for start, end in clipped[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    covered_hours = sum(
        (end - start).total_seconds() / 3600 for start, end in merged
    )
    return covered_hours


def analyze_method(
    method: str,
    episodes: pd.DataFrame,
    events: pd.DataFrame,
    freq_minutes: int = 10,
    cusum_coverage_hours: int = 12,
    require_original_overlap: bool = False,
) -> pd.DataFrame:
    episodes = episodes.copy()
    episodes["start_time"] = pd.to_datetime(episodes["start_time"], errors="coerce")
    episodes["end_time"] = pd.to_datetime(episodes["end_time"], errors="coerce")
    episodes = episodes.dropna(subset=["start_time", "end_time"])
    extension = pd.Timedelta(
        hours=cusum_coverage_hours if method == "CUSUM" else 0,
        minutes=0 if method == "CUSUM" else freq_minutes,
    )
    episodes["coverage_end"] = episodes["end_time"] + extension
    episodes["original_end_exclusive"] = episodes["end_time"] + pd.Timedelta(minutes=freq_minutes)
    by_turbine = {
        turbine: frame for turbine, frame in episodes.groupby("turbine_id", sort=False)
    }

    rows = []
    for event in events.itertuples(index=False):
        horizon_start = event.horizon_start
        event_start = event.event_start
        candidates = by_turbine.get(event.turbine_id)
        intervals = []
        if candidates is not None:
            valid = candidates.loc[
                candidates["start_time"].lt(event_start)
                & candidates["coverage_end"].gt(horizon_start)
                & (
                    candidates["original_end_exclusive"].ge(horizon_start)
                    if require_original_overlap else True
                )
            ]
            intervals = list(zip(valid["start_time"], valid["coverage_end"]))

        covered_hours = union_coverage(
            intervals, horizon_start, event_start
        )
        horizon_hours = (event_start - horizon_start).total_seconds() / 3600
        rows.append(
            {
                "method": method,
                "event_uid": event.event_uid,
                "turbine_id": event.turbine_id,
                "event_start": event_start,
                "horizon_start": horizon_start,
                "covered_hours": covered_hours,
                "covered_days": covered_hours / 24,
                "coverage_fraction": covered_hours / horizon_hours,
                "contributing_episodes": len(intervals),
            }
        )
    return pd.DataFrame(rows)


def summarize(coverage: pd.DataFrame) -> dict:
    values = coverage["covered_days"]
    detected = coverage.loc[coverage["covered_days"].gt(0)]
    positive = detected["covered_days"]
    return {
        "total_events": int(len(values)),
        "events_with_coverage": int(len(positive)),
        "events_without_coverage": int((values == 0).sum()),
        "mean_covered_days_detected_events": float(positive.mean()) if not positive.empty else None,
        "median_covered_days_detected_events": float(positive.median()) if not positive.empty else None,
        "mean_horizon_coverage_percent_detected_events": (
            float(detected["coverage_fraction"].mean() * 100) if not detected.empty else None
        ),
        "median_horizon_coverage_percent_detected_events": (
            float(detected["coverage_fraction"].median() * 100) if not detected.empty else None
        ),
        "p25_covered_days_detected_events": float(positive.quantile(0.25)) if not positive.empty else None,
        "p75_covered_days_detected_events": float(positive.quantile(0.75)) if not positive.empty else None,
        "events_with_at_least_one_day_coverage": int((values >= 1).sum()),
        "events_with_at_least_three_days_coverage": int((values >= 3).sum()),
    }


def plot_method_distribution(
    method: str,
    setting: str,
    coverage: pd.DataFrame,
    summary: dict,
    output: Path,
    cusum_coverage_hours: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("white")
    values = coverage["covered_days"]
    median = summary["median_covered_days_detected_events"]

    ax = axes[0]
    if not values.empty:
        bins = min(24, max(8, int(values.nunique())))
        ax.hist(values, bins=bins, color="#4c78a8", alpha=0.82, edgecolor="white")
        ax.axvline(median, color="#d62728", linestyle="--", linewidth=1.4,
                   label=f"median={median:.2f} days")
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "No detected events for this setting", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Alarm coverage distribution for detected target events")
    ax.set_xlabel("alarm-covered time within the seven-day horizon (days)")
    ax.set_ylabel("event count")
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    if not values.empty:
        ax2.boxplot(values, vert=False, widths=0.55, patch_artist=True)
        ax2.set_yticks([])
    ax2.set_xlabel("alarm-covered time within the seven-day horizon (days)")
    ax2.grid(True, axis="x", alpha=0.25)

    carry = f"; each alarm covers the next {cusum_coverage_hours} h" if method == "CUSUM" else ""
    fig.suptitle(
        f"{method} alarm coverage | {setting}\n"
        f"detected events={summary['events_with_coverage']}/{summary['total_events']}; "
        f"mean horizon coverage={summary['mean_horizon_coverage_percent_detected_events']:.1f}%"
        f"{carry}",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare unique alarm coverage in the seven days before target events."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--cusum-coverage-hours", type=int, default=12)
    parser.add_argument("--profile", choices=PROFILES, default="previous")
    args = parser.parse_args()
    if args.horizon_days != 7:
        raise ValueError("This comparison requires a seven-day event horizon")
    if args.cusum_coverage_hours <= 0:
        raise ValueError("CUSUM coverage hours must be positive")

    result_dir, _, _ = find_experiment(PROJECT_ROOT, args.run_id)
    settings = PROFILES[args.profile]
    events = load_events(PROJECT_ROOT, horizon_days=args.horizon_days)
    if args.profile == "thesis_best_no_warning" and len(events) != 459:
        raise ValueError(f"Expected 459 target events without Warning, found {len(events)}")
    coverage_parts = []
    summaries = {}
    for method, setting in settings.items():
        filename = f"{setting}.csv" if "/" in setting else f"{method.lower()}_alarm_episodes_{setting}.csv"
        path = result_dir / filename
        episodes = load_operational_turbine_episodes(path, PROJECT_ROOT)
        frame = analyze_method(
            method, episodes, events,
            cusum_coverage_hours=args.cusum_coverage_hours,
            require_original_overlap=args.profile == "thesis_best_no_warning",
        )
        coverage_parts.append(frame)
        summaries[method] = {"setting": setting, **summarize(frame)}
        print(method, summaries[method])

    coverage = pd.concat(coverage_parts, ignore_index=True)
    detected_coverage = coverage.loc[coverage["covered_hours"].gt(0)].copy()
    suffix = f"7d_cusumcarry{args.cusum_coverage_hours}h_split6h_turbine"
    if args.profile != "previous":
        suffix = f"{args.profile}_{suffix}"
    csv_path = result_dir / f"alarm_horizon_coverage_{suffix}.csv"
    json_path = result_dir / f"alarm_horizon_coverage_summary_{suffix}.json"
    figure_dir = PROJECT_ROOT / "results" / "kelmarsh" / "figures" / "alarm_horizon_coverage"
    if args.profile != "previous":
        figure_dir /= args.profile
    detected_coverage.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    for method, setting in settings.items():
        method_figure = figure_dir / f"{method.lower()}_alarm_coverage_{setting}_{suffix}.png"
        plot_method_distribution(
            method, setting,
            detected_coverage.loc[detected_coverage["method"].eq(method)],
            summaries[method], method_figure, args.cusum_coverage_hours,
        )
        print(f"{method} figure:", method_figure)
    print("Coverage CSV:", csv_path)
    print("Summary JSON:", json_path)


if __name__ == "__main__":
    main()
