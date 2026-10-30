"""Prepare the seven Chapter 7 settings under the revised shared evaluation rule."""

from __future__ import annotations

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

from main_detect_residual_baseline import (
    annotate_alarm_episode_horizon_matches,
    annotate_alarm_episode_operating_context,
    fuse_alarm_episodes,
    update_event_summary_from_alarm_episodes,
)
from scripts.analysis.detection_results_viewer_data import (
    _episode_path,
    _target_events,
    experiment_dir,
    plot_timeline,
)
from scripts.analysis.main_analyze_alarm_horizon_coverage import (
    analyze_method,
    plot_method_distribution,
    summarize,
)
from scripts.analysis.main_analyze_alarm_lead_time import (
    DEFAULT_RUN_ID,
    analyze_episode_trigger_times,
    make_summary,
    plot_lead_time_distribution,
)


SETTINGS = {
    "scc": ("scc", "abs_meanstd_k5_c6_split6h_no_warning_turbine"),
    "bcad_folded": ("bcad", "foldedmle_samecenterabs_w72_q990_vm5_c72"),
    "bcad_gaussian": ("bcad", "abslog_w72_q990_vm5_c72"),
    "bcad_weibull_max": ("bcad", "weibullmle_foldedcalib2021_2022_w72_q990_c72"),
    "bcad_weibull_joint": ("bcad", "weibulljointmle_foldedcalib2021_2022_w72_q990_c72"),
}


def load_source_episodes(run_dir: Path, method: str, setting: str) -> pd.DataFrame:
    if method != "bcad":
        source = (run_dir / f"{method.split('_')[0]}_alarm_episodes_{setting}.csv"
                  if setting.endswith("_split6h_no_warning_turbine")
                  else _episode_path(run_dir, method, setting))
        return pd.read_csv(source, parse_dates=["start_time", "end_time"])

    original = setting.removesuffix("_c72") + "_c6"
    source = _episode_path(run_dir, method, original)
    episodes = pd.read_csv(source, parse_dates=["start_time", "end_time"])
    episodes = episodes.loc[episodes["alarm_points"].ge(67)].copy()
    episodes["start_time"] += pd.Timedelta(minutes=66 * 10)
    episodes["alarm_points"] -= 66
    return episodes


def evaluate(
    episodes: pd.DataFrame, events: pd.DataFrame, c: int,
    evaluation_cache: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    fused = fuse_alarm_episodes(episodes)
    fused = annotate_alarm_episode_horizon_matches(fused, events)
    scored = annotate_alarm_episode_operating_context(
        fused, PROJECT_ROOT, buffer_hours=6, non_operational_rule="remaining_consecutive",
        min_operational_consecutive=c,
        evaluation_cache=evaluation_cache,
    )
    event_summary = update_event_summary_from_alarm_episodes(events.copy(), scored)
    ta = int(scored["in_fault_horizon"].sum())
    fa = int(scored["is_operational_false_alarm"].sum())
    detected = int(event_summary["detected_in_horizon"].sum())
    precision = ta / (ta + fa) if ta + fa else 0.0
    recall = detected / len(events)
    metrics = {
        "total_events": len(events), "detected_events": detected, "missed_events": len(events) - detected,
        "total_alarm_episodes": len(scored), "true_alarm_episodes": ta,
        "operational_false_alarm_episodes": fa, "operational_false_alarm_rate": fa / (ta + fa) if ta + fa else 0.0,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "suppressed_source_episodes": int(scored.attrs["suppressed_source_episodes"]),
    }
    return scored, metrics


def main() -> None:
    run_dir = experiment_dir(PROJECT_ROOT, DEFAULT_RUN_ID)
    revision = PROJECT_ROOT / "results" / "kelmarsh" / "thesis_revision_split6h"
    chosen = pd.read_csv(revision / "selected_cusum.csv")
    settings = {
        "scc": SETTINGS["scc"],
        "cusum_two_sided": ("cusum_two_sided", chosen.loc[chosen["mode"].eq("two-sided"), "setting"].iloc[0]),
        "cusum_one_sided": ("cusum_absolute", chosen.loc[chosen["mode"].eq("absolute"), "setting"].iloc[0]),
        **{key: value for key, value in SETTINGS.items() if key.startswith("bcad_")},
    }
    out = revision / "chapter7_selected_revised"
    out.mkdir(parents=True, exist_ok=True)
    events = _target_events(PROJECT_ROOT).copy().reset_index(drop=True)
    events["event_uid"] = events.index + 1
    evaluation_cache = {"target_events": events}
    rows = []

    for slug, (method, setting) in settings.items():
        c = 72 if method == "bcad" else 1 if method.startswith("cusum") else 6
        source = load_source_episodes(run_dir, method, setting)
        scored, metrics = evaluate(source, events, c, evaluation_cache=evaluation_cache)
        scored.to_csv(out / f"{slug}_evaluated_episodes.csv", index=False)
        valid = scored.loc[~scored["in_non_operational_interval"].fillna(False)].copy()
        label = "CUSUM" if method.startswith("cusum") else method.upper()
        lead = analyze_episode_trigger_times(valid, events, freq_minutes=10)
        lead_summary = make_summary(label, setting, lead, len(events))
        lead.to_csv(out / f"{slug}_lead_time_events.csv", index=False)
        plot_lead_time_distribution(lead, lead_summary, out / f"{slug}_lead_time.png")

        coverage = analyze_method(label, valid, events, cusum_coverage_hours=12,
                                  require_original_overlap=True)
        coverage_summary = summarize(coverage)
        coverage.to_csv(out / f"{slug}_horizon_coverage_events.csv", index=False)
        plot_method_distribution(label, setting, coverage.loc[coverage["covered_hours"].gt(0)],
                                 coverage_summary, out / f"{slug}_horizon_coverage.png", 12)

        year_start, year_end = pd.Timestamp(2023, 1, 1), pd.Timestamp(2024, 1, 1)
        shown_events = events.loc[events["turbine_id"].eq("Kelmarsh_1")
                                  & events["event_start"].lt(year_end) & events["event_end"].gt(year_start)]
        shown_episodes = scored.loc[scored["turbine_id"].eq("Kelmarsh_1")
                                    & scored["start_time"].lt(year_end) & scored["end_time"].ge(year_start)]
        fig = plot_timeline(shown_events, shown_episodes, "Kelmarsh_1", 2023, method, setting,
                            [], show_non_operational=False, buffer_hours=6,
                            non_operational_rule="remaining_consecutive")
        fig.savefig(out / f"{slug}_example_timeline_kelmarsh1_2023.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        rows.append({"variant": slug, "setting": setting, "post_split_c": c, **metrics,
                     "median_lead_days": lead_summary.get("lead_days_50%"),
                     "median_covered_days": coverage_summary["median_covered_days_detected_events"],
                     "mean_coverage_percent": coverage_summary["mean_horizon_coverage_percent_detected_events"]})
        print(f"{slug}: {metrics['detected_events']}/{len(events)}, FA={metrics['operational_false_alarm_episodes']}, "
              f"F1={metrics['f1']:.3f}", flush=True)

    pd.DataFrame(rows).to_csv(out / "chapter7_comparison.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "run_id": DEFAULT_RUN_ID,
        "target_events": "459 no-Warning target events, horizon=7d",
        "non_operational": "maintenance, communication, curtailment, non-Full-Performance status, target-event intervals; informational/environmental buffer +/-6h",
        "episode_rule": "TA-first; for other episodes remove non-operational points, split, then apply post-split c",
        "bcad_c72_derivation": "Shift each existing c6 episode start by 66 ten-minute points, discard episodes shorter than 67 c6 alarm points. Verified exactly against precomputed W72 c72 source (2835/2835 target episodes).",
        "cusum_coverage_hours": 12,
        "caveat": "Retrospective TA-first classification uses known test event labels; parameter selection and comparisons remain exploratory.",
        "settings": settings,
    }, indent=2), encoding="utf-8")
    print("Prepared seven-setting thesis outputs:", out)


if __name__ == "__main__":
    main()
