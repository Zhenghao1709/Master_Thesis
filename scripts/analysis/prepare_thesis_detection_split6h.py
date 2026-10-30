"""Re-evaluate selected thesis methods under the shared point-splitting rule."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from scripts.analysis.detection_results_viewer_data import (
    DEFAULT_SETTINGS,
    _fused_episodes,
    _target_events,
    load_result,
    load_timeline,
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


PROFILE = {
    "scc": DEFAULT_SETTINGS["scc"],
    "cusum_two_sided": DEFAULT_SETTINGS["cusum_two_sided"],
    "bcad": DEFAULT_SETTINGS["bcad"],
}
RULE = "remaining_consecutive"
BUFFER_HOURS = 6
CUSUM_COVERAGE_HOURS = 12


def main() -> None:
    out = PROJECT_ROOT / "results" / "kelmarsh" / "thesis_revision_split6h"
    out.mkdir(parents=True, exist_ok=True)
    events = _target_events(PROJECT_ROOT).copy().reset_index(drop=True)
    events["event_uid"] = events.index + 1
    rows = []
    manifest = {
        "run_id": DEFAULT_RUN_ID,
        "target_events": "459 no-Warning target events; seven-day horizon",
        "non_operational": "maintenance, communication, curtailment, non-Full-Performance status, target-event intervals, environmental Information +/-6h",
        "episode_rule": "Preserve source true alarms; remove non-operational alarm points from other source episodes, split at gaps, retain operational fragments meeting method c",
        "method_c": {"scc": 6, "cusum_two_sided": 1, "bcad": 72},
        "cusum_coverage_hours": CUSUM_COVERAGE_HOURS,
        "warning": "TA-first evaluation uses target labels and is not deployable online postprocessing; these figures are for retrospective evaluation only.",
        "settings": PROFILE,
    }
    for method, setting in PROFILE.items():
        _, metrics, _ = load_result(PROJECT_ROOT, DEFAULT_RUN_ID, method, setting, BUFFER_HOURS, RULE)
        episodes = _fused_episodes(PROJECT_ROOT, DEFAULT_RUN_ID, method, setting, BUFFER_HOURS, RULE)
        operational = episodes.loc[~episodes["in_non_operational_interval"].fillna(False)].copy()
        label = "CUSUM" if method.startswith("cusum") else method.upper()

        lead = analyze_episode_trigger_times(operational, events, freq_minutes=10)
        lead_summary = make_summary(label, setting, lead, len(events))
        lead.to_csv(out / f"{method}_lead_time_events.csv", index=False)
        (out / f"{method}_lead_time_summary.json").write_text(json.dumps(lead_summary, indent=2), encoding="utf-8")
        plot_lead_time_distribution(lead, lead_summary, out / f"{method}_lead_time.png")

        coverage = analyze_method(label, operational, events, cusum_coverage_hours=CUSUM_COVERAGE_HOURS,
                                  require_original_overlap=True)
        coverage_summary = summarize(coverage)
        coverage.to_csv(out / f"{method}_horizon_coverage_events.csv", index=False)
        (out / f"{method}_horizon_coverage_summary.json").write_text(
            json.dumps(coverage_summary, indent=2), encoding="utf-8"
        )
        detected_coverage = coverage.loc[coverage["covered_hours"].gt(0)]
        plot_method_distribution(label, setting, detected_coverage, coverage_summary,
                                 out / f"{method}_horizon_coverage.png", CUSUM_COVERAGE_HOURS)

        timeline_events, timeline_episodes = load_timeline(
            PROJECT_ROOT, DEFAULT_RUN_ID, method, setting, "Kelmarsh_1", 2023, BUFFER_HOURS, RULE,
        )
        fig = plot_timeline(timeline_events, timeline_episodes, "Kelmarsh_1", 2023, method, setting,
                            [], show_non_operational=False, buffer_hours=BUFFER_HOURS,
                            non_operational_rule=RULE)
        fig.savefig(out / f"{method}_example_timeline_kelmarsh1_2023.png", dpi=180, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)

        rows.append({"method": label, "setting": setting, **metrics,
                     "lead_time_events": lead_summary["events_with_valid_ta_alarm"],
                     "median_lead_days": lead_summary.get("lead_days_50%"),
                     "coverage_events": coverage_summary["events_with_coverage"],
                     "mean_coverage_percent": coverage_summary["mean_horizon_coverage_percent_detected_events"]})
        print(f"{label}: detected {metrics['detected_events']}/{metrics['total_events']}, "
              f"FA {metrics['operational_false_alarm_episodes']}, F1 {metrics['f1']:.3f}", flush=True)

    pd.DataFrame(rows).to_csv(out / "selected_method_comparison.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Prepared thesis outputs:", out)


if __name__ == "__main__":
    main()
