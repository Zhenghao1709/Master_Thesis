"""Recreate SCC timing evidence for the revised best absolute-error setting."""

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
import numpy as np
import pandas as pd

from scripts.analysis.main_analyze_alarm_horizon_coverage import analyze_method, summarize
from scripts.analysis.main_analyze_alarm_lead_time import analyze_episode_trigger_times, load_events


RUN = "all6_multi3_seq12_h64_l1_bs64_lr1e-03_wd0_do0_e60_p8_seed42"
SETTING = "abs_meanstd_k5_c6_split6h_no_warning_turbine"
RESULTS = PROJECT_ROOT / "results" / "kelmarsh" / "experiments" / RUN
OUTPUT = PROJECT_ROOT / "results" / "kelmarsh" / "thesis_revision_split6h" / "scc_best_k5"


def histogram(values: np.ndarray, output: Path, xlabel: str, maximum: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 2.65), layout="constrained")
    median = float(np.median(values))
    bins = min(28, max(12, int(np.sqrt(len(values))))) if maximum is None else np.linspace(0, maximum, 25)
    ax.hist(values, bins=bins, color="#4C78A8", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(median, color="#C93C37", linestyle="--", linewidth=1.2,
               label=f"Median: {median:.2f} days")
    ax.legend(loc="upper left" if maximum is None and max(values) < 8 else "upper right",
              frameon=False, fontsize=8)
    ax.set_ylabel("Events")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, length=3)
    if maximum is not None:
        ax.set_xlim(0, maximum)
        ax.set_xticks(np.arange(0, maximum + 1, 1))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    events = load_events(PROJECT_ROOT, horizon_days=7)
    episodes = pd.read_csv(
        RESULTS / f"scc_alarm_episodes_{SETTING}.csv",
        parse_dates=["start_time", "end_time"],
    )
    lead = analyze_episode_trigger_times(episodes, events, freq_minutes=10)
    coverage = analyze_method("SCC", episodes, events, require_original_overlap=True)
    detected_coverage = coverage.loc[coverage["covered_days"].gt(0)]
    if len(lead) != 407 or len(detected_coverage) != 407:
        raise ValueError(f"Expected 407 SCC detections, got lead={len(lead)}, coverage={len(detected_coverage)}")
    lead.to_csv(OUTPUT / "scc_lead_time_events.csv", index=False)
    coverage.to_csv(OUTPUT / "scc_horizon_coverage_events.csv", index=False)
    histogram(lead["lead_days"].to_numpy(), OUTPUT / "lead_time_hist_scc_no_warning.pdf",
              "Lead time before event start (days)")
    histogram(detected_coverage["covered_days"].to_numpy(), OUTPUT / "coverage_scc_no_warning.pdf",
              "Alarm-covered time within the seven-day horizon (days)", maximum=7)
    summary = {
        "setting": SETTING,
        "detected_events": len(lead),
        "median_lead_days": float(lead["lead_days"].median()),
        "lead_q25_days": float(lead["lead_days"].quantile(0.25)),
        "lead_q75_days": float(lead["lead_days"].quantile(0.75)),
        **summarize(coverage),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
