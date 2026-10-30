"""Export individual and cross-method timing figures for revised Chapter 7 settings."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from scripts.analysis.prepare_scc_thesis_best import histogram
from scripts.plotting.plot_thesis_seven_settings_split6h import plot_comparison


DATA = ROOT / "results" / "kelmarsh" / "thesis_revision_split6h" / "chapter7_selected_revised"
FIGURES = ROOT.parent / "Thesis_LaTeX" / "figures" / "chapter7"
NAMES = {
    "cusum_two_sided": "cusum_two",
    "cusum_one_sided": "cusum_one",
    "bcad_folded": "bcad_folded",
    "bcad_gaussian": "bcad_gaussian",
    "bcad_weibull_max": "bcad_maximum",
    "bcad_weibull_joint": "bcad_joint",
}


def main() -> None:
    comparison = pd.read_csv(DATA / "chapter7_comparison.csv")
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    for slug, name in NAMES.items():
        expected = int(comparison.loc[comparison["variant"].eq(slug), "detected_events"].iloc[0])
        lead = pd.read_csv(DATA / f"{slug}_lead_time_events.csv")["lead_days"].dropna()
        coverage = pd.read_csv(DATA / f"{slug}_horizon_coverage_events.csv")["covered_days"]
        coverage = coverage.loc[coverage.gt(0)]
        if len(lead) != expected or len(coverage) != expected:
            raise ValueError(f"{slug}: timing rows {len(lead)}/{len(coverage)} != {expected} detections")
        histogram(lead.to_numpy(), FIGURES / f"lead_time_hist_{name}_no_warning.pdf",
                  "Lead time before event start (days)")
        histogram(coverage.to_numpy(), FIGURES / f"coverage_{name}_no_warning.pdf",
                  "Alarm-covered time within the seven-day horizon (days)", maximum=7)
        print(f"{slug}: {expected} event-level timing rows", flush=True)

    for column, filename, xlabel, limits in (
        ("lead_time", "lead_time_comparison_no_warning.pdf",
         "Lead time before event start (days)", (0, 12)),
        ("horizon_coverage", "coverage_comparison_no_warning.pdf",
         "Alarm-covered time in seven-day horizon (days)", (0, 7)),
    ):
        plot_comparison(DATA, column, filename, xlabel, limits)
        shutil.copy2(DATA / filename, FIGURES / filename)
    print("Exported revised Chapter 7 timing figures to", FIGURES)


if __name__ == "__main__":
    main()
