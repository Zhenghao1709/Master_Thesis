"""Plot revised Chapter 7 cross-method comparisons from prepared event tables."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from scripts.analysis.detection_results_viewer_data import load_coverage_comparison, plot_coverage_comparison


ORDER = ["scc", "cusum_two_sided", "cusum_one_sided", "bcad_folded", "bcad_gaussian",
         "bcad_weibull_max", "bcad_weibull_joint"]
LABELS = ["SCC", "Two-sided CUSUM", "One-sided CUSUM", "Folded-normal BCAD",
          "Gaussian BCAD", "Window-max Weibull BCAD", "Joint-window Weibull BCAD"]


def plot_comparison(folder: Path, column: str, filename: str, xlabel: str, xlim: tuple[float, float]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.labelsize": 12, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "pdf.fonttype": 42})
    values = [pd.read_csv(folder / f"{slug}_{column}_events.csv")[
        "lead_days" if column == "lead_time" else "covered_days"
    ].dropna().to_numpy() for slug in ORDER]
    if column == "horizon_coverage":
        values = [value[value > 0] for value in values]
    if column == "lead_time":
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), facecolor="white",
                                 gridspec_kw={"width_ratios": [1, 1.15]})
        for ax, group, names, limit in (
            (axes[0], values[:3], LABELS[:3], (0, 8)),
            (axes[1], values[3:], LABELS[3:], xlim),
        ):
            names = [name.replace(" CUSUM", "\nCUSUM").replace(" BCAD", "\nBCAD") for name in names]
            box = ax.boxplot(group, vert=False, tick_labels=names, patch_artist=True, showfliers=False)
            for patch in box["boxes"]:
                patch.set_facecolor("#4486aa" if ax is axes[0] else "#2c9878")
                patch.set_alpha(0.78)
            ax.set_xlim(*limit)
            ax.grid(axis="x", alpha=0.25)
            ax.invert_yaxis()
            ax.set_xlabel("Lead time (days)")
        fig.tight_layout()
        fig.savefig(folder / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(7.2, 3.9), facecolor="white")
    box = ax.boxplot(values, vert=False, tick_labels=LABELS, patch_artist=True, showfliers=False)
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor("#4486aa" if index < 3 else "#2c9878")
        patch.set_alpha(0.78)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(folder / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    folder = PROJECT_ROOT / "results" / "kelmarsh" / "thesis_revision_split6h" / "chapter7_selected_revised"
    if not (folder / "chapter7_comparison.csv").is_file():
        raise FileNotFoundError("Run prepare_thesis_seven_settings_split6h.py first")
    plot_comparison(folder, "lead_time", "lead_time_comparison.png",
                    "Lead time before event start (days)", (0, 12))
    plot_comparison(folder, "horizon_coverage", "coverage_comparison.png",
                    "Alarm-covered time in seven-day horizon (days)", (0, 7))
    coverage = load_coverage_comparison(PROJECT_ROOT)
    coverage.to_csv(folder / "non_operational_buffer_coverage.csv", index=False)
    fig = plot_coverage_comparison(coverage)
    fig.savefig(folder / "non_operational_buffer_coverage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved cross-method and non-operational coverage figures to", folder)


if __name__ == "__main__":
    main()
