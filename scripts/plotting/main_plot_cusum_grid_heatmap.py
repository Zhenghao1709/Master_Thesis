from __future__ import annotations

import argparse
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

from main_detect_residual_baseline import find_experiment


DEFAULT_RUN_ID = "all6_multi3_seq12_h64_l1_bs64_lr1e-03_wd0_do0_e60_p8_seed42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CUSUM grid heatmaps.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--grid-name",
        default="fixed_meanstd_no_consecutive_k0-1_h20-200_buffer6h",
        help="Grid filename label without 'cusum_grid_' prefix or '.csv' suffix.",
    )
    return parser.parse_args()


def plot_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    value_col: str,
    title: str,
    cmap: str,
    fmt: str = ".1%",
) -> None:
    pivot = data.pivot(index="h", columns="k", values=value_col).sort_index(ascending=False)
    im = ax.imshow(pivot.to_numpy(), cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("k")
    ax.set_ylabel("h")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{value:g}" for value in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{value:g}" for value in pivot.index])
    for row_idx, h_value in enumerate(pivot.index):
        for col_idx, k_value in enumerate(pivot.columns):
            value = pivot.loc[h_value, k_value]
            ax.text(
                col_idx,
                row_idx,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    result_dir, _, metadata = find_experiment(project_root, args.run_id)
    grid_path = result_dir / f"cusum_grid_{args.grid_name}.csv"
    if not grid_path.exists():
        raise FileNotFoundError(grid_path)

    data = pd.read_csv(grid_path)
    best = data.loc[data["f1_score"].idxmax()]
    score_mode = data["score_mode"].iloc[0] if "score_mode" in data.columns else "two-sided"
    figure_dir = project_root / "results" / "kelmarsh" / "figures" / "cusum_grids"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f"cusum_grid_heatmap_{args.grid_name}.png"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    fig.suptitle(
        f"CUSUM {score_mode} fixed mean/std grid with ±6h environmental-information buffer\n"
        f"Run: {metadata['run_id']} | Best F1: k={best['k']:g}, h={best['h']:g}, F1={best['f1_score']:.3f}",
        fontsize=14,
    )
    plot_heatmap(axes[0], data, "f1_score", "F1 score", "YlGnBu")
    plot_heatmap(axes[1], data, "detection_rate", "Recall: event detection rate", "Greens")
    plot_heatmap(axes[2], data, "operational_false_alarm_rate", "Operational FA rate", "Reds")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print("Grid CSV:", grid_path)
    print("Heatmap saved to:", figure_path)
    print(
        data.sort_values("f1_score", ascending=False)
        .head(12)[
            [
                "k",
                "h",
                "f1_score",
                "precision",
                "recall",
                "miss_rate",
                "operational_false_alarm_rate",
                "detected_events",
                "total_events",
                "operational_false_alarm_episodes",
            ]
        ]
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
