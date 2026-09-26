from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


TRAIN_COLOR = "#4472A8"
VALIDATION_COLOR = "#55A868"
TEST_COLOR = "#E07A3F"


def main() -> None:
    project_root = PROJECT_ROOT
    data_dir = project_root / "data" / "processed" / "kelmarsh" / "train_val"
    output_dir = project_root / "results" / "kelmarsh" / "figures" / "thesis"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(
        data_dir / "all6_train_healthy.parquet", columns=["Date and time"]
    )
    validation = pd.read_parquet(
        data_dir / "all6_val_healthy.parquet", columns=["Date and time"]
    )
    split_time = max(
        pd.to_datetime(train["Date and time"]).max(),
        pd.to_datetime(validation["Date and time"]).min(),
    )

    study_start = pd.Timestamp("2016-01-01")
    test_start = pd.Timestamp("2023-01-01")
    study_end = pd.Timestamp("2025-01-01")
    fig, ax = plt.subplots(figsize=(12.2, 3.8))
    sections = [
        (study_start, split_time, TRAIN_COLOR),
        (split_time, test_start, VALIDATION_COLOR),
        (test_start, study_end, TEST_COLOR),
    ]
    for start, end, color in sections:
        ax.barh(
            0,
            mdates.date2num(end) - mdates.date2num(start),
            left=mdates.date2num(start),
            height=0.54,
            color=color,
            edgecolor="white",
            linewidth=1.0,
        )

    ax.axvline(split_time, color="#333333", linewidth=1.1, linestyle="--", zorder=5)
    ax.axvline(test_start, color="#333333", linewidth=1.4, zorder=5)

    ax.annotate(
        "Healthy model-development\ndata",
        xy=(mdates.date2num(pd.Timestamp("2019-01-01")), 0.48),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="semibold",
        color="#333333",
    )
    ax.annotate(
        "Independent full-timeline\nevaluation",
        xy=(mdates.date2num(pd.Timestamp("2024-01-01")), 0.48),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="semibold",
        color="#333333",
    )

    ax.set_yticks([])
    ax.set_xlim(study_start, study_end - pd.Timedelta(days=1))
    ax.set_ylim(-0.58, 0.95)
    ax.set_xticks(
        [mdates.date2num(pd.Timestamp(f"{year}-01-01")) for year in range(2016, 2025)]
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_xlabel("")
    ax.set_ylabel("")

    legend_handles = [
        mpatches.Patch(color=TRAIN_COLOR, label="Training: selected healthy observations"),
        mpatches.Patch(color=VALIDATION_COLOR, label="Validation: selected healthy observations"),
        mpatches.Patch(color=TEST_COLOR, label="Test: full timeline with QC flags"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.36),
        ncol=3,
        frameon=False,
        fontsize=9.2,
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#777777")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.91, bottom=0.36)

    stem = output_dir / "kelmarsh_chronological_data_partition"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Chronological split: {split_time}")
    print(f"PNG: {stem.with_suffix('.png')}")
    print(f"PDF: {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
