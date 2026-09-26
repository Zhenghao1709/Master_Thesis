from __future__ import annotations

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
import seaborn as sns

from src.config.kelmarsh_config import INPUT_COLS


DISPLAY_LABELS = {
    "Wind speed (m/s)": "Wind speed",
    "Power (kW)": "Power",
    "Nacelle ambient temperature (°C)": "Nacelle ambient\ntemperature",
    "Nacelle temperature (°C)": "Nacelle\ntemperature",
    "Generator RPM (RPM)": "Generator speed",
    "Stator temperature 1 (°C)": "Stator\ntemperature",
    "Generator bearing front temperature (°C)": "Generator front bearing\ntemperature",
    "Rear bearing temperature (°C)": "Rear bearing\ntemperature",
}


def main() -> None:
    project_root = PROJECT_ROOT
    data_dir = project_root / "data" / "processed" / "kelmarsh" / "train_val"
    output_dir = project_root / "results" / "kelmarsh" / "figures" / "thesis"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        pd.read_parquet(data_dir / "all6_train_healthy.parquet", columns=INPUT_COLS),
        pd.read_parquet(data_dir / "all6_val_healthy.parquet", columns=INPUT_COLS),
    ]
    healthy_development = pd.concat(frames, ignore_index=True).dropna(subset=INPUT_COLS)
    correlation = healthy_development[INPUT_COLS].corr(method="pearson")
    labels = [DISPLAY_LABELS[column] for column in INPUT_COLS]

    sns.set_theme(style="white", context="paper", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    heatmap = sns.heatmap(
        correlation,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.7,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"fontsize": 9},
        cbar_kws={"label": "Pearson correlation coefficient", "shrink": 0.82},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=42, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    heatmap.collections[0].colorbar.ax.tick_params(labelsize=9)
    fig.tight_layout()

    stem = output_dir / "selected_scada_signal_correlation_matrix"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    correlation.index = labels
    correlation.columns = labels
    correlation.to_csv(stem.with_suffix(".csv"), encoding="utf-8-sig")

    print(f"Rows used: {len(healthy_development):,}")
    print(f"PNG: {stem.with_suffix('.png')}")
    print(f"PDF: {stem.with_suffix('.pdf')}")
    print(f"CSV: {stem.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
