"""Plot final input-signal distributions by scaler for thesis Chapter 3.

This follows the histogram/quantile-display approach in signal_viewer.ipynb,
but uses the final healthy training partition only. The display sample and
clipping do not change the data used to fit the thesis scalers.
"""

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
from matplotlib.lines import Line2D

from src.config.kelmarsh_config import SCALER_TYPE_BY_SIGNAL


SIGNALS = [
    ("Wind speed (m/s)", "Wind speed (m/s)"),
    ("Power (kW)", "Power (kW)"),
    ("Nacelle ambient temperature (°C)", "Nacelle ambient (°C)"),
    ("Nacelle temperature (°C)", "Nacelle temperature (°C)"),
    ("Generator RPM (RPM)", "Generator speed (RPM)"),
    ("Generator bearing front temperature (°C)", "Front bearing (°C)"),
    ("Stator temperature 1 (°C)", "Stator 1 (°C)"),
    ("Rear bearing temperature (°C)", "Rear bearing (°C)"),
]

METHOD_COLORS = {"robust": "#298273", "standard": "#476fa3", "minmax": "#d08136"}
GROUP_LAYOUTS = {
    "standard": (2, 2, (7.2, 4.2), 0.55, 0.31),
    "robust": (1, 3, (7.2, 2.6), 0.55, 0.36),
    "minmax": (1, 1, (7.2, 2.2), 0.55, 0.31),
}


def main() -> None:
    input_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "kelmarsh"
        / "train_val"
        / "all6_train_healthy.parquet"
    )
    output_dir = PROJECT_ROOT / "results" / "kelmarsh" / "figures" / "thesis"
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = [column for column, _ in SIGNALS]
    train = pd.read_parquet(input_path, columns=columns)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    handles = [
        Line2D([0], [0], color="#9e3b39", linestyle="--", lw=1.2, label="Mean"),
        Line2D([0], [0], color="#253f30", linestyle=":", lw=1.5, label="Median"),
    ]
    print(f"Training rows: {len(train):,}")

    for method in ("standard", "robust", "minmax"):
        group_signals = [item for item in SIGNALS if SCALER_TYPE_BY_SIGNAL[item[0]] == method]
        nrows, ncols, size, hspace, wspace = GROUP_LAYOUTS[method]
        fig, axes = plt.subplots(nrows, ncols, figsize=size)

        for ax, (column, label) in zip(np.atleast_1d(axes).flat, group_signals):
            full = pd.to_numeric(train[column], errors="coerce").dropna()
            low, high = full.quantile([0.005, 0.995])
            visible = full[full.between(low, high)]

            ax.hist(
                visible,
                bins=35,
                density=True,
                color=METHOD_COLORS[method],
                alpha=0.72,
                edgecolor="none",
            )
            ax.axvline(full.mean(), color="#9e3b39", linestyle="--", linewidth=1.2)
            ax.axvline(full.median(), color="#253f30", linestyle=":", linewidth=1.5)
            ax.set_title(label, fontsize=10.2, pad=5)
            ax.set_xlim(low, high)
            ax.set_ylabel("Density", fontsize=8.5)
            ax.tick_params(axis="both", labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)

        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
        fig.subplots_adjust(
            left=0.10, right=0.98, top=0.93 if nrows == 1 else 0.95,
            bottom=0.22 if nrows == 1 else 0.13,
            hspace=hspace, wspace=wspace,
        )

        stem = output_dir / f"signal_distributions_{method}"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"{method}: {len(group_signals)} signals -> {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
