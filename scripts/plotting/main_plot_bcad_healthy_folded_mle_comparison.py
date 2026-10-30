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
from scipy.stats import foldnorm

from main_detect_residual_baseline import find_experiment, load_prediction_file
from src.config.kelmarsh_config import TARGET_COLS
from src.detection.residuals import resolve_prediction_column


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy and MLE healthy folded-normal fits.")
    parser.add_argument("--run-id", help="Experiment run ID. Defaults to the newest experiment.")
    args = parser.parse_args()

    result_dir, _, metadata = find_experiment(PROJECT_ROOT, args.run_id)
    validation = load_prediction_file(
        PROJECT_ROOT, metadata, path_key="predictions", fallback_name="healthy_val_predictions.csv"
    )
    mle = pd.read_csv(result_dir / "bcad_folded_normal_mle_healthy_parameters.csv")
    targets = metadata.get("targets", TARGET_COLS)
    fig, axes = plt.subplots(1, len(targets), figsize=(15, 4.6), constrained_layout=True)

    for ax, target in zip(np.atleast_1d(axes), targets):
        error_col = resolve_prediction_column(validation, "error", target)
        signed = pd.to_numeric(validation[error_col], errors="coerce").dropna()
        absolute = signed.abs().to_numpy()
        old_center = float(signed.mean())
        old_scale = float(signed.std(ddof=0))
        row = mle.loc[mle["target"].eq(target)].iloc[0]
        new_center = float(row["healthy_center"])
        new_scale = float(row["healthy_scale"])
        x_max = float(np.quantile(absolute, 0.995))
        edges = np.linspace(0, x_max, 85)
        counts, _ = np.histogram(absolute, bins=edges)
        ax.stairs(counts / (len(absolute) * np.diff(edges)), edges,
                  color="#555555", linewidth=1.15, label="Validation |error|")
        x = np.linspace(0, x_max, 600)
        ax.plot(x, foldnorm.pdf(x, c=abs(old_center) / old_scale, scale=old_scale),
                color="#d97732", linewidth=2, label="Legacy moment parameters")
        ax.plot(x, foldnorm.pdf(x, c=new_center / new_scale, scale=new_scale),
                color="#087e8b", linewidth=2, label="Maximum-likelihood fit")
        ax.set_title(target, fontsize=10)
        ax.set_xlabel("Absolute validation error (°C)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)
    fig.suptitle("Healthy residual distribution: moment parameters vs folded-normal MLE", fontsize=13)
    output = PROJECT_ROOT / "results" / "kelmarsh" / "figures" / "meeting" / "bcad_healthy_folded_mle_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print("Figure saved to:", output)


if __name__ == "__main__":
    main()
