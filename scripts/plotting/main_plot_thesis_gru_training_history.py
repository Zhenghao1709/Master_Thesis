from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "all6_multi3_seq12_h64_l1_bs64_lr1e-03_wd0_do0_e60_p8_seed42"
RUN_DIR = PROJECT_ROOT / "results" / "kelmarsh" / "experiments" / RUN_ID
OUTPUT_DIR = PROJECT_ROOT / "results" / "kelmarsh" / "figures" / "thesis"


def main() -> None:
    history = pd.read_csv(RUN_DIR / "train_log.csv")
    metadata = json.loads((RUN_DIR / "metadata.json").read_text(encoding="utf-8"))
    required = {"epoch", "train_loss", "val_loss"}
    if not required.issubset(history.columns):
        raise ValueError(f"Missing history columns: {sorted(required - set(history.columns))}")
    if history.empty or not np.array_equal(history["epoch"], np.arange(1, len(history) + 1)):
        raise ValueError("Epoch history must start at 1 and be consecutive")
    if not np.isfinite(history[["train_loss", "val_loss"]].to_numpy()).all():
        raise ValueError("Epoch history contains non-finite losses")

    best_row = history.loc[history["val_loss"].idxmin()]
    best_epoch = int(best_row["epoch"])
    stop_epoch = int(history["epoch"].iloc[-1])
    if best_epoch != metadata["best_epoch"] or stop_epoch != metadata["epochs_completed"]:
        raise ValueError("Epoch history does not match saved experiment metadata")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(history["epoch"], history["train_loss"], marker="o", markersize=3,
            linewidth=1.5, label="Training loss")
    ax.plot(history["epoch"], history["val_loss"], marker="o", markersize=3,
            linewidth=1.5, label="Validation loss")
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.scatter(best_epoch, best_row["val_loss"], color="black", s=38,
               zorder=5, label=f"Best epoch {best_epoch}")
    ax.scatter(stop_epoch, history["val_loss"].iloc[-1], marker="s", s=43,
               facecolor="white", edgecolor="black", linewidth=1.3,
               zorder=5, label=f"Stopped at epoch {stop_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (scaled target space)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.58, 1.0))
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "gru_training_validation_loss"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"Best epoch: {best_epoch}; validation MSE: {best_row['val_loss']:.6f}")
    print(f"Stopped at epoch: {stop_epoch}")
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
