from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "all6_multi3_seq12_h64_l1_bs64_lr1e-03_wd0_do0_e60_p8_seed42"
PREDICTIONS_PATH = (
    PROJECT_ROOT / "results" / "kelmarsh" / "experiments" / RUN_ID
    / "healthy_val_predictions.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "kelmarsh" / "figures" / "thesis"
TARGET = "Generator bearing front temperature (°C)"
TURBINE = "Kelmarsh_1"
START = pd.Timestamp("2021-10-25 00:00:00")
END = pd.Timestamp("2021-11-01 00:00:00")


def main() -> None:
    true_col = f"y_true::{TARGET}"
    pred_col = f"y_pred::{TARGET}"
    error_col = f"error::{TARGET}"
    columns = ["turbine_id", "segment_id", "Date and time", true_col, pred_col, error_col]
    data = pd.read_csv(PREDICTIONS_PATH, usecols=columns)
    data["Date and time"] = pd.to_datetime(data["Date and time"], errors="raise")
    interval = data.loc[
        data["turbine_id"].eq(TURBINE)
        & data["Date and time"].between(START, END, inclusive="left")
    ].sort_values("Date and time")

    if interval.empty or interval["Date and time"].duplicated().any():
        raise ValueError("Selected interval is empty or has duplicate timestamps")
    values = interval[[true_col, pred_col, error_col]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.allclose(values[:, 0] - values[:, 1], values[:, 2], atol=1e-8):
        raise ValueError("Predictions or signed residuals are incomplete or inconsistent")
    groups = (
        interval["segment_id"].ne(interval["segment_id"].shift())
        | interval["Date and time"].diff().ne(pd.Timedelta(minutes=10))
    ).cumsum()
    runs = [run for _, run in interval.groupby(groups)]
    if any(not run["Date and time"].diff().iloc[1:].eq(pd.Timedelta(minutes=10)).all() for run in runs):
        raise ValueError("A prediction run contains a timestamp gap")
    if len(interval) != 7 * 144 or len(runs) != 1:
        raise ValueError("Selected week must contain one complete 10-minute prediction run")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax_signal, ax_error) = plt.subplots(
        2, 1, sharex=True, figsize=(7.2, 4.7),
        gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.08},
    )
    for i, run in enumerate(runs):
        time = run["Date and time"]
        ax_signal.plot(time, run[true_col], color="#0057B8", lw=0.9,
                       label="Measured" if i == 0 else None)
        ax_signal.plot(time, run[pred_col], color="#F28E2B", lw=0.8, alpha=0.85,
                       label="Prediction" if i == 0 else None)
    ax_signal.set_ylabel("Temperature (°C)")
    ax_signal.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.09),
                     ncol=2, fontsize=9)

    error = interval[error_col].to_numpy()
    ax_error.axhline(0, color="#595959", lw=0.9)
    for run in runs:
        time = run["Date and time"]
        run_error = run[error_col]
        ax_error.plot(time, run_error, color="#2A9D8F", lw=0.75)
    ax_error.set_ylabel("Signed residual (°C)")
    ax_error.set_xlabel("25-31 October 2021")
    ax_error.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax_error.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax_signal.set_xlim(START, END)
    for ax in (ax_signal, ax_error):
        ax.grid(axis="y", color="#D9DEE2", lw=0.65)
        ax.tick_params(axis="both", labelsize=9)

    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.13, top=0.91, hspace=0.08)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "gru_prediction_residual_example"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"{TURBINE}, {TARGET}, {START.date()}, {len(interval)} points in {len(runs)} runs")
    print(f"Mean absolute residual: {np.mean(np.abs(error)):.3f} °C")
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
