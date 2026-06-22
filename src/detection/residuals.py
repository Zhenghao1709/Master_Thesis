# src/detection/residuals.py

from __future__ import annotations

import numpy as np
import pandas as pd


def add_residual_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()

    out["residual"] = np.abs(out["y_true"] - out["y_pred"])
    out["error"] = out["y_true"] - out["y_pred"]
    out["squared_error"] = out["error"] ** 2

    return out


def compute_regression_metrics(pred_df: pd.DataFrame) -> dict:
    error = pred_df["y_true"] - pred_df["y_pred"]

    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
    }