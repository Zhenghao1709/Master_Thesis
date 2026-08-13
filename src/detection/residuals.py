# src/detection/residuals.py

from __future__ import annotations

import numpy as np
import pandas as pd


def add_residual_columns(
    pred_df: pd.DataFrame,
    target_cols: list[str],
) -> pd.DataFrame:
    out = pred_df.copy()

    for target_col in target_cols:
        error = out[f"y_true::{target_col}"] - out[f"y_pred::{target_col}"]
        out[f"residual::{target_col}"] = np.abs(error)
        out[f"error::{target_col}"] = error
        out[f"squared_error::{target_col}"] = error ** 2

    return out


def compute_regression_metrics(
    pred_df: pd.DataFrame,
    target_cols: list[str],
) -> dict:
    per_target = {}
    for target_col in target_cols:
        error = pred_df[f"y_true::{target_col}"] - pred_df[f"y_pred::{target_col}"]
        mse = np.mean(error ** 2)
        per_target[target_col] = {
            "mae": float(np.mean(np.abs(error))),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
        }

    macro_average = {
        metric: float(np.mean([values[metric] for values in per_target.values()]))
        for metric in ("mae", "mse", "rmse")
    }
    return {
        "per_target": per_target,
        "macro_average": macro_average,
    }


def base_signal_name(signal: str) -> str:
    """Return the readable signal name without the unit suffix."""
    return signal.split("(", 1)[0].strip()


def resolve_prediction_column(
    pred_df: pd.DataFrame,
    prefix: str,
    target_col: str,
) -> str:
    """
    Locate a prediction/residual column even when the degree symbol was decoded
    differently between old CSV files and current configuration strings.
    """
    exact_col = f"{prefix}::{target_col}"
    if exact_col in pred_df.columns:
        return exact_col

    target_base = base_signal_name(target_col)
    candidates = [col for col in pred_df.columns if col.startswith(f"{prefix}::")]
    for col in candidates:
        candidate_target = col.split("::", 1)[1]
        if base_signal_name(candidate_target) == target_base:
            return col

    raise KeyError(f"Could not find {prefix} column for target: {target_col}")


def fit_residual_quantile_thresholds(
    validation_predictions: pd.DataFrame,
    target_cols: list[str],
    quantile: float = 0.95,
) -> pd.DataFrame:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")

    rows = []
    for target_col in target_cols:
        residual_col = resolve_prediction_column(validation_predictions, "residual", target_col)
        residuals = pd.to_numeric(validation_predictions[residual_col], errors="coerce").dropna()
        if residuals.empty:
            raise ValueError(f"No residual values found for target: {target_col}")
        rows.append(
            {
                "target": target_col,
                "residual_column": residual_col,
                "quantile": quantile,
                "threshold": float(residuals.quantile(quantile)),
                "validation_samples": int(len(residuals)),
            }
        )

    return pd.DataFrame(rows)


def apply_residual_baseline_detection(
    predictions: pd.DataFrame,
    thresholds: pd.DataFrame,
    target_cols: list[str],
    min_consecutive: int = 1,
) -> pd.DataFrame:
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1")

    required_cols = ["Date and time", "turbine_id", "segment_id"]
    missing_cols = [col for col in required_cols if col not in predictions.columns]
    if missing_cols:
        raise ValueError(f"Missing required prediction columns: {missing_cols}")

    threshold_by_target = dict(zip(thresholds["target"], thresholds["threshold"]))
    rows = []

    for target_col in target_cols:
        true_col = resolve_prediction_column(predictions, "y_true", target_col)
        pred_col = resolve_prediction_column(predictions, "y_pred", target_col)
        residual_col = resolve_prediction_column(predictions, "residual", target_col)
        error_col = resolve_prediction_column(predictions, "error", target_col)
        threshold = threshold_by_target[target_col]

        part = predictions[
            ["Date and time", "turbine_id", "segment_id", true_col, pred_col, residual_col, error_col]
        ].copy()
        part = part.rename(
            columns={
                true_col: "measured",
                pred_col: "predicted",
                residual_col: "residual",
                error_col: "error",
            }
        )
        part["target"] = target_col
        part["threshold"] = threshold
        part["is_anomaly"] = part["residual"] > threshold
        rows.append(part)

    out = pd.concat(rows, ignore_index=True)
    out["Date and time"] = pd.to_datetime(out["Date and time"], errors="coerce")
    out = out.sort_values(["turbine_id", "target", "segment_id", "Date and time"]).reset_index(drop=True)

    out["consecutive_anomaly_count"] = 0
    for _, idx in out.groupby(["turbine_id", "target", "segment_id"], sort=False).groups.items():
        anomaly = out.loc[idx, "is_anomaly"].to_numpy(dtype=bool)
        counts = []
        current = 0
        for flag in anomaly:
            current = current + 1 if flag else 0
            counts.append(current)
        out.loc[idx, "consecutive_anomaly_count"] = counts

    out["is_alarm"] = out["consecutive_anomaly_count"] >= min_consecutive
    return out[
        [
            "Date and time",
            "turbine_id",
            "segment_id",
            "target",
            "measured",
            "predicted",
            "error",
            "residual",
            "threshold",
            "is_anomaly",
            "consecutive_anomaly_count",
            "is_alarm",
        ]
    ]
