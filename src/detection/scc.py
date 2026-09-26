from __future__ import annotations

import numpy as np
import pandas as pd

from src.detection.residuals import resolve_prediction_column


def fit_scc_thresholds(
    validation_predictions: pd.DataFrame,
    target_cols: list[str],
    k: float = 3.0,
    residual_mode: str = "signed_error",
) -> pd.DataFrame:
    if k < 0:
        raise ValueError("k must be non-negative")
    if residual_mode != "signed_error":
        raise ValueError("Only signed-error SCC is currently implemented.")

    rows = []
    for target_col in target_cols:
        error_col = resolve_prediction_column(validation_predictions, "error", target_col)
        errors = pd.to_numeric(validation_predictions[error_col], errors="coerce").dropna()
        if errors.empty:
            raise ValueError(f"No error values found for target: {target_col}")

        center = float(errors.mean())
        scale = float(errors.std(ddof=0))
        rows.append(
            {
                "target": target_col,
                "error_column": error_col,
                "method": "scc",
                "residual_mode": residual_mode,
                "center_type": "mean",
                "scale_type": "std",
                "center": center,
                "scale": scale,
                "k": float(k),
                "threshold": abs(center + float(k) * scale),
                "validation_samples": int(len(errors)),
            }
        )

    return pd.DataFrame(rows)


def apply_scc_detection(
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
        part["is_anomaly"] = part["error"].abs() > threshold
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
