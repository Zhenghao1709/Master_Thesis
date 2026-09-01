from __future__ import annotations

import numpy as np
import pandas as pd

from src.detection.residuals import resolve_prediction_column


def fit_cusum_reference_stats(
    validation_predictions: pd.DataFrame,
    target_cols: list[str],
    reference_value: float = 0.5,
    decision_threshold: float = 5.0,
    threshold_quantile: float | None = None,
    scale_type: str = "std",
) -> pd.DataFrame:
    if reference_value < 0:
        raise ValueError("reference_value must be non-negative")
    if threshold_quantile is None and decision_threshold <= 0:
        raise ValueError("decision_threshold must be positive")
    if threshold_quantile is not None and not 0 < threshold_quantile < 1:
        raise ValueError("threshold_quantile must be between 0 and 1")
    if scale_type not in {"std", "mad"}:
        raise ValueError("scale_type must be either 'std' or 'mad'")

    rows = []
    for target_col in target_cols:
        error_col = resolve_prediction_column(validation_predictions, "error", target_col)
        errors = pd.to_numeric(validation_predictions[error_col], errors="coerce").dropna()
        if errors.empty:
            raise ValueError(f"No error values found for target: {target_col}")

        if scale_type == "mad":
            center = float(errors.median())
            scale = float((errors - center).abs().median())
            center_type = "median"
        else:
            center = float(errors.mean())
            scale = float(errors.std(ddof=0))
            center_type = "mean"
        if scale == 0 or np.isnan(scale):
            raise ValueError(f"Validation error {scale_type} scale is zero or NaN for target: {target_col}")

        if threshold_quantile is not None:
            score_values = _compute_cusum_scores_for_reference(
                validation_predictions=validation_predictions,
                target_col=target_col,
                error_col=error_col,
                center=center,
                scale=scale,
                reference_value=reference_value,
            )
            if score_values.empty:
                raise ValueError(f"No validation CUSUM scores found for target: {target_col}")
            threshold = float(score_values.quantile(threshold_quantile))
            threshold_type = "validation_cusum_quantile"
        else:
            threshold = float(decision_threshold)
            threshold_type = "fixed"

        rows.append(
            {
                "target": target_col,
                "error_column": error_col,
                "method": "cusum",
                "sides": "two-sided",
                "center_type": center_type,
                "scale_type": scale_type,
                "center": center,
                "scale": scale,
                "reference_value": float(reference_value),
                "decision_threshold": threshold,
                "threshold": threshold,
                "threshold_type": threshold_type,
                "threshold_quantile": threshold_quantile,
                "validation_samples": int(len(errors)),
            }
        )

    return pd.DataFrame(rows)


def _compute_cusum_scores_for_reference(
    validation_predictions: pd.DataFrame,
    target_col: str,
    error_col: str,
    center: float,
    scale: float,
    reference_value: float,
) -> pd.Series:
    required_cols = ["Date and time", "turbine_id", "segment_id", error_col]
    missing_cols = [col for col in required_cols if col not in validation_predictions.columns]
    if missing_cols:
        raise ValueError(f"Missing required validation prediction columns: {missing_cols}")

    part = validation_predictions[required_cols].copy()
    part["Date and time"] = pd.to_datetime(part["Date and time"], errors="coerce")
    part["standardized_error"] = (pd.to_numeric(part[error_col], errors="coerce") - center) / scale
    part = part.sort_values(["turbine_id", "segment_id", "Date and time"]).reset_index(drop=True)

    scores = []
    for _, group in part.groupby(["turbine_id", "segment_id"], sort=False):
        positive_sum = 0.0
        negative_sum = 0.0
        for z in group["standardized_error"].to_numpy():
            if pd.isna(z):
                positive_sum = 0.0
                negative_sum = 0.0
                continue
            positive_sum = max(0.0, positive_sum + float(z) - reference_value)
            negative_sum = min(0.0, negative_sum + float(z) + reference_value)
            scores.append(max(positive_sum, abs(negative_sum)))

    return pd.Series(scores, dtype=float).dropna()


def apply_cusum_detection(
    predictions: pd.DataFrame,
    reference_stats: pd.DataFrame,
    target_cols: list[str],
    min_consecutive: int = 1,
    reset_on_alarm: bool = True,
) -> pd.DataFrame:
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1")

    required_cols = ["Date and time", "turbine_id", "segment_id"]
    missing_cols = [col for col in required_cols if col not in predictions.columns]
    if missing_cols:
        raise ValueError(f"Missing required prediction columns: {missing_cols}")

    stats_by_target = reference_stats.set_index("target").to_dict(orient="index")
    rows = []

    for target_col in target_cols:
        true_col = resolve_prediction_column(predictions, "y_true", target_col)
        pred_col = resolve_prediction_column(predictions, "y_pred", target_col)
        residual_col = resolve_prediction_column(predictions, "residual", target_col)
        error_col = resolve_prediction_column(predictions, "error", target_col)
        stats = stats_by_target[target_col]

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
        part["threshold"] = float(stats["decision_threshold"])
        part["cusum_positive"] = 0.0
        part["cusum_negative"] = 0.0
        part["standardized_error"] = (
            pd.to_numeric(part["error"], errors="coerce") - float(stats["center"])
        ) / float(stats["scale"])
        rows.append(part)

    out = pd.concat(rows, ignore_index=True)
    out["Date and time"] = pd.to_datetime(out["Date and time"], errors="coerce")
    out = out.sort_values(["turbine_id", "target", "segment_id", "Date and time"]).reset_index(drop=True)
    out["is_anomaly"] = False
    out["consecutive_anomaly_count"] = 0
    out["is_alarm"] = False

    for (_, target_col, _), idx in out.groupby(["turbine_id", "target", "segment_id"], sort=False).groups.items():
        stats = stats_by_target[target_col]
        reference_value = float(stats["reference_value"])
        decision_threshold = float(stats["decision_threshold"])
        positive_sum = 0.0
        negative_sum = 0.0
        consecutive = 0
        group_index = list(idx)
        z_values = out.loc[group_index, "standardized_error"].to_numpy()
        positive_values = np.zeros(len(group_index), dtype=float)
        negative_values = np.zeros(len(group_index), dtype=float)
        anomaly_values = np.zeros(len(group_index), dtype=bool)
        consecutive_values = np.zeros(len(group_index), dtype=int)
        alarm_values = np.zeros(len(group_index), dtype=bool)

        for position, z in enumerate(z_values):
            if pd.isna(z):
                positive_sum = 0.0
                negative_sum = 0.0
                consecutive = 0
                continue

            positive_sum = max(0.0, positive_sum + float(z) - reference_value)
            negative_sum = min(0.0, negative_sum + float(z) + reference_value)
            is_anomaly = positive_sum > decision_threshold or negative_sum < -decision_threshold

            consecutive = consecutive + 1 if is_anomaly else 0
            is_alarm = consecutive >= min_consecutive

            positive_values[position] = positive_sum
            negative_values[position] = negative_sum
            anomaly_values[position] = is_anomaly
            consecutive_values[position] = consecutive
            alarm_values[position] = is_alarm

            if is_alarm and reset_on_alarm:
                positive_sum = 0.0
                negative_sum = 0.0
                consecutive = 0

        out.loc[group_index, "cusum_positive"] = positive_values
        out.loc[group_index, "cusum_negative"] = negative_values
        out.loc[group_index, "is_anomaly"] = anomaly_values
        out.loc[group_index, "consecutive_anomaly_count"] = consecutive_values
        out.loc[group_index, "is_alarm"] = alarm_values

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
            "standardized_error",
            "cusum_positive",
            "cusum_negative",
            "threshold",
            "is_anomaly",
            "consecutive_anomaly_count",
            "is_alarm",
        ]
    ]
