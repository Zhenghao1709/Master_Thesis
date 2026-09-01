from __future__ import annotations

import numpy as np
import pandas as pd

from src.detection.residuals import resolve_prediction_column


def _bcad_scores_for_series(
    values: pd.Series,
    center: float,
    scale: float,
    variance_multiplier: float,
    window_size: int,
) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    deviation_sq = (values - center) ** 2
    rolling_ss = deviation_sq.rolling(window=window_size, min_periods=window_size).sum()

    healthy_std = scale
    abnormal_std = scale * np.sqrt(variance_multiplier)
    constant = window_size * np.log(abnormal_std / healthy_std)
    log_h_minus_log_a = (
        -0.5 * rolling_ss / (healthy_std**2)
        + 0.5 * rolling_ss / (abnormal_std**2)
        + constant
    )
    log_h_minus_log_a = log_h_minus_log_a.clip(lower=-700, upper=700)
    return 1.0 / (1.0 + np.exp(log_h_minus_log_a))


def add_bcad_scores(
    predictions: pd.DataFrame,
    target_cols: list[str],
    reference_stats: pd.DataFrame,
) -> pd.DataFrame:
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
        part["threshold"] = float(stats["threshold"])
        part["window_size"] = int(stats["window_size"])
        part["bcad_score"] = np.nan
        rows.append(part)

    out = pd.concat(rows, ignore_index=True)
    out["Date and time"] = pd.to_datetime(out["Date and time"], errors="coerce")
    out = out.sort_values(["turbine_id", "target", "segment_id", "Date and time"]).reset_index(drop=True)

    for (_, target_col, _), idx in out.groupby(["turbine_id", "target", "segment_id"], sort=False).groups.items():
        stats = stats_by_target[target_col]
        group_index = list(idx)
        scores = _bcad_scores_for_series(
            out.loc[group_index, "residual"],
            center=float(stats["center"]),
            scale=float(stats["scale"]),
            variance_multiplier=float(stats["variance_multiplier"]),
            window_size=int(stats["window_size"]),
        )
        out.loc[group_index, "bcad_score"] = scores.to_numpy()

    return out


def fit_bcad_thresholds(
    validation_predictions: pd.DataFrame,
    target_cols: list[str],
    window_size: int = 36,
    quantile: float = 0.95,
    variance_multiplier: float = 5.0,
) -> pd.DataFrame:
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    if variance_multiplier <= 1:
        raise ValueError("variance_multiplier must be greater than 1")

    base_rows = []
    for target_col in target_cols:
        residual_col = resolve_prediction_column(validation_predictions, "residual", target_col)
        residuals = pd.to_numeric(validation_predictions[residual_col], errors="coerce").dropna()
        if residuals.empty:
            raise ValueError(f"No residual values found for target: {target_col}")

        center = float(residuals.mean())
        scale = float(residuals.std(ddof=0))
        if scale == 0 or np.isnan(scale):
            raise ValueError(f"Validation residual std is zero or NaN for target: {target_col}")

        base_rows.append(
            {
                "target": target_col,
                "residual_column": residual_col,
                "method": "bcad",
                "residual_mode": "absolute",
                "score_type": "posterior_abnormal_probability",
                "center_type": "mean",
                "scale_type": "std",
                "center": center,
                "scale": scale,
                "window_size": int(window_size),
                "variance_multiplier": float(variance_multiplier),
                "threshold_quantile": float(quantile),
                "threshold": np.nan,
                "validation_samples": int(len(residuals)),
                "validation_score_samples": 0,
            }
        )

    thresholds = pd.DataFrame(base_rows)
    scored = add_bcad_scores(validation_predictions, target_cols, thresholds)
    for index, row in thresholds.iterrows():
        scores = pd.to_numeric(
            scored.loc[scored["target"] == row["target"], "bcad_score"],
            errors="coerce",
        ).dropna()
        if scores.empty:
            raise ValueError(f"No validation BCAD scores found for target: {row['target']}")
        thresholds.loc[index, "threshold"] = float(scores.quantile(quantile))
        thresholds.loc[index, "validation_score_samples"] = int(len(scores))

    return thresholds


def apply_bcad_detection(
    predictions: pd.DataFrame,
    thresholds: pd.DataFrame,
    target_cols: list[str],
    min_consecutive: int = 1,
) -> pd.DataFrame:
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1")

    out = add_bcad_scores(predictions, target_cols, thresholds)
    out["is_anomaly"] = out["bcad_score"] > out["threshold"]
    out["is_anomaly"] = out["is_anomaly"].fillna(False)
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
            "bcad_score",
            "threshold",
            "window_size",
            "is_anomaly",
            "consecutive_anomaly_count",
            "is_alarm",
        ]
    ]
