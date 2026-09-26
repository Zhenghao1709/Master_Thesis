from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import foldnorm, weibull_min

from src.detection.residuals import resolve_prediction_column


def _bcad_scores_for_series(
    values: pd.Series,
    center: float,
    scale: float,
    variance_multiplier: float,
    window_size: int,
    distribution: str = "gaussian",
    abnormal_center: float | None = None,
    abnormal_scale: float | None = None,
    weibull_shape: float | None = None,
    weibull_scale: float | None = None,
    healthy_weibull_shape: float | None = None,
    healthy_weibull_scale: float | None = None,
) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    healthy_std = scale
    abnormal_center = center if abnormal_center is None else abnormal_center
    abnormal_scale = scale if abnormal_scale is None else abnormal_scale
    abnormal_std = abnormal_scale * np.sqrt(variance_multiplier)

    if distribution == "gaussian":
        deviation_sq = (values - center) ** 2
        abnormal_deviation_sq = (values - abnormal_center) ** 2
        rolling_h_ss = deviation_sq.rolling(window=window_size, min_periods=window_size).sum()
        rolling_a_ss = abnormal_deviation_sq.rolling(window=window_size, min_periods=window_size).sum()
        constant = window_size * np.log(abnormal_std / healthy_std)
        log_h_minus_log_a = -0.5 * rolling_h_ss / (healthy_std**2) + 0.5 * rolling_a_ss / (abnormal_std**2) + constant
    elif distribution in {
        "folded_normal", "folded_normal_mle_shared_center", "folded_normal_mle_legacy_a"
    }:
        log_h = _folded_normal_logpdf(values, center, healthy_std)
        log_a = _folded_normal_logpdf(values, abnormal_center, abnormal_std)
        if distribution in {"folded_normal_mle_shared_center", "folded_normal_mle_legacy_a"}:
            return (log_a - log_h).rolling(window=window_size, min_periods=window_size).sum()
        log_h_minus_log_a = (log_h - log_a).rolling(
            window=window_size,
            min_periods=window_size,
        ).sum()
    elif distribution == "folded_normal_h_gaussian_a":
        log_h = _folded_normal_logpdf(values, center, healthy_std)
        log_a = _normal_logpdf(values, abnormal_center, abnormal_std)
        log_h_minus_log_a = (log_h - log_a).rolling(
            window=window_size,
            min_periods=window_size,
        ).sum()
    elif _is_weibull_h_weibull_a_distribution(distribution):
        if healthy_weibull_shape is None or healthy_weibull_scale is None:
            raise ValueError("Weibull H distribution requires healthy_weibull_shape and healthy_weibull_scale")
        if weibull_shape is None or weibull_scale is None:
            raise ValueError("Weibull A distribution requires weibull_shape and weibull_scale")
        if distribution == "weibull_h_weibull_a_joint_mle_calibration":
            point_log_ratio = _weibull_logpdf(
                values, shape=weibull_shape, scale=weibull_scale
            ) - _weibull_logpdf(
                values, shape=healthy_weibull_shape, scale=healthy_weibull_scale
            )
            return point_log_ratio.rolling(window=window_size, min_periods=window_size).sum()
        window_max = values.rolling(window=window_size, min_periods=window_size).max()
        log_h_minus_log_a = _weibull_logpdf(
            window_max, shape=healthy_weibull_shape, scale=healthy_weibull_scale
        ) - _weibull_logpdf(window_max, shape=weibull_shape, scale=weibull_scale)
    else:
        raise ValueError(f"Unsupported BCAD distribution: {distribution}")

    log_h_minus_log_a = log_h_minus_log_a.clip(lower=-700, upper=700)
    return 1.0 / (1.0 + np.exp(log_h_minus_log_a))


def _is_weibull_h_weibull_a_distribution(distribution: str) -> bool:
    return distribution in {
        "weibull_h_weibull_a_mle_calibration",
        "weibull_h_weibull_a_mle_folded_calibration",
        "weibull_h_weibull_a_joint_mle_calibration",
    }


def fit_weibull_mle(values: pd.Series) -> tuple[float, float]:
    """Fit a two-parameter Weibull distribution by MLE with location fixed at zero."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    clean = clean[clean > 0]
    if len(clean) < 3:
        raise ValueError("At least three positive values are required for Weibull MLE")
    shape, location, scale = weibull_min.fit(clean.to_numpy(dtype=float), floc=0.0)
    if location != 0 or not np.isfinite(shape) or not np.isfinite(scale):
        raise ValueError("Weibull MLE returned invalid parameters")
    if shape <= 0 or scale <= 0:
        raise ValueError("Weibull MLE shape and scale must be positive")
    return float(shape), float(scale)


def fit_folded_normal_mle(values: pd.Series) -> tuple[float, float]:
    """Fit |Normal(mu, sigma)| by MLE with folded-normal location fixed at zero."""
    clean = pd.to_numeric(values, errors="coerce").dropna().abs()
    if len(clean) < 3:
        raise ValueError("At least three absolute residuals are required for folded-normal MLE")
    shape, location, scale = foldnorm.fit(clean.to_numpy(dtype=float), floc=0.0)
    center = shape * scale
    if location != 0 or not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Folded-normal MLE returned invalid parameters")
    return float(center), float(scale)


def rolling_absolute_error_max(
    predictions: pd.DataFrame,
    target_col: str,
    window_size: int,
) -> pd.DataFrame:
    """Return timestamp-aligned rolling maxima without crossing turbine/segment gaps."""
    error_col = resolve_prediction_column(predictions, "error", target_col)
    required = ["Date and time", "turbine_id", "segment_id", error_col]
    out = predictions[required].copy()
    out["Date and time"] = pd.to_datetime(out["Date and time"], errors="coerce")
    out["abs_error"] = pd.to_numeric(out[error_col], errors="coerce").abs()
    out = out.dropna(subset=["Date and time", "abs_error"])
    out = out.sort_values(["turbine_id", "segment_id", "Date and time"]).reset_index(drop=True)
    out["window_max"] = (
        out.groupby(["turbine_id", "segment_id"], sort=False)["abs_error"]
        .rolling(window=window_size, min_periods=window_size)
        .max()
        .reset_index(level=[0, 1], drop=True)
    )
    return out.dropna(subset=["window_max"]).reset_index(drop=True)


def _normal_logpdf(values: pd.Series, center: float, scale: float) -> pd.Series:
    if scale <= 0:
        raise ValueError("scale must be positive")

    x = pd.to_numeric(values, errors="coerce").astype(float)
    log_density = -np.log(scale) - 0.5 * np.log(2 * np.pi) - 0.5 * ((x - center) / scale) ** 2
    return pd.Series(log_density, index=values.index)


def _folded_normal_logpdf(values: pd.Series, center: float, scale: float) -> pd.Series:
    if scale <= 0:
        raise ValueError("scale must be positive")

    x = pd.to_numeric(values, errors="coerce").astype(float)
    z_pos = -0.5 * ((x - center) / scale) ** 2
    z_neg = -0.5 * ((x + center) / scale) ** 2
    max_z = np.maximum(z_pos, z_neg)
    log_density = (
        -np.log(scale)
        -0.5 * np.log(2 * np.pi)
        + max_z
        + np.log(np.exp(z_pos - max_z) + np.exp(z_neg - max_z))
    )
    log_density = pd.Series(log_density, index=values.index)
    log_density[x < 0] = np.nan
    return log_density


def _weibull_logpdf(values: pd.Series, shape: float, scale: float) -> pd.Series:
    if shape <= 0 or scale <= 0:
        raise ValueError("Weibull shape and scale must be positive")

    x = pd.to_numeric(values, errors="coerce").astype(float)
    x_safe = x.clip(lower=1e-300)
    log_density = (
        np.log(shape)
        - np.log(scale)
        + (shape - 1.0) * (np.log(x_safe) - np.log(scale))
        - (x_safe / scale) ** shape
    )
    log_density = pd.Series(log_density, index=values.index)
    log_density[x < 0] = np.nan
    return log_density


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
        residual_mode = str(stats.get("residual_mode", "signed_error"))
        distribution = str(stats.get("distribution", "gaussian"))
        score_values = out.loc[group_index, "error"]
        if residual_mode == "absolute":
            score_values = score_values.abs()
        elif residual_mode != "signed_error":
            raise ValueError(f"Unsupported BCAD residual_mode: {residual_mode}")

        scores = _bcad_scores_for_series(
            score_values,
            center=float(stats["center"]),
            scale=float(stats["scale"]),
            variance_multiplier=float(stats["variance_multiplier"]),
            window_size=int(stats["window_size"]),
            distribution=distribution,
            abnormal_center=float(stats.get("abnormal_center", stats["center"])),
            abnormal_scale=float(stats.get("abnormal_scale", stats["scale"])),
            weibull_shape=(
                float(stats["weibull_shape"])
                if "weibull_shape" in stats and pd.notna(stats.get("weibull_shape"))
                else None
            ),
            weibull_scale=(
                float(stats["weibull_scale"])
                if "weibull_scale" in stats and pd.notna(stats.get("weibull_scale"))
                else None
            ),
            healthy_weibull_shape=(
                float(stats["healthy_weibull_shape"])
                if "healthy_weibull_shape" in stats and pd.notna(stats.get("healthy_weibull_shape"))
                else None
            ),
            healthy_weibull_scale=(
                float(stats["healthy_weibull_scale"])
                if "healthy_weibull_scale" in stats and pd.notna(stats.get("healthy_weibull_scale"))
                else None
            ),
        )
        out.loc[group_index, "bcad_score"] = scores.to_numpy()

    return out


def fit_bcad_thresholds(
    validation_predictions: pd.DataFrame,
    target_cols: list[str],
    window_size: int = 36,
    quantile: float = 0.95,
    variance_multiplier: float = 5.0,
    residual_mode: str = "signed_error",
    distribution: str = "gaussian",
    abnormal_mean_multiplier: float | None = None,
    weibull_parameters: pd.DataFrame | None = None,
    folded_mle_parameters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    if variance_multiplier <= 0:
        raise ValueError("variance_multiplier must be greater than 0")
    if residual_mode not in {"signed_error", "absolute"}:
        raise ValueError("residual_mode must be 'signed_error' or 'absolute'")
    if distribution not in {
        "gaussian",
        "folded_normal",
        "folded_normal_mle_shared_center",
        "folded_normal_mle_legacy_a",
        "folded_normal_h_gaussian_a",
        "weibull_h_weibull_a_mle_calibration",
        "weibull_h_weibull_a_mle_folded_calibration",
        "weibull_h_weibull_a_joint_mle_calibration",
    }:
        raise ValueError(
            "distribution must be 'gaussian', 'folded_normal', "
            "'folded_normal_mle_shared_center', "
            "'folded_normal_mle_legacy_a', "
            "'folded_normal_h_gaussian_a', or "
            "'weibull_h_weibull_a_mle_calibration', "
            "'weibull_h_weibull_a_mle_folded_calibration', or "
            "'weibull_h_weibull_a_joint_mle_calibration'"
        )
    if distribution in {
        "folded_normal",
        "folded_normal_mle_shared_center",
        "folded_normal_mle_legacy_a",
        "folded_normal_h_gaussian_a",
        "weibull_h_weibull_a_mle_calibration",
        "weibull_h_weibull_a_mle_folded_calibration",
        "weibull_h_weibull_a_joint_mle_calibration",
    } and residual_mode != "absolute":
        raise ValueError("This BCAD distribution requires residual_mode='absolute'")
    if abnormal_mean_multiplier is not None and abnormal_mean_multiplier <= 0:
        raise ValueError("abnormal_mean_multiplier must be greater than 0")

    base_rows = []
    for target_col in target_cols:
        error_col = resolve_prediction_column(validation_predictions, "error", target_col)
        errors = pd.to_numeric(validation_predictions[error_col], errors="coerce").dropna()
        if errors.empty:
            raise ValueError(f"No error values found for target: {target_col}")

        abs_errors = errors.abs()
        signed_center = float(errors.mean())
        signed_scale = float(errors.std(ddof=0))
        abs_center = float(abs_errors.mean())
        abs_scale = float(abs_errors.std(ddof=0))
        if signed_scale == 0 or np.isnan(signed_scale):
            raise ValueError(f"Validation error std is zero or NaN for target: {target_col}")
        if abs_scale == 0 or np.isnan(abs_scale):
            raise ValueError(f"Validation absolute error std is zero or NaN for target: {target_col}")

        if residual_mode == "absolute" and distribution == "gaussian":
            center = abs_center
            scale = abs_scale
            abnormal_center = (
                abs_center if abnormal_mean_multiplier is None else abs_center * abnormal_mean_multiplier
            )
            abnormal_scale = abs_scale
            abnormal_center_type = (
                "mean_abs_validation_error"
                if abnormal_mean_multiplier is None
                else f"{abnormal_mean_multiplier:g}_times_mean_abs_validation_error"
            )
            abnormal_scale_type = "std_abs_validation_error"
        else:
            center = signed_center
            scale = signed_scale
            abnormal_center = abs_center
            abnormal_scale = abs_scale
            abnormal_center_type = "mean_abs_validation_error"
            abnormal_scale_type = "std_abs_validation_error"
        if distribution in {"folded_normal_mle_shared_center", "folded_normal_mle_legacy_a"}:
            if folded_mle_parameters is None:
                raise ValueError("MLE folded-normal BCAD requires folded_mle_parameters")
            parameter_rows = folded_mle_parameters.loc[folded_mle_parameters["target"].eq(target_col)]
            if len(parameter_rows) != 1:
                raise ValueError(f"Expected one folded-normal MLE row for target: {target_col}")
            parameter_row = parameter_rows.iloc[0]
            center = float(parameter_row["healthy_center"])
            scale = float(parameter_row["healthy_scale"])
            if distribution == "folded_normal_mle_shared_center":
                abnormal_center = center
                abnormal_scale = scale
                abnormal_center_type = "same_as_healthy_folded_mle_center"
                abnormal_scale_type = "same_as_healthy_folded_mle_scale"
        if _is_weibull_h_weibull_a_distribution(distribution):
            if weibull_parameters is None:
                raise ValueError("MLE-calibrated Weibull BCAD requires weibull_parameters")
            parameter_rows = weibull_parameters.loc[
                weibull_parameters["target"].eq(target_col)
            ]
            if len(parameter_rows) != 1:
                raise ValueError(f"Expected one MLE parameter row for target: {target_col}")
            parameter_row = parameter_rows.iloc[0]
            healthy_weibull_shape = float(parameter_row["healthy_weibull_shape"])
            healthy_weibull_scale = float(parameter_row["healthy_weibull_scale"])
            weibull_shape = float(parameter_row["weibull_shape"])
            weibull_scale = float(parameter_row["weibull_scale"])
            sample_type = (
                "point_abs_error"
                if distribution == "weibull_h_weibull_a_joint_mle_calibration"
                else "window_max"
            )
            abnormal_center_type = f"weibull_shape_mle_from_calibration_true_alarm_{sample_type}"
            abnormal_scale_type = f"weibull_scale_mle_from_calibration_true_alarm_{sample_type}"
        else:
            weibull_shape = np.nan
            weibull_scale = np.nan
        if _is_weibull_h_weibull_a_distribution(distribution):
            center = healthy_weibull_shape
            scale = healthy_weibull_scale
        else:
            healthy_weibull_shape = np.nan
            healthy_weibull_scale = np.nan

        base_rows.append(
            {
                "target": target_col,
                "error_column": error_col,
                "method": "bcad",
                "residual_mode": residual_mode,
                "score_type": (
                    "window_log_likelihood_ratio"
                    if distribution in {
                        "weibull_h_weibull_a_joint_mle_calibration",
                        "folded_normal_mle_shared_center",
                        "folded_normal_mle_legacy_a",
                    }
                    else "posterior_abnormal_probability"
                ),
                "distribution": distribution,
                "center_type": (
                    "weibull_shape_mle_from_validation_point_abs_error"
                    if distribution == "weibull_h_weibull_a_joint_mle_calibration"
                    else (
                        "folded_normal_mle_center"
                        if distribution in {"folded_normal_mle_shared_center", "folded_normal_mle_legacy_a"}
                        else "mean"
                    )
                ),
                "scale_type": (
                    "weibull_scale_mle_from_validation_point_abs_error"
                    if distribution == "weibull_h_weibull_a_joint_mle_calibration"
                    else (
                        "folded_normal_mle_scale"
                        if distribution in {"folded_normal_mle_shared_center", "folded_normal_mle_legacy_a"}
                        else "std"
                    )
                ),
                "center": center,
                "scale": scale,
                "abnormal_center_type": abnormal_center_type,
                "abnormal_scale_type": abnormal_scale_type,
                "abnormal_center": abnormal_center,
                "abnormal_scale": abnormal_scale,
                "healthy_weibull_shape": healthy_weibull_shape,
                "healthy_weibull_scale": healthy_weibull_scale,
                "weibull_shape": weibull_shape,
                "weibull_scale": weibull_scale,
                "abnormal_mean_multiplier": abnormal_mean_multiplier,
                "window_size": int(window_size),
                "variance_multiplier": (
                    np.nan
                    if distribution == "weibull_h_weibull_a_joint_mle_calibration"
                    else float(variance_multiplier)
                ),
                "threshold_quantile": float(quantile),
                "threshold": np.nan,
                "validation_samples": int(len(errors)),
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
