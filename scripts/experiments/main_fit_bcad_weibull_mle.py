from __future__ import annotations

import argparse
import json
import subprocess
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
from scipy.stats import weibull_min

from main_detect_residual_baseline import (
    annotate_alarm_episode_horizon_matches,
    annotate_alarm_horizon_matches,
    build_alarm_episodes,
    build_target_event_summary,
    find_experiment,
    load_prediction_file,
    update_event_summary_from_alarm_episodes,
)
from main_predict_bcad_calibration import (
    CALIBRATION_END_EXCLUSIVE,
    CALIBRATION_START,
    OUTPUT_NAME,
)
from src.config.kelmarsh_config import TARGET_COLS
from src.detection.bcad import (
    apply_bcad_detection,
    fit_bcad_thresholds,
    fit_weibull_mle,
    rolling_absolute_error_max,
)


OLD_GAUSSIAN_WINDOW = 72
OLD_GAUSSIAN_QUANTILE = 0.99
OLD_GAUSSIAN_VARIANCE_MULTIPLIER = 5.0
OLD_GAUSSIAN_MIN_CONSECUTIVE = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit BCAD healthy and abnormal Weibull distributions by maximum likelihood. "
            "H uses healthy validation residuals; A uses old-Gaussian true-alarm residuals "
            "from the complete validation-period SCADA timeline."
        )
    )
    parser.add_argument("--run-id", help="Experiment run ID. Defaults to the newest experiment.")
    parser.add_argument("--window-size", type=int, default=72)
    parser.add_argument("--horizon-days", type=int, default=7)
    return parser.parse_args()


def ensure_calibration_predictions(
    project_root: Path,
    run_id: str,
    result_dir: Path,
) -> Path:
    path = result_dir / OUTPUT_NAME
    if path.exists():
        return path
    command = [
        sys.executable,
        str(
            project_root
            / "scripts"
            / "experiments"
            / "main_predict_bcad_calibration.py"
        ),
        "--run-id",
        run_id,
    ]
    print("Calibration predictions are missing; generating them first.")
    subprocess.run(command, cwd=project_root, check=True)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def select_true_alarm_window_max(
    rolling_max: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.Series:
    target_episodes = episodes[
        episodes["in_fault_horizon"].fillna(False).astype(bool)
    ].copy()
    if target_episodes.empty:
        return pd.Series(dtype=float)
    target_episodes["start_time"] = pd.to_datetime(target_episodes["start_time"], errors="coerce")
    target_episodes["end_time"] = pd.to_datetime(target_episodes["end_time"], errors="coerce")

    selected = []
    for turbine_id, part in rolling_max.groupby("turbine_id", sort=False):
        turbine_episodes = target_episodes[target_episodes["turbine_id"].eq(turbine_id)]
        if turbine_episodes.empty:
            continue
        mask = pd.Series(False, index=part.index)
        for episode in turbine_episodes.itertuples(index=False):
            mask |= part["Date and time"].between(
                episode.start_time,
                episode.end_time,
                inclusive="both",
            )
        selected.append(part.loc[mask, "window_max"])
    if not selected:
        return pd.Series(dtype=float)
    return pd.concat(selected, ignore_index=True).dropna()


def slugify(value: str) -> str:
    return "_".join("".join(ch if ch.isalnum() else " " for ch in value).split())


def plot_mle_fit(
    target_col: str,
    healthy: pd.Series,
    abnormal: pd.Series,
    healthy_shape: float,
    healthy_scale: float,
    abnormal_shape: float,
    abnormal_scale: float,
    window_size: int,
    output_path: Path,
    selection_label: str | None = None,
) -> None:
    combined = pd.concat([healthy, abnormal], ignore_index=True)
    x_max = float(combined.quantile(0.995))
    x = np.linspace(max(x_max / 2000.0, 1e-6), x_max, 1000)
    bins = np.linspace(0, x_max, 80)
    healthy_pdf = weibull_min.pdf(x, healthy_shape, loc=0, scale=healthy_scale)
    abnormal_pdf = weibull_min.pdf(x, abnormal_shape, loc=0, scale=abnormal_scale)
    healthy_hist, _ = np.histogram(healthy, bins=bins, density=True)
    abnormal_hist, _ = np.histogram(abnormal, bins=bins, density=True)
    y_top = 2.0 * max(
        float(np.nanmax(healthy_pdf)),
        float(np.nanmax(abnormal_pdf)),
        float(np.nanmax(healthy_hist)),
        float(np.nanmax(abnormal_hist)),
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), constrained_layout=True)
    for ax in axes:
        ax.hist(healthy, bins=bins, density=True, alpha=0.42, color="#4C78A8", label="H data: healthy validation")
        alarm_label = selection_label or "calibration TA episodes"
        ax.hist(abnormal, bins=bins, density=True, alpha=0.38, color="#E45756", label=f"A data: {alarm_label}")
        ax.plot(x, healthy_pdf, color="#1F5FA8", linewidth=2.3, label=f"H Weibull MLE: k={healthy_shape:.3f}, lambda={healthy_scale:.3f}")
        ax.plot(x, abnormal_pdf, color="#B22222", linewidth=2.3, label=f"A Weibull MLE: k={abnormal_shape:.3f}, lambda={abnormal_scale:.3f}")
        ax.set_xlabel(f"window-max absolute residual over {window_size} points")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Linear density scale")
    axes[1].set_title("Log density scale")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-5, y_top)
    axes[1].legend(loc="upper right", fontsize=8.5)
    title = "BCAD calibration Weibull MLE fit"
    if selection_label:
        title += f" | {selection_label}"
    fig.suptitle(f"{title}\n{target_col}", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.window_size < 2:
        raise ValueError("--window-size must be at least 2")

    project_root = PROJECT_ROOT
    result_dir, metadata_path, metadata = find_experiment(project_root, args.run_id)
    target_cols = metadata.get("targets", TARGET_COLS)
    validation = load_prediction_file(
        project_root,
        metadata,
        path_key="predictions",
        fallback_name="healthy_val_predictions.csv",
    )
    calibration_path = ensure_calibration_predictions(
        project_root,
        metadata["run_id"],
        result_dir,
    )
    calibration = pd.read_csv(calibration_path, encoding="utf-8-sig")
    cached_episode_path = result_dir / "bcad_calibration_old_gaussian_ta_episodes_calib2021_2022_w72.csv"
    cached_event_path = result_dir / "bcad_calibration_event_summary_calib2021_2022_w72.csv"
    old_threshold_path = result_dir / "bcad_calibration_old_gaussian_thresholds_w72_q990_vm5_c6.csv"
    if cached_episode_path.exists() and cached_event_path.exists() and old_threshold_path.exists():
        old_episodes = pd.read_csv(cached_episode_path, encoding="utf-8-sig")
        events = pd.read_csv(cached_event_path, encoding="utf-8-sig")
        old_thresholds = pd.read_csv(old_threshold_path, encoding="utf-8-sig")
        print("Reusing cached old-Gaussian calibration TA episodes:", cached_episode_path)
    else:
        old_thresholds = fit_bcad_thresholds(
            validation,
            target_cols=target_cols,
            window_size=OLD_GAUSSIAN_WINDOW,
            quantile=OLD_GAUSSIAN_QUANTILE,
            variance_multiplier=OLD_GAUSSIAN_VARIANCE_MULTIPLIER,
            residual_mode="absolute",
            distribution="gaussian",
        )
        old_detections = apply_bcad_detection(
            calibration,
            thresholds=old_thresholds,
            target_cols=target_cols,
            min_consecutive=OLD_GAUSSIAN_MIN_CONSECUTIVE,
        )
        events = build_target_event_summary(
            project_root,
            horizon_days=args.horizon_days,
            evaluation_start=CALIBRATION_START,
            evaluation_end_exclusive=CALIBRATION_END_EXCLUSIVE,
        )
        old_detections = annotate_alarm_horizon_matches(old_detections, events)
        old_episodes = build_alarm_episodes(old_detections)
        old_episodes = annotate_alarm_episode_horizon_matches(old_episodes, events)
        events = update_event_summary_from_alarm_episodes(events, old_episodes)
        old_episodes.to_csv(cached_episode_path, index=False, encoding="utf-8-sig")
        events.to_csv(cached_event_path, index=False, encoding="utf-8-sig")
        old_thresholds.to_csv(old_threshold_path, index=False, encoding="utf-8-sig")

    rows = []
    figure_paths = []
    for target_col in target_cols:
        healthy = rolling_absolute_error_max(validation, target_col, args.window_size)["window_max"]
        calibration_window_max = rolling_absolute_error_max(
            calibration,
            target_col,
            args.window_size,
        )
        abnormal = select_true_alarm_window_max(
            calibration_window_max,
            old_episodes,
        )
        if abnormal.empty:
            raise ValueError(f"No calibration true-alarm window maxima for target: {target_col}")
        healthy_shape, healthy_scale = fit_weibull_mle(healthy)
        abnormal_shape, abnormal_scale = fit_weibull_mle(abnormal)
        rows.append(
            {
                "target": target_col,
                "fit_method": "scipy.stats.weibull_min.fit with floc=0",
                "calibration_start": CALIBRATION_START,
                "calibration_end_exclusive": CALIBRATION_END_EXCLUSIVE,
                "window_size": args.window_size,
                "healthy_source": "healthy validation prediction window-max absolute residual",
                "abnormal_source": "old abs Gaussian TA episode window-max absolute residual",
                "healthy_weibull_shape": healthy_shape,
                "healthy_weibull_scale": healthy_scale,
                "weibull_shape": abnormal_shape,
                "weibull_scale": abnormal_scale,
                "healthy_samples": len(healthy),
                "abnormal_samples": len(abnormal),
                "old_gaussian_window": OLD_GAUSSIAN_WINDOW,
                "old_gaussian_quantile": OLD_GAUSSIAN_QUANTILE,
                "old_gaussian_variance_multiplier": OLD_GAUSSIAN_VARIANCE_MULTIPLIER,
                "old_gaussian_min_consecutive": OLD_GAUSSIAN_MIN_CONSECUTIVE,
            }
        )
        figure_path = (
            project_root
            / "results"
            / "kelmarsh"
            / "figures"
            / "bcad_distribution_fits"
            / f"bcad_weibull_mle_calibration_w{args.window_size}_{slugify(target_col)}.png"
        )
        plot_mle_fit(
            target_col,
            healthy,
            abnormal,
            healthy_shape,
            healthy_scale,
            abnormal_shape,
            abnormal_scale,
            args.window_size,
            figure_path,
        )
        figure_paths.append(str(figure_path.relative_to(project_root)))

    suffix = f"calib2021_2022_w{args.window_size}"
    parameters_path = result_dir / f"bcad_weibull_mle_parameters_{suffix}.csv"
    episode_path = result_dir / f"bcad_calibration_old_gaussian_ta_episodes_{suffix}.csv"
    event_path = result_dir / f"bcad_calibration_event_summary_{suffix}.csv"
    parameters = pd.DataFrame(rows)
    parameters.to_csv(parameters_path, index=False, encoding="utf-8-sig")
    old_episodes.to_csv(episode_path, index=False, encoding="utf-8-sig")
    events.to_csv(event_path, index=False, encoding="utf-8-sig")

    metadata.setdefault("bcad_calibration", {})[f"weibull_mle_w{args.window_size}"] = {
        "parameter_path": str(parameters_path.relative_to(project_root)),
        "old_gaussian_ta_episode_path": str(episode_path.relative_to(project_root)),
        "event_summary_path": str(event_path.relative_to(project_root)),
        "test_data_used": False,
        "fit_method": "maximum likelihood with Weibull location fixed at zero",
        "fit_figures": figure_paths,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(parameters.to_string(index=False))
    print("Calibration events detected by old Gaussian:", int(events["detected_in_horizon"].sum()), "/", len(events))
    print("MLE parameters saved to:", parameters_path)


if __name__ == "__main__":
    main()
