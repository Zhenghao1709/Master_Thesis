from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from main_detect_residual_baseline import (
    annotate_alarm_episode_horizon_matches,
    annotate_alarm_horizon_matches,
    build_alarm_episodes,
    build_target_event_summary,
    find_experiment,
    load_prediction_file,
    update_event_summary_from_alarm_episodes,
)
from main_predict_bcad_calibration import CALIBRATION_END_EXCLUSIVE, CALIBRATION_START
from scripts.experiments.main_fit_bcad_weibull_mle import (
    ensure_calibration_predictions,
    plot_mle_fit,
    select_true_alarm_window_max,
    slugify,
)
from src.config.kelmarsh_config import TARGET_COLS
from src.detection.bcad import apply_bcad_detection, fit_weibull_mle, rolling_absolute_error_max


SETTING = "foldedmle_samecenterabs_w72_q990_vm5_c6"
WINDOW_SIZE = 72
MIN_CONSECUTIVE = 6
HORIZON_DAYS = 7


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit window-max Weibull H/A models from folded-normal-MLE calibration TA episodes."
    )
    parser.add_argument("--run-id", help="Experiment run ID. Defaults to the newest experiment.")
    args = parser.parse_args()

    result_dir, metadata_path, metadata = find_experiment(PROJECT_ROOT, args.run_id)
    threshold_path = result_dir / f"bcad_thresholds_{SETTING}.csv"
    if not threshold_path.exists():
        raise FileNotFoundError(f"Run folded-normal-MLE BCAD first: {threshold_path}")
    thresholds = pd.read_csv(threshold_path, encoding="utf-8-sig")
    if not thresholds["distribution"].eq("folded_normal_mle_shared_center").all():
        raise ValueError("Threshold file is not the selected folded-normal-MLE setting")
    if not thresholds["window_size"].eq(WINDOW_SIZE).all():
        raise ValueError("Unexpected calibration window size")

    target_cols = metadata.get("targets", TARGET_COLS)
    validation = load_prediction_file(
        PROJECT_ROOT, metadata, path_key="predictions", fallback_name="healthy_val_predictions.csv"
    )
    calibration_path = ensure_calibration_predictions(PROJECT_ROOT, metadata["run_id"], result_dir)
    calibration = pd.read_csv(calibration_path, encoding="utf-8-sig")

    detections = apply_bcad_detection(
        calibration, thresholds=thresholds, target_cols=target_cols,
        min_consecutive=MIN_CONSECUTIVE,
    )
    events = build_target_event_summary(
        PROJECT_ROOT,
        horizon_days=HORIZON_DAYS,
        evaluation_start=CALIBRATION_START,
        evaluation_end_exclusive=CALIBRATION_END_EXCLUSIVE,
    )
    detections = annotate_alarm_horizon_matches(detections, events)
    episodes = build_alarm_episodes(detections)
    episodes = annotate_alarm_episode_horizon_matches(episodes, events)
    events = update_event_summary_from_alarm_episodes(events, episodes)

    stem = f"foldedmle_{SETTING}_calib2021_2022"
    episode_path = result_dir / f"bcad_calibration_ta_episodes_{stem}.csv"
    event_path = result_dir / f"bcad_calibration_event_summary_{stem}.csv"
    episodes.to_csv(episode_path, index=False, encoding="utf-8-sig")
    events.to_csv(event_path, index=False, encoding="utf-8-sig")

    rows = []
    figure_paths = []
    for target in target_cols:
        healthy = rolling_absolute_error_max(validation, target, WINDOW_SIZE)["window_max"]
        calibration_maxima = rolling_absolute_error_max(calibration, target, WINDOW_SIZE)
        abnormal = select_true_alarm_window_max(calibration_maxima, episodes)
        if abnormal.empty:
            raise ValueError(f"No calibration true-alarm window maxima for target: {target}")
        healthy_shape, healthy_scale = fit_weibull_mle(healthy)
        abnormal_shape, abnormal_scale = fit_weibull_mle(abnormal)
        rows.append({
            "target": target,
            "calibration_detector": "folded_normal_mle_shared_center",
            "calibration_setting": SETTING,
            "calibration_start": CALIBRATION_START,
            "calibration_end_exclusive": CALIBRATION_END_EXCLUSIVE,
            "window_size": WINDOW_SIZE,
            "fit_method": "scipy.stats.weibull_min.fit with floc=0",
            "healthy_source": "healthy validation prediction window-max absolute residual",
            "abnormal_source": "folded-normal-MLE TA episode window-max absolute residual",
            "healthy_weibull_shape": healthy_shape,
            "healthy_weibull_scale": healthy_scale,
            "weibull_shape": abnormal_shape,
            "weibull_scale": abnormal_scale,
            "healthy_samples": len(healthy),
            "abnormal_samples": len(abnormal),
            "healthy_median": float(healthy.median()),
            "abnormal_median": float(abnormal.median()),
            "healthy_q95": float(healthy.quantile(0.95)),
            "abnormal_q95": float(abnormal.quantile(0.95)),
        })
        figure_path = (
            PROJECT_ROOT / "results" / "kelmarsh" / "figures" / "bcad_distribution_fits"
            / f"bcad_weibull_mle_from_foldedmle_w{WINDOW_SIZE}_{slugify(target)}.png"
        )
        plot_mle_fit(
            target, healthy, abnormal,
            healthy_shape, healthy_scale, abnormal_shape, abnormal_scale,
            WINDOW_SIZE, figure_path, selection_label="folded-normal MLE TA episodes",
        )
        figure_paths.append(str(figure_path.relative_to(PROJECT_ROOT)))

    parameter_path = result_dir / f"bcad_weibull_mle_parameters_from_{stem}.csv"
    parameters = pd.DataFrame(rows)
    parameters.to_csv(parameter_path, index=False, encoding="utf-8-sig")
    metadata.setdefault("bcad_calibration", {})[f"weibull_mle_from_{SETTING}"] = {
        "parameter_path": str(parameter_path.relative_to(PROJECT_ROOT)),
        "alarm_episode_path": str(episode_path.relative_to(PROJECT_ROOT)),
        "event_summary_path": str(event_path.relative_to(PROJECT_ROOT)),
        "fit_figures": figure_paths,
        "test_data_used": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Calibration TA episodes:", int(episodes["in_fault_horizon"].fillna(False).sum()))
    print("Calibration events detected:", int(events["detected_in_horizon"].sum()), "/", len(events))
    print(parameters.to_string(index=False))
    print("Parameters saved to:", parameter_path)
    for path in figure_paths:
        print("Figure:", PROJECT_ROOT / path)


if __name__ == "__main__":
    main()
