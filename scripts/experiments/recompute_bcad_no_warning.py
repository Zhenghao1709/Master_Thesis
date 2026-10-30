"""Re-evaluate four BCAD window scans without status Warning target events.

Results are written to a separate directory. Existing warning-inclusive scans,
calibration parameters, and figures are never overwritten.
"""

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
    annotate_alarm_episode_operating_context,
    annotate_alarm_horizon_matches,
    build_alarm_episodes,
    build_target_event_summary,
    find_experiment,
    load_prediction_file,
    update_event_summary_from_alarm_episodes,
)
from scripts.experiments.main_fit_bcad_weibull_mle import (
    CALIBRATION_END_EXCLUSIVE,
    CALIBRATION_START,
    select_true_alarm_window_max,
)
from scripts.experiments.main_fit_bcad_weibull_point_mle import select_calibration_alarm_points
from src.config.kelmarsh_config import TARGET_COLS
from src.detection.bcad import (
    apply_bcad_detection,
    fit_bcad_thresholds,
    fit_weibull_mle,
    rolling_absolute_error_max,
)
from src.detection.residuals import resolve_prediction_column


RUN = "all6_multi3_seq12_h64_l1_bs64_lr1e-03_wd0_do0_e60_p8_seed42"
WINDOWS = range(36, 1081, 36)
SUFFIXES = {
    "folded": "foldedmle_samecenterabs_w{w}_q990_vm5_c6",
    "gaussian": "abslog_w{w}_q990_vm5_c6",
    "maximum": "weibullmle_foldedcalib2021_2022_w{w}_q990_c6",
    "joint": "weibulljointmle_foldedcalib2021_2022_w{w}_q990_c6",
}
DISTRIBUTIONS = {
    "maximum": "weibull_h_weibull_a_mle_folded_calibration",
    "joint": "weibull_h_weibull_a_joint_mle_folded_calibration",
}
CALIBRATION_EPISODES = (
    "bcad_calibration_ta_episodes_foldedmle_"
    "foldedmle_samecenterabs_w72_q990_vm5_c6_calib2021_2022.csv"
)


def output_path(output: Path, variant: str, window: int, prefix: str = "bcad_performance") -> Path:
    return output / f"{prefix}_{SUFFIXES[variant].format(w=window)}.json"


def evaluate(episodes: pd.DataFrame, events: pd.DataFrame, already_classified: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    episodes = annotate_alarm_episode_horizon_matches(episodes, events)
    if not already_classified:
        episodes = annotate_alarm_episode_operating_context(episodes, PROJECT_ROOT)
    else:
        episodes["is_operational_false_alarm"] = (
            ~episodes["in_fault_horizon"].astype(bool)
            & ~episodes["in_non_operational_interval"].astype(bool)
        )
    events = update_event_summary_from_alarm_episodes(events, episodes)
    n_events = len(events)
    detected = int(events["detected_in_horizon"].sum())
    n_episodes = len(episodes)
    true = int(episodes["in_fault_horizon"].sum())
    op_false = int(episodes["is_operational_false_alarm"].sum())
    precision = true / (true + op_false) if true + op_false else 0.0
    recall = detected / n_events if n_events else 0.0
    result = {
        "target_event_definition": "fault_corrective_maintenance_selected_stop_no_warning",
        "event_level": {
            "total_events": n_events,
            "detected_events": detected,
            "missed_events": n_events - detected,
            "detection_rate": recall,
            "miss_rate": 1 - recall,
        },
        "alarm_episode_level": {
            "total_alarm_episodes": n_episodes,
            "true_alarm_episodes": true,
            "operational_false_alarm_episodes": op_false,
            "operational_alarm_opportunities": true + op_false,
            "operational_false_alarm_rate": 1 - precision if true + op_false else 0.0,
        },
        "precision": precision,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }
    return episodes, events, result


def save(output: Path, variant: str, window: int, episodes: pd.DataFrame, events: pd.DataFrame, result: dict) -> None:
    stem = SUFFIXES[variant].format(w=window)
    (output / f"bcad_alarm_episodes_{stem}.csv").write_text(episodes.to_csv(index=False), encoding="utf-8")
    (output / f"bcad_event_summary_{stem}.csv").write_text(events.to_csv(index=False), encoding="utf-8")
    output_path(output, variant, window).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{variant} W={window}: {result['event_level']['detected_events']}/459, "
          f"P={result['precision']:.4f}, F1={result['f1']:.4f}", flush=True)


def relabel_saved(result_dir: Path, output: Path, variant: str, events: pd.DataFrame) -> None:
    for window in WINDOWS:
        if output_path(output, variant, window).exists():
            continue
        source = result_dir / f"bcad_alarm_episodes_{SUFFIXES[variant].format(w=window)}.csv"
        episodes = pd.read_csv(source, encoding="utf-8-sig")
        episodes, updated_events, result = evaluate(episodes, events.copy(), already_classified=True)
        save(output, variant, window, episodes, updated_events, result)


def fit_joint_parameters(validation: pd.DataFrame, calibration: pd.DataFrame,
                         calibration_episodes: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    alarm_points = select_calibration_alarm_points(calibration, calibration_episodes)
    rows = []
    for target in targets:
        healthy = pd.to_numeric(validation[resolve_prediction_column(validation, "error", target)], errors="coerce").abs().dropna()
        abnormal = pd.to_numeric(calibration.loc[alarm_points, resolve_prediction_column(calibration, "error", target)], errors="coerce").abs().dropna()
        h_shape, h_scale = fit_weibull_mle(healthy)
        a_shape, a_scale = fit_weibull_mle(abnormal)
        rows.append({"target": target, "healthy_weibull_shape": h_shape, "healthy_weibull_scale": h_scale,
                     "weibull_shape": a_shape, "weibull_scale": a_scale,
                     "healthy_samples": len(healthy), "abnormal_samples": len(abnormal)})
    return pd.DataFrame(rows)


def fit_maximum_parameters(validation: pd.DataFrame, calibration: pd.DataFrame,
                           calibration_episodes: pd.DataFrame, targets: list[str], window: int) -> pd.DataFrame:
    rows = []
    for target in targets:
        healthy = rolling_absolute_error_max(validation, target, window)["window_max"]
        calibration_maxima = rolling_absolute_error_max(calibration, target, window)
        abnormal = select_true_alarm_window_max(calibration_maxima, calibration_episodes)
        h_shape, h_scale = fit_weibull_mle(healthy)
        a_shape, a_scale = fit_weibull_mle(abnormal)
        rows.append({"target": target, "window_size": window,
                     "healthy_weibull_shape": h_shape, "healthy_weibull_scale": h_scale,
                     "weibull_shape": a_shape, "weibull_scale": a_scale,
                     "healthy_samples": len(healthy), "abnormal_samples": len(abnormal)})
    return pd.DataFrame(rows)


def recompute_weibull(result_dir: Path, output: Path, metadata: dict,
                      variant: str, events: pd.DataFrame) -> None:
    targets = metadata.get("targets", TARGET_COLS)
    validation = load_prediction_file(PROJECT_ROOT, metadata, path_key="predictions", fallback_name="healthy_val_predictions.csv")
    calibration = pd.read_csv(result_dir / "bcad_calibration_2021-05-26_2022_predictions.csv", encoding="utf-8-sig")
    calibration_episodes = pd.read_csv(result_dir / CALIBRATION_EPISODES, encoding="utf-8-sig")
    pretest_events = build_target_event_summary(PROJECT_ROOT, 7, CALIBRATION_START, CALIBRATION_END_EXCLUSIVE)
    calibration_episodes = annotate_alarm_episode_horizon_matches(calibration_episodes, pretest_events)
    calibration_episodes.to_csv(output / "bcad_calibration_ta_episodes_no_warning.csv", index=False)
    print("Pre-test calibration true episodes:", int(calibration_episodes["in_fault_horizon"].sum()), flush=True)
    test = load_prediction_file(PROJECT_ROOT, metadata, path_key="test_predictions", fallback_name="test_2023_2024_predictions.csv")

    joint_parameters = None
    if variant == "joint":
        joint_parameters = fit_joint_parameters(validation, calibration, calibration_episodes, targets)
        joint_parameters.to_csv(output / "bcad_weibull_joint_parameters_no_warning.csv", index=False)

    for window in WINDOWS:
        if output_path(output, variant, window).exists():
            continue
        parameters = joint_parameters
        if variant == "maximum":
            parameters = fit_maximum_parameters(validation, calibration, calibration_episodes, targets, window)
            parameters.to_csv(output / f"bcad_weibull_maximum_parameters_no_warning_w{window}.csv", index=False)
        thresholds = fit_bcad_thresholds(
            validation, target_cols=targets, window_size=window, quantile=0.99,
            variance_multiplier=5, residual_mode="absolute",
            distribution=DISTRIBUTIONS[variant], weibull_parameters=parameters,
        )
        thresholds.to_csv(output / f"bcad_thresholds_{SUFFIXES[variant].format(w=window)}.csv", index=False)
        detections = apply_bcad_detection(test, thresholds, targets, min_consecutive=6)
        # Episode grouping only needs the point-level column to exist; the
        # episode-level horizon match is recomputed against the revised events.
        detections["in_fault_horizon"] = False
        episodes = build_alarm_episodes(detections)
        episodes, updated_events, result = evaluate(episodes, events.copy(), already_classified=False)
        save(output, variant, window, episodes, updated_events, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["folded", "gaussian", "maximum", "joint"], required=True)
    args = parser.parse_args()
    result_dir, _, metadata = find_experiment(PROJECT_ROOT, RUN)
    output = result_dir / "no_warning_bcad"
    output.mkdir(exist_ok=True)
    events = build_target_event_summary(PROJECT_ROOT, horizon_days=7)
    if len(events) != 459:
        raise AssertionError(f"Expected 459 target events, got {len(events)}")
    if args.mode in {"folded", "gaussian"}:
        relabel_saved(result_dir, output, args.mode, events)
    else:
        recompute_weibull(result_dir, output, metadata, args.mode, events)


if __name__ == "__main__":
    main()
