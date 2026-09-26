from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.config.kelmarsh_config import TARGET_COLS
from src.detection.bcad import apply_bcad_detection, fit_bcad_thresholds
from main_detect_residual_baseline import (
    annotate_alarm_episode_horizon_matches,
    annotate_alarm_episode_operating_context,
    annotate_alarm_horizon_matches,
    build_alarm_episodes,
    build_target_event_summary,
    find_experiment,
    load_prediction_file,
    summarize_detection_performance,
    update_event_summary_from_alarm_episodes,
)


BCAD_MIN_CONSECUTIVE = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BCAD detection on 2023-2024 test predictions."
    )
    parser.add_argument("--run-id", help="Experiment run ID. Defaults to the newest experiment.")
    parser.add_argument(
        "--window-size",
        type=int,
        default=36,
        help="Sliding residual window size used for each BCAD score.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.95,
        help="Validation BCAD score quantile used as the anomaly threshold.",
    )
    parser.add_argument(
        "--variance-multiplier",
        type=float,
        default=5.0,
        help="Abnormal-model variance multiplier relative to the healthy residual variance.",
    )
    parser.add_argument(
        "--abnormal-mean-multiplier",
        type=float,
        help="Optional multiplier for the abnormal Gaussian mean in absolute Gaussian BCAD.",
    )
    parser.add_argument(
        "--residual-mode",
        choices=["signed_error", "absolute"],
        default="signed_error",
        help="Residual representation used for BCAD scoring.",
    )
    parser.add_argument(
        "--distribution",
        choices=[
            "gaussian",
            "folded_normal",
            "folded_normal_mle_shared_center",
            "folded_normal_mle_legacy_a",
            "folded_normal_h_gaussian_a",
            "weibull_h_weibull_a_mle_calibration",
            "weibull_h_weibull_a_mle_folded_calibration",
            "weibull_h_weibull_a_joint_mle_calibration",
        ],
        default="gaussian",
        help="Likelihood distribution used for BCAD scoring.",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=7,
        help="A fault is detected if an alarm occurs this many days before its start.",
    )
    parser.add_argument(
        "--skip-detections-file",
        action="store_true",
        help="Compute all metrics but do not save the large point-level detection CSV.",
    )
    return parser.parse_args()


def format_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def ensure_mle_parameter_file(
    project_root: Path,
    result_dir: Path,
    run_id: str,
    window_size: int,
    horizon_days: int,
) -> Path:
    path = result_dir / f"bcad_weibull_mle_parameters_calib2021_2022_w{window_size}.csv"
    if path.exists():
        return path
    command = [
        sys.executable,
        str(project_root / "scripts" / "experiments" / "main_fit_bcad_weibull_mle.py"),
        "--run-id",
        run_id,
        "--window-size",
        str(window_size),
        "--horizon-days",
        str(horizon_days),
    ]
    print("MLE Weibull parameters are missing; fitting them first.")
    subprocess.run(command, cwd=project_root, check=True)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def ensure_point_mle_parameter_file(project_root: Path, result_dir: Path, run_id: str) -> Path:
    path = result_dir / "bcad_weibull_point_mle_parameters_calib2021_2022.csv"
    if path.exists():
        return path
    command = [
        sys.executable,
        str(project_root / "scripts" / "experiments" / "main_fit_bcad_weibull_point_mle.py"),
        "--run-id",
        run_id,
    ]
    print("Point-level MLE Weibull parameters are missing; fitting them first.")
    subprocess.run(command, cwd=project_root, check=True)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def ensure_folded_calibrated_weibull_parameter_file(
    project_root: Path, result_dir: Path, run_id: str, window_size: int
) -> Path:
    if window_size != 72:
        raise ValueError("Folded-calibrated Weibull parameters are currently fitted for window_size=72")
    path = result_dir / (
        "bcad_weibull_mle_parameters_from_foldedmle_"
        "foldedmle_samecenterabs_w72_q990_vm5_c6_calib2021_2022.csv"
    )
    if not path.exists():
        subprocess.run(
            [sys.executable, str(project_root / "scripts" / "experiments" /
             "main_fit_bcad_weibull_from_folded_mle.py"), "--run-id", run_id],
            cwd=project_root, check=True,
        )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def ensure_folded_mle_parameter_file(project_root: Path, result_dir: Path, run_id: str) -> Path:
    path = result_dir / "bcad_folded_normal_mle_healthy_parameters.csv"
    if path.exists():
        return path
    command = [
        sys.executable,
        str(project_root / "scripts" / "experiments" / "main_fit_bcad_folded_normal_mle.py"),
        "--run-id",
        run_id,
    ]
    print("Healthy folded-normal MLE parameters are missing; fitting them first.")
    subprocess.run(command, cwd=project_root, check=True)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    args = parse_args()
    if args.window_size < 2:
        raise ValueError("--window-size must be at least 2")
    if not 0 < args.quantile < 1:
        raise ValueError("--quantile must be between 0 and 1")
    if args.variance_multiplier <= 0:
        raise ValueError("--variance-multiplier must be greater than 0")
    if args.abnormal_mean_multiplier is not None and args.abnormal_mean_multiplier <= 0:
        raise ValueError("--abnormal-mean-multiplier must be greater than 0")
    if args.abnormal_mean_multiplier is not None and not (
        args.residual_mode == "absolute" and args.distribution == "gaussian"
    ):
        raise ValueError("--abnormal-mean-multiplier is only supported for absolute Gaussian BCAD")
    if args.distribution in {
        "folded_normal",
        "folded_normal_mle_shared_center",
        "folded_normal_mle_legacy_a",
        "folded_normal_h_gaussian_a",
        "weibull_h_weibull_a_mle_calibration",
        "weibull_h_weibull_a_mle_folded_calibration",
        "weibull_h_weibull_a_joint_mle_calibration",
    } and args.residual_mode != "absolute":
        raise ValueError("This distribution requires --residual-mode absolute")
    if args.horizon_days < 1:
        raise ValueError("--horizon-days must be at least 1")

    project_root = Path(__file__).resolve().parent
    result_dir, metadata_path, metadata = find_experiment(project_root, args.run_id)

    val_predictions = load_prediction_file(
        project_root,
        metadata,
        path_key="predictions",
        fallback_name="healthy_val_predictions.csv",
    )
    test_predictions = load_prediction_file(
        project_root,
        metadata,
        path_key="test_predictions",
        fallback_name="test_2023_2024_predictions.csv",
    )

    target_cols = metadata.get("targets", TARGET_COLS)
    weibull_parameters = None
    weibull_parameter_path = None
    folded_mle_parameters = None
    folded_mle_parameter_path = None
    if args.distribution == "weibull_h_weibull_a_mle_calibration":
        weibull_parameter_path = ensure_mle_parameter_file(
            project_root,
            result_dir,
            metadata["run_id"],
            args.window_size,
            args.horizon_days,
        )
        weibull_parameters = pd.read_csv(weibull_parameter_path, encoding="utf-8-sig")
    elif args.distribution == "weibull_h_weibull_a_mle_folded_calibration":
        weibull_parameter_path = ensure_folded_calibrated_weibull_parameter_file(
            project_root, result_dir, metadata["run_id"], args.window_size
        )
        weibull_parameters = pd.read_csv(weibull_parameter_path, encoding="utf-8-sig")
    elif args.distribution == "weibull_h_weibull_a_joint_mle_calibration":
        weibull_parameter_path = ensure_point_mle_parameter_file(
            project_root, result_dir, metadata["run_id"]
        )
        weibull_parameters = pd.read_csv(weibull_parameter_path, encoding="utf-8-sig")
    elif args.distribution in {"folded_normal_mle_shared_center", "folded_normal_mle_legacy_a"}:
        folded_mle_parameter_path = ensure_folded_mle_parameter_file(
            project_root, result_dir, metadata["run_id"]
        )
        folded_mle_parameters = pd.read_csv(folded_mle_parameter_path, encoding="utf-8-sig")
    thresholds = fit_bcad_thresholds(
        val_predictions,
        target_cols=target_cols,
        window_size=args.window_size,
        quantile=args.quantile,
        variance_multiplier=args.variance_multiplier,
        residual_mode=args.residual_mode,
        distribution=args.distribution,
        abnormal_mean_multiplier=args.abnormal_mean_multiplier,
        weibull_parameters=weibull_parameters,
        folded_mle_parameters=folded_mle_parameters,
    )
    detections = apply_bcad_detection(
        test_predictions,
        thresholds=thresholds,
        target_cols=target_cols,
        min_consecutive=BCAD_MIN_CONSECUTIVE,
    )

    if args.distribution == "weibull_h_weibull_a_mle_calibration":
        mode_label = "weibullmle_calib2021_2022"
    elif args.distribution == "weibull_h_weibull_a_mle_folded_calibration":
        mode_label = "weibullmle_foldedcalib2021_2022"
    elif args.distribution == "weibull_h_weibull_a_joint_mle_calibration":
        mode_label = "weibulljointmle_calib2021_2022"
    elif args.distribution == "folded_normal_mle_shared_center":
        mode_label = "foldedmle_samecenterabs"
    elif args.distribution == "folded_normal_mle_legacy_a":
        mode_label = "foldedmle_legacyAabs"
    elif args.distribution == "folded_normal_h_gaussian_a":
        mode_label = "hybridabs"
    elif args.distribution == "folded_normal":
        mode_label = "foldedabs"
    else:
        mode_label = "abs" if args.residual_mode == "absolute" else "signed"
    if args.abnormal_mean_multiplier is not None:
        mode_label = f"{mode_label}_am{format_number(args.abnormal_mean_multiplier)}"
    variance_label = (
        "" if args.distribution in {
            "weibull_h_weibull_a_mle_calibration",
            "weibull_h_weibull_a_mle_folded_calibration",
            "weibull_h_weibull_a_joint_mle_calibration",
        }
        else f"_vm{format_number(args.variance_multiplier)}"
    )
    suffix = (
        f"{mode_label}_w{args.window_size}_q{int(args.quantile * 1000):03d}"
        f"{variance_label}_c{BCAD_MIN_CONSECUTIVE}"
    )
    threshold_path = result_dir / f"bcad_thresholds_{suffix}.csv"
    detection_path = result_dir / f"bcad_detections_{suffix}.csv"
    event_summary_path = result_dir / f"bcad_event_summary_{suffix}.csv"
    episode_path = result_dir / f"bcad_alarm_episodes_{suffix}.csv"
    performance_path = result_dir / f"bcad_performance_{suffix}.json"

    thresholds.to_csv(threshold_path, index=False, encoding="utf-8-sig")

    event_summary = build_target_event_summary(project_root, horizon_days=args.horizon_days)
    detections = annotate_alarm_horizon_matches(detections, event_summary)
    alarm_episodes = build_alarm_episodes(detections)
    alarm_episodes = annotate_alarm_episode_horizon_matches(alarm_episodes, event_summary)
    alarm_episodes = annotate_alarm_episode_operating_context(alarm_episodes, project_root)
    event_summary = update_event_summary_from_alarm_episodes(event_summary, alarm_episodes)
    performance = summarize_detection_performance(detections, event_summary, alarm_episodes)

    event_summary.to_csv(event_summary_path, index=False, encoding="utf-8-sig")
    alarm_episodes.to_csv(episode_path, index=False, encoding="utf-8-sig")
    if not args.skip_detections_file:
        detections.to_csv(detection_path, index=False, encoding="utf-8-sig")
    performance_path.write_text(json.dumps(performance, indent=2), encoding="utf-8")

    metadata["bcad_detection"] = {
        "residual_mode": args.residual_mode,
        "score_type": (
            "window_log_likelihood_ratio"
            if args.distribution in {
                "weibull_h_weibull_a_joint_mle_calibration",
                "folded_normal_mle_shared_center",
                "folded_normal_mle_legacy_a",
            }
            else "posterior_abnormal_probability"
        ),
        "distribution": args.distribution,
        "window_size": args.window_size,
        "quantile": args.quantile,
        "variance_multiplier": (
            None
            if args.distribution == "weibull_h_weibull_a_joint_mle_calibration"
            else args.variance_multiplier
        ),
        "abnormal_mean_multiplier": args.abnormal_mean_multiplier,
        "min_consecutive": BCAD_MIN_CONSECUTIVE,
        "episode_definition": "continuous is_alarm points after the c=6 rule",
        "episode_trigger_time": "episode start (the sixth consecutive anomaly point)",
        "weibull_parameter_path": (
            str(weibull_parameter_path.relative_to(project_root))
            if weibull_parameter_path is not None
            else None
        ),
        "folded_mle_parameter_path": (
            str(folded_mle_parameter_path.relative_to(project_root))
            if folded_mle_parameter_path is not None
            else None
        ),
        "horizon_days": args.horizon_days,
        "thresholds_path": str(threshold_path.relative_to(project_root)),
        "detections_path": (
            None
            if args.skip_detections_file
            else str(detection_path.relative_to(project_root))
        ),
        "event_summary_path": str(event_summary_path.relative_to(project_root)),
        "alarm_episodes_path": str(episode_path.relative_to(project_root)),
        "performance_path": str(performance_path.relative_to(project_root)),
        "performance": performance,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    alarm_rate = float(detections["is_alarm"].mean())
    detected_count = int(event_summary["detected_in_horizon"].sum())
    print("Run ID:", metadata["run_id"])
    print("Method: BCAD")
    print("Setting:", suffix)
    print("Thresholds saved to:", threshold_path)
    if args.skip_detections_file:
        print("Point-level detection CSV skipped.")
    else:
        print("Detections saved to:", detection_path)
    print("Alarm rate:", f"{alarm_rate:.4%}")
    print("Detected events in horizon:", f"{detected_count}/{len(event_summary)}")
    print("Performance summary saved to:", performance_path)
    print("Miss rate:", f"{performance['event_level']['miss_rate']:.4%}")
    print(
        "Operational false alarm rate:",
        f"{performance['alarm_episode_level']['operational_false_alarm_rate']:.4%}",
    )


if __name__ == "__main__":
    main()
