from __future__ import annotations

import argparse
import json
from pathlib import Path

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
        "--min-consecutive",
        type=int,
        default=6,
        help="Minimum consecutive BCAD anomaly points required to raise an alarm.",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=7,
        help="A fault is detected if an alarm occurs this many days before its start.",
    )
    return parser.parse_args()


def format_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def main() -> None:
    args = parse_args()
    if args.window_size < 2:
        raise ValueError("--window-size must be at least 2")
    if not 0 < args.quantile < 1:
        raise ValueError("--quantile must be between 0 and 1")
    if args.variance_multiplier <= 1:
        raise ValueError("--variance-multiplier must be greater than 1")
    if args.min_consecutive < 1:
        raise ValueError("--min-consecutive must be at least 1")
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
    thresholds = fit_bcad_thresholds(
        val_predictions,
        target_cols=target_cols,
        window_size=args.window_size,
        quantile=args.quantile,
        variance_multiplier=args.variance_multiplier,
    )
    detections = apply_bcad_detection(
        test_predictions,
        thresholds=thresholds,
        target_cols=target_cols,
        min_consecutive=args.min_consecutive,
    )

    suffix = (
        f"abs_w{args.window_size}_q{int(args.quantile * 1000):03d}_"
        f"vm{format_number(args.variance_multiplier)}_c{args.min_consecutive}"
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
    detections.to_csv(detection_path, index=False, encoding="utf-8-sig")
    performance_path.write_text(json.dumps(performance, indent=2), encoding="utf-8")

    metadata["bcad_detection"] = {
        "residual_mode": "absolute",
        "score_type": "posterior_abnormal_probability",
        "window_size": args.window_size,
        "quantile": args.quantile,
        "variance_multiplier": args.variance_multiplier,
        "min_consecutive": args.min_consecutive,
        "horizon_days": args.horizon_days,
        "thresholds_path": str(threshold_path.relative_to(project_root)),
        "detections_path": str(detection_path.relative_to(project_root)),
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
