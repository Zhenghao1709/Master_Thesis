from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config.kelmarsh_config import TARGET_COLS
from src.detection.cusum import apply_cusum_detection, fit_cusum_reference_stats
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
        description="Run two-sided CUSUM detection on 2023-2024 test predictions."
    )
    parser.add_argument("--run-id", help="Experiment run ID. Defaults to the newest experiment.")
    parser.add_argument(
        "--reference-value",
        type=float,
        default=0.5,
        help="Reference value k used in the standardized CUSUM update.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=5.0,
        help="Fixed decision threshold h for the positive and negative CUSUM statistics.",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        help="Use this validation CUSUM score quantile as the decision threshold instead of fixed h.",
    )
    parser.add_argument(
        "--scale-type",
        choices=["std", "mad"],
        default="std",
        help="How to standardize validation/test errors before CUSUM.",
    )
    parser.add_argument(
        "--min-consecutive",
        type=int,
        default=6,
        help="Minimum consecutive CUSUM anomaly points required to raise an alarm.",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=7,
        help="A fault is detected if an alarm occurs this many days before its start.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not reset CUSUM statistics after an alarm point.",
    )
    return parser.parse_args()


def format_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def main() -> None:
    args = parse_args()
    if args.reference_value < 0:
        raise ValueError("--reference-value must be non-negative")
    if args.threshold_quantile is None and args.decision_threshold <= 0:
        raise ValueError("--decision-threshold must be positive")
    if args.threshold_quantile is not None and not 0 < args.threshold_quantile < 1:
        raise ValueError("--threshold-quantile must be between 0 and 1")
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
    reference_stats = fit_cusum_reference_stats(
        val_predictions,
        target_cols=target_cols,
        reference_value=args.reference_value,
        decision_threshold=args.decision_threshold,
        threshold_quantile=args.threshold_quantile,
        scale_type=args.scale_type,
    )
    detections = apply_cusum_detection(
        test_predictions,
        reference_stats=reference_stats,
        target_cols=target_cols,
        min_consecutive=args.min_consecutive,
        reset_on_alarm=not args.no_reset,
    )

    reset_label = "reset" if not args.no_reset else "noreset"
    threshold_label = (
        f"q{format_number(args.threshold_quantile * 1000)}"
        if args.threshold_quantile is not None
        else f"h{format_number(args.decision_threshold)}"
    )
    suffix = (
        f"twosided_{args.scale_type}_k{format_number(args.reference_value)}_"
        f"{threshold_label}_c{args.min_consecutive}_{reset_label}"
    )
    threshold_path = result_dir / f"cusum_thresholds_{suffix}.csv"
    detection_path = result_dir / f"cusum_detections_{suffix}.csv"
    event_summary_path = result_dir / f"cusum_event_summary_{suffix}.csv"
    episode_path = result_dir / f"cusum_alarm_episodes_{suffix}.csv"
    performance_path = result_dir / f"cusum_performance_{suffix}.json"

    reference_stats.to_csv(threshold_path, index=False, encoding="utf-8-sig")

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

    metadata["cusum_detection"] = {
        "sides": "two-sided",
        "center_type": "median" if args.scale_type == "mad" else "mean",
        "scale_type": args.scale_type,
        "reference_value": args.reference_value,
        "decision_threshold": (
            float(reference_stats["decision_threshold"].max())
            if args.threshold_quantile is not None
            else args.decision_threshold
        ),
        "threshold_type": "validation_cusum_quantile" if args.threshold_quantile is not None else "fixed",
        "threshold_quantile": args.threshold_quantile,
        "min_consecutive": args.min_consecutive,
        "horizon_days": args.horizon_days,
        "reset_on_alarm": not args.no_reset,
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
    print("Method: CUSUM")
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
