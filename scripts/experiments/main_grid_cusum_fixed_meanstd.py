from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config.kelmarsh_config import TARGET_COLS
from src.detection.residuals import resolve_prediction_column
from main_detect_cusum import CUSUM_MIN_CONSECUTIVE, format_number
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


def parse_range(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("Range must contain at least one value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight fixed-threshold mean/std CUSUM k-h grid."
    )
    parser.add_argument("--run-id", help="Experiment run ID. Defaults to the newest experiment.")
    parser.add_argument(
        "--score-mode",
        choices=["two-sided", "absolute"],
        default="two-sided",
        help="Accumulate signed positive/negative errors or absolute standardized errors.",
    )
    parser.add_argument(
        "--k-values",
        default="0,1,2,3,4,5,6,7,8,9,10",
        help="Comma-separated CUSUM reference values.",
    )
    parser.add_argument(
        "--h-values",
        default="5,15,25,35,45,55",
        help="Comma-separated fixed CUSUM decision thresholds.",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=7,
        help="A fault is detected if an alarm occurs this many days before its start.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when the output summary CSV already exists.",
    )
    parser.add_argument(
        "--label-suffix",
        default="",
        help="Optional suffix appended to the output grid filename label.",
    )
    return parser.parse_args()


def fit_mean_std_reference(
    validation_predictions: pd.DataFrame,
    target_cols: list[str],
) -> dict[str, dict[str, float]]:
    stats = {}
    for target_col in target_cols:
        error_col = resolve_prediction_column(validation_predictions, "error", target_col)
        errors = pd.to_numeric(validation_predictions[error_col], errors="coerce").dropna()
        if errors.empty:
            raise ValueError(f"No error values found for target: {target_col}")
        center = float(errors.mean())
        scale = float(errors.std(ddof=0))
        if scale == 0 or np.isnan(scale):
            raise ValueError(f"Validation error std is zero or NaN for target: {target_col}")
        stats[target_col] = {"error_col": error_col, "center": center, "scale": scale}
    return stats


def prepare_test_groups(
    test_predictions: pd.DataFrame,
    target_cols: list[str],
    reference_stats: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    prepared = []
    for target_col in target_cols:
        error_col = resolve_prediction_column(test_predictions, "error", target_col)
        stats = reference_stats[target_col]
        part = test_predictions[["Date and time", "turbine_id", "segment_id", error_col]].copy()
        part["Date and time"] = pd.to_datetime(part["Date and time"], errors="coerce")
        part["standardized_error"] = (
            pd.to_numeric(part[error_col], errors="coerce") - stats["center"]
        ) / stats["scale"]
        part = part.sort_values(["turbine_id", "segment_id", "Date and time"]).reset_index(drop=True)
        for (turbine_id, segment_id), group in part.groupby(["turbine_id", "segment_id"], sort=False):
            prepared.append(
                {
                    "target": target_col,
                    "turbine_id": turbine_id,
                    "segment_id": segment_id,
                    "times": group["Date and time"].to_numpy(),
                    "z": group["standardized_error"].to_numpy(dtype=float),
                }
            )
    return prepared


def build_cusum_alarm_rows(
    prepared_groups: list[dict[str, object]],
    reference_value: float,
    decision_threshold: float,
    score_mode: str = "two-sided",
) -> pd.DataFrame:
    if score_mode not in {"two-sided", "absolute"}:
        raise ValueError(f"Unsupported CUSUM score mode: {score_mode}")
    rows = []
    for group in prepared_groups:
        positive_sum = 0.0
        negative_sum = 0.0
        absolute_sum = 0.0
        times = group["times"]
        z_values = group["z"]
        alarm_mask = np.zeros(len(z_values), dtype=bool)

        for position, z in enumerate(z_values):
            if np.isnan(z):
                positive_sum = 0.0
                negative_sum = 0.0
                absolute_sum = 0.0
                continue

            if score_mode == "absolute":
                absolute_sum = max(0.0, absolute_sum + abs(float(z)) - reference_value)
                is_alarm = absolute_sum > decision_threshold
            else:
                positive_sum = max(0.0, positive_sum + float(z) - reference_value)
                negative_sum = min(0.0, negative_sum + float(z) + reference_value)
                is_alarm = positive_sum > decision_threshold or negative_sum < -decision_threshold
            if is_alarm:
                alarm_mask[position] = True
                positive_sum = 0.0
                negative_sum = 0.0
                absolute_sum = 0.0

        if not alarm_mask.any():
            continue

        alarm_times = times[alarm_mask]
        rows.append(
            pd.DataFrame(
                {
                    "Date and time": alarm_times,
                    "turbine_id": group["turbine_id"],
                    "segment_id": group["segment_id"],
                    "target": group["target"],
                    "is_alarm": True,
                }
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=["Date and time", "turbine_id", "segment_id", "target", "is_alarm"]
        )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    k_values = parse_range(args.k_values)
    h_values = parse_range(args.h_values)
    if args.horizon_days < 1:
        raise ValueError("--horizon-days must be at least 1")

    project_root = PROJECT_ROOT
    result_dir, _, metadata = find_experiment(project_root, args.run_id)
    mode_label = "absolute_" if args.score_mode == "absolute" else ""
    grid_label = (
        f"{mode_label}fixed_meanstd_no_consecutive_"
        f"k{format_number(min(k_values))}-{format_number(max(k_values))}_"
        f"h{format_number(min(h_values))}-{format_number(max(h_values))}"
    )
    if args.label_suffix:
        grid_label = f"{grid_label}_{args.label_suffix}"
    output_path = result_dir / f"cusum_grid_{grid_label}.csv"
    if output_path.exists() and not args.force:
        print("Grid summary already exists:", output_path)
        return

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
    event_summary_base = build_target_event_summary(project_root, horizon_days=args.horizon_days)
    reference_stats = fit_mean_std_reference(val_predictions, target_cols)
    prepared_groups = prepare_test_groups(test_predictions, target_cols, reference_stats)

    rows = []
    total = len(k_values) * len(h_values)
    count = 0
    for k in k_values:
        for h in h_values:
            count += 1
            print(
                f"[{count}/{total}] CUSUM {args.score_mode} fixed mean/std k={k:g}, h={h:g}, no consecutive rule",
                flush=True,
            )
            detections = build_cusum_alarm_rows(
                prepared_groups,
                reference_value=k,
                decision_threshold=h,
                score_mode=args.score_mode,
            )
            event_summary = event_summary_base.copy()
            detections = annotate_alarm_horizon_matches(detections, event_summary)
            alarm_episodes = build_alarm_episodes(detections)
            alarm_episodes = annotate_alarm_episode_horizon_matches(alarm_episodes, event_summary)
            alarm_episodes = annotate_alarm_episode_operating_context(alarm_episodes, project_root)
            event_summary = update_event_summary_from_alarm_episodes(event_summary, alarm_episodes)
            performance = summarize_detection_performance(detections, event_summary, alarm_episodes)
            event_level = performance["event_level"]
            episode_level = performance["alarm_episode_level"]
            point_level = performance["alarm_point_level"]
            precision_denominator = episode_level["operational_alarm_opportunities"]
            precision = (
                episode_level["true_alarm_episodes"] / precision_denominator
                if precision_denominator
                else 0.0
            )
            recall = event_level["detection_rate"]
            f1_score = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            rows.append(
                {
                    "run_id": metadata["run_id"],
                    "method": f"cusum_{args.score_mode.replace('-', '_')}_fixed_meanstd",
                    "score_mode": args.score_mode,
                    "k": k,
                    "h": h,
                    "min_consecutive": CUSUM_MIN_CONSECUTIVE,
                    "consecutive_rule": "disabled",
                    "horizon_days": args.horizon_days,
                    "detected_events": event_level["detected_events"],
                    "total_events": event_level["total_events"],
                    "detection_rate": event_level["detection_rate"],
                    "recall": recall,
                    "miss_rate": event_level["miss_rate"],
                    "total_alarm_points": point_level["total_alarm_points"],
                    "total_alarm_episodes": episode_level["total_alarm_episodes"],
                    "true_alarm_episodes": episode_level["true_alarm_episodes"],
                    "operational_false_alarm_episodes": episode_level[
                        "operational_false_alarm_episodes"
                    ],
                    "operational_alarm_opportunities": episode_level[
                        "operational_alarm_opportunities"
                    ],
                    "precision": precision,
                    "f1_score": f1_score,
                    "operational_false_alarm_rate": episode_level[
                        "operational_false_alarm_rate"
                    ],
                }
            )
            pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    metadata.setdefault("cusum_grid", {})[grid_label] = {
        "score_mode": args.score_mode,
        "scale_type": "std",
        "center_type": "mean",
        "threshold_type": "fixed",
        "min_consecutive": CUSUM_MIN_CONSECUTIVE,
        "consecutive_rule": "disabled",
        "horizon_days": args.horizon_days,
        "k_values": k_values,
        "h_values": h_values,
        "path": str(output_path.relative_to(project_root)),
    }
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Saved CUSUM grid summary:", output_path)


if __name__ == "__main__":
    main()
