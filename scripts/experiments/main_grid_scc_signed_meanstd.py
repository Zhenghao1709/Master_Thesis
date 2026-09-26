from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config.kelmarsh_config import TARGET_COLS
from src.detection.scc import apply_scc_detection, fit_scc_thresholds
from main_detect_residual_baseline import (
    annotate_alarm_episode_horizon_matches,
    annotate_alarm_horizon_matches,
    build_alarm_episodes,
    build_target_event_summary,
    find_experiment,
    informational_environmental_spec_intervals_from_status,
    intervals_from_flag_file,
    load_prediction_file,
    non_full_performance_intervals_from_status,
    overlaps_any_interval,
    summarize_detection_performance,
    update_event_summary_from_alarm_episodes,
)
from main_detect_scc import SCC_MIN_CONSECUTIVE, format_number


def default_k_values() -> str:
    return ",".join(f"{i / 2:g}" for i in range(21))


def parse_range(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError("Range must contain at least one value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight signed-error SCC k grid."
    )
    parser.add_argument("--run-id", help="Experiment run ID. Defaults to the newest experiment.")
    parser.add_argument(
        "--k-values",
        default=default_k_values(),
        help="Comma-separated SCC k values. Defaults to 0,0.5,...,10.",
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
    return parser.parse_args()


def build_operating_context_cache(project_root: Path, turbine_ids: list[str]) -> dict[str, dict[str, object]]:
    flags_dir = project_root / "data" / "interim" / "kelmarsh" / "flags"
    flag_cols = ["in_manual_event", "in_event", "in_maintenance", "in_communication", "in_curtailment"]
    cache = {}
    for turbine_id in turbine_ids:
        flag_path = flags_dir / f"{turbine_id.lower()}_with_flags.parquet"
        cache[turbine_id] = {
            "flag_intervals": intervals_from_flag_file(flag_path, flag_cols),
            "non_full_performance": non_full_performance_intervals_from_status(
                project_root,
                turbine_id=turbine_id,
            ),
            "information_environmental": informational_environmental_spec_intervals_from_status(
                project_root,
                turbine_id=turbine_id,
            ),
        }
    return cache


def annotate_alarm_episode_operating_context_cached(
    alarm_episodes: pd.DataFrame,
    operating_context_cache: dict[str, dict[str, object]],
) -> pd.DataFrame:
    out = alarm_episodes.copy()
    context_flags = {
        "in_manual_event_interval": "in_manual_event",
        "in_status_event_interval": "in_event",
        "in_maintenance_interval": "in_maintenance",
        "in_communication_interval": "in_communication",
        "in_curtailment_interval": "in_curtailment",
    }
    for output_col in context_flags:
        out[output_col] = False
    out["in_non_full_performance_interval"] = False
    out["in_information_environmental_buffer_interval"] = False
    out["in_non_operational_interval"] = False
    out["is_operational_false_alarm"] = False
    if out.empty:
        return out

    out["start_time"] = pd.to_datetime(out["start_time"], errors="coerce")
    out["end_time"] = pd.to_datetime(out["end_time"], errors="coerce")
    for turbine_id, idx in out.groupby("turbine_id", sort=False).groups.items():
        cache = operating_context_cache.get(str(turbine_id), {})
        intervals_by_flag = cache.get("flag_intervals", {})
        non_full_performance_intervals = cache.get("non_full_performance", [])
        information_environmental_intervals = cache.get("information_environmental", [])
        for row_index in idx:
            start = out.loc[row_index, "start_time"]
            end = out.loc[row_index, "end_time"] + pd.Timedelta(minutes=10)
            if pd.isna(start) or pd.isna(end):
                continue
            for output_col, flag_col in context_flags.items():
                out.loc[row_index, output_col] = overlaps_any_interval(
                    start,
                    end,
                    intervals_by_flag.get(flag_col, []),
                )
            out.loc[row_index, "in_non_full_performance_interval"] = overlaps_any_interval(
                start,
                end,
                non_full_performance_intervals,
            )
            out.loc[row_index, "in_information_environmental_buffer_interval"] = overlaps_any_interval(
                start,
                end,
                information_environmental_intervals,
            )

    non_operational_cols = list(context_flags.keys()) + [
        "in_non_full_performance_interval",
        "in_information_environmental_buffer_interval",
    ]
    out["in_non_operational_interval"] = out[non_operational_cols].any(axis=1)
    out["is_operational_false_alarm"] = (
        ~out["in_fault_horizon"].fillna(False).astype(bool)
        & ~out["in_non_operational_interval"].fillna(False).astype(bool)
    )
    return out


def main() -> None:
    args = parse_args()
    k_values = parse_range(args.k_values)
    if args.horizon_days < 1:
        raise ValueError("--horizon-days must be at least 1")
    if min(k_values) < 0:
        raise ValueError("k values must be non-negative")

    project_root = PROJECT_ROOT
    result_dir, _, metadata = find_experiment(project_root, args.run_id)
    grid_label = (
        f"signed_meanstd_abslimit_c{SCC_MIN_CONSECUTIVE}_"
        f"k{format_number(min(k_values))}-{format_number(max(k_values))}_step0p5"
    )
    output_path = result_dir / f"scc_grid_{grid_label}.csv"
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
    turbine_ids = sorted(test_predictions["turbine_id"].dropna().astype(str).unique())
    operating_context_cache = build_operating_context_cache(project_root, turbine_ids)

    rows = []
    total = len(k_values)
    for count, k in enumerate(k_values, start=1):
        print(f"[{count}/{total}] SCC signed mean/std k={k:g}, c={SCC_MIN_CONSECUTIVE}")
        thresholds = fit_scc_thresholds(
            val_predictions,
            target_cols=target_cols,
            k=k,
            residual_mode="signed_error",
        )
        detections = apply_scc_detection(
            test_predictions,
            thresholds=thresholds,
            target_cols=target_cols,
            min_consecutive=SCC_MIN_CONSECUTIVE,
        )
        event_summary = event_summary_base.copy()
        detections = annotate_alarm_horizon_matches(detections, event_summary)
        alarm_episodes = build_alarm_episodes(detections)
        alarm_episodes = annotate_alarm_episode_horizon_matches(alarm_episodes, event_summary)
        alarm_episodes = annotate_alarm_episode_operating_context_cached(
            alarm_episodes,
            operating_context_cache,
        )
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
                "method": "scc_signed_meanstd",
                "k": k,
                "min_consecutive": SCC_MIN_CONSECUTIVE,
                "horizon_days": args.horizon_days,
                "threshold_mean": float(thresholds["threshold"].mean()),
                "threshold_min": float(thresholds["threshold"].min()),
                "threshold_max": float(thresholds["threshold"].max()),
                "center_mean": float(thresholds["center"].mean()),
                "scale_mean": float(thresholds["scale"].mean()),
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

    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    metadata.setdefault("scc_grid", {})[grid_label] = {
        "residual_mode": "signed_error",
        "threshold_rule": "abs(mean(validation_error) + k * std(validation_error))",
        "center_type": "mean",
        "scale_type": "std",
        "min_consecutive": SCC_MIN_CONSECUTIVE,
        "horizon_days": args.horizon_days,
        "k_values": k_values,
        "path": str(output_path.relative_to(project_root)),
    }
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Saved SCC grid summary:", output_path)


if __name__ == "__main__":
    main()
