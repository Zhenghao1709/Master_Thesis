"""Save detailed episode and event results for an absolute-error SCC setting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.kelmarsh_config import TARGET_COLS
from src.detection.scc import apply_scc_detection, fit_scc_thresholds
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--k", type=float, default=6.0)
    args = parser.parse_args()

    result_dir, _, metadata = find_experiment(PROJECT_ROOT, args.run_id)
    val = load_prediction_file(
        PROJECT_ROOT, metadata, path_key="predictions",
        fallback_name="healthy_val_predictions.csv",
    )
    test = load_prediction_file(
        PROJECT_ROOT, metadata, path_key="test_predictions",
        fallback_name="test_2023_2024_predictions.csv",
    )
    targets = metadata.get("targets", TARGET_COLS)
    thresholds = fit_scc_thresholds(val, targets, args.k, residual_mode="absolute")
    detections = apply_scc_detection(test, thresholds, targets, min_consecutive=6)
    events = build_target_event_summary(PROJECT_ROOT, horizon_days=7)
    detections = annotate_alarm_horizon_matches(detections, events)
    episodes = build_alarm_episodes(detections)
    episodes = annotate_alarm_episode_horizon_matches(episodes, events)
    episodes = annotate_alarm_episode_operating_context(episodes, PROJECT_ROOT)
    events = update_event_summary_from_alarm_episodes(events, episodes)
    performance = summarize_detection_performance(detections, events, episodes)

    suffix = f"abs_meanstd_k{args.k:g}_c6".replace(".", "p")
    paths = {
        "thresholds": result_dir / f"scc_thresholds_{suffix}.csv",
        "events": result_dir / f"scc_event_summary_{suffix}.csv",
        "episodes": result_dir / f"scc_alarm_episodes_{suffix}.csv",
        "performance": result_dir / f"scc_performance_{suffix}.json",
    }
    # Existing runs belong to earlier evaluations; do not overwrite them.
    if any(path.exists() for path in paths.values()):
        suffix += "_current"
        paths = {
            "thresholds": result_dir / f"scc_thresholds_{suffix}.csv",
            "events": result_dir / f"scc_event_summary_{suffix}.csv",
            "episodes": result_dir / f"scc_alarm_episodes_{suffix}.csv",
            "performance": result_dir / f"scc_performance_{suffix}.json",
        }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Detailed SCC output already exists for {suffix}")

    thresholds.to_csv(paths["thresholds"], index=False, encoding="utf-8-sig")
    events.to_csv(paths["events"], index=False, encoding="utf-8-sig")
    episodes.to_csv(paths["episodes"], index=False, encoding="utf-8-sig")
    paths["performance"].write_text(
        json.dumps(performance, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "paths": {key: str(path) for key, path in paths.items()},
        "event_level": performance["event_level"],
        "alarm_episode_level": performance["alarm_episode_level"],
    }, indent=2))


if __name__ == "__main__":
    main()
