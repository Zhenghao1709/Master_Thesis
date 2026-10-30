"""Evaluate selected folded-normal BCAD W/c settings without Warning targets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from main_detect_residual_baseline import build_alarm_episodes, build_target_event_summary, find_experiment, load_prediction_file
from scripts.experiments.recompute_bcad_no_warning import RUN, evaluate
from src.config.kelmarsh_config import TARGET_COLS
from src.detection.bcad import apply_bcad_detection


SETTINGS = [(360, 144), (360, 216), (180, 144), (180, 72), (180, 216), (72, 72)]


def main() -> None:
    result_dir, _, metadata = find_experiment(PROJECT_ROOT, RUN)
    output = result_dir / "no_warning_bcad"
    output.mkdir(exist_ok=True)
    test = load_prediction_file(
        PROJECT_ROOT, metadata, path_key="test_predictions",
        fallback_name="test_2023_2024_predictions.csv",
    )
    events = build_target_event_summary(PROJECT_ROOT, horizon_days=7, include_long_warnings=False)
    if len(events) != 459:
        raise ValueError(f"Expected 459 no-Warning target events, found {len(events)}")
    for window_size, min_consecutive in SETTINGS:
        setting = f"foldedmle_samecenterabs_w{window_size}_q990_vm5_c{min_consecutive}"
        performance_path = output / f"bcad_performance_{setting}.json"
        if performance_path.is_file():
            print(f"Already exists, skipped: {setting}", flush=True)
            continue
        source = f"foldedmle_samecenterabs_w{window_size}_q990_vm5_c6"
        thresholds = pd.read_csv(result_dir / f"bcad_thresholds_{source}.csv")
        detections = apply_bcad_detection(
            test, thresholds, metadata.get("targets", TARGET_COLS),
            min_consecutive=min_consecutive,
        )
        detections["in_fault_horizon"] = False
        episodes = build_alarm_episodes(detections)
        episodes, scored_events, result = evaluate(episodes, events.copy(), already_classified=False)
        thresholds.to_csv(output / f"bcad_thresholds_{setting}.csv", index=False)
        episodes.to_csv(output / f"bcad_alarm_episodes_{setting}.csv", index=False)
        scored_events.to_csv(output / f"bcad_event_summary_{setting}.csv", index=False)
        performance_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{setting}: events={result['event_level']['detected_events']}/459, "
              f"episodes={result['alarm_episode_level']['total_alarm_episodes']}, "
              f"F1={result['f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
