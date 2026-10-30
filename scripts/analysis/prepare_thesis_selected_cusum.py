"""Save raw fused episodes for the best revised CUSUM grid settings."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from main_detect_cusum import format_number
from main_detect_residual_baseline import (
    annotate_alarm_horizon_matches, build_target_event_summary, build_turbine_alarm_episodes, find_experiment,
    load_prediction_file,
)
from scripts.analysis.prepare_thesis_seven_settings_split6h import evaluate
from scripts.experiments.main_grid_cusum_fixed_meanstd import (
    build_cusum_alarm_rows, fit_mean_std_reference, prepare_test_groups,
)
from src.config.kelmarsh_config import TARGET_COLS


RUN = "all6_multi3_seq12_h64_l1_bs64_lr1e-03_wd0_do0_e60_p8_seed42"
GRID_NAMES = {
    "two-sided": "cusum_grid_fixed_meanstd_no_consecutive_k0-1_h20-200_split6h_no_warning_turbine.csv",
    "absolute": "cusum_grid_absolute_fixed_meanstd_no_consecutive_k0-1_h20-200_split6h_no_warning_turbine.csv",
}


def main() -> None:
    result_dir, _, metadata = find_experiment(ROOT, RUN)
    selected = {}
    for mode, filename in GRID_NAMES.items():
        grid = pd.read_csv(result_dir / filename)
        if len(grid) != 110:
            raise ValueError(f"Incomplete CUSUM grid: {filename} ({len(grid)}/110)")
        selected[mode] = grid.loc[grid["f1_score"].idxmax()]

    validation = load_prediction_file(ROOT, metadata, "predictions", "healthy_val_predictions.csv")
    test = load_prediction_file(ROOT, metadata, "test_predictions", "test_2023_2024_predictions.csv")
    targets = metadata.get("targets", TARGET_COLS)
    reference = fit_mean_std_reference(validation, targets)
    groups = prepare_test_groups(test, targets, reference)
    del validation, test
    events = build_target_event_summary(ROOT, horizon_days=7, include_long_warnings=False)
    cache = {"target_events": events}
    rows = []
    for mode, grid_row in selected.items():
        k, h = float(grid_row["k"]), float(grid_row["h"])
        detections = build_cusum_alarm_rows(groups, k, h, mode)
        detections = annotate_alarm_horizon_matches(detections, events)
        episodes = build_turbine_alarm_episodes(detections)
        setting = (f"twosided_std_k{format_number(k)}_h{format_number(h)}_reset"
                   if mode == "two-sided" else
                   f"absolute_fixed_meanstd_k{format_number(k)}_h{format_number(h)}_reset")
        setting += "_split6h_no_warning_turbine"
        path = result_dir / f"cusum_alarm_episodes_{setting}.csv"
        episodes.to_csv(path, index=False, encoding="utf-8-sig")
        pd.DataFrame([
            {"target": target, "center": stats["center"], "scale": stats["scale"],
             "reference_value": k, "decision_threshold": h}
            for target, stats in reference.items()
        ]).to_csv(result_dir / f"cusum_thresholds_{setting}.csv", index=False, encoding="utf-8-sig")
        _, metrics = evaluate(episodes, events, 1, evaluation_cache=cache)
        for key, field in (("detected_events", "detected_events"),
                           ("true_alarm_episodes", "true_alarm_episodes"),
                           ("operational_false_alarm_episodes", "operational_false_alarm_episodes")):
            if int(metrics[key]) != int(grid_row[field]):
                raise ValueError(f"{setting}: {key} does not match grid")
        if not np.isclose(metrics["f1"], float(grid_row["f1_score"])):
            raise ValueError(f"{setting}: F1 does not match grid")
        rows.append({"mode": mode, "k": k, "h": h, "setting": setting, **metrics})
        print(f"{mode}: k={k:g}, h={h:g}, F1={metrics['f1']:.4f}, episodes={path}", flush=True)
    output = ROOT / "results" / "kelmarsh" / "thesis_revision_split6h" / "selected_cusum.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
