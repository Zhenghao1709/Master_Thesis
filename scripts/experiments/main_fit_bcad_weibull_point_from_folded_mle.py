from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from main_detect_residual_baseline import find_experiment, load_prediction_file
from scripts.experiments.main_fit_bcad_weibull_from_folded_mle import SETTING
from scripts.experiments.main_fit_bcad_weibull_mle import (
    CALIBRATION_END_EXCLUSIVE,
    CALIBRATION_START,
    ensure_calibration_predictions,
)
from scripts.experiments.main_fit_bcad_weibull_point_mle import select_calibration_alarm_points
from src.config.kelmarsh_config import TARGET_COLS
from src.detection.bcad import fit_weibull_mle
from src.detection.residuals import resolve_prediction_column


PARAMETER_NAME = "bcad_weibull_point_mle_parameters_from_foldedmle_calib2021_2022.csv"
EPISODE_NAME = f"bcad_calibration_ta_episodes_foldedmle_{SETTING}_calib2021_2022.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit point-level Weibull H/A from healthy validation and folded-MLE calibration alarms."
    )
    parser.add_argument("--run-id", help="Experiment run ID. Defaults to the newest experiment.")
    args = parser.parse_args()

    result_dir, _, metadata = find_experiment(PROJECT_ROOT, args.run_id)
    validation = load_prediction_file(
        PROJECT_ROOT, metadata, path_key="predictions", fallback_name="healthy_val_predictions.csv"
    )
    calibration_path = ensure_calibration_predictions(PROJECT_ROOT, metadata["run_id"], result_dir)
    calibration = pd.read_csv(calibration_path, encoding="utf-8-sig")

    episode_path = result_dir / EPISODE_NAME
    if not episode_path.exists():
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "experiments" /
             "main_fit_bcad_weibull_from_folded_mle.py"), "--run-id", metadata["run_id"]],
            cwd=PROJECT_ROOT, check=True,
        )
    episodes = pd.read_csv(episode_path, encoding="utf-8-sig")
    alarm_points = select_calibration_alarm_points(calibration, episodes)
    if not alarm_points.any():
        raise ValueError("No folded-MLE true-alarm calibration points were selected")

    rows = []
    for target in metadata.get("targets", TARGET_COLS):
        healthy_col = resolve_prediction_column(validation, "error", target)
        abnormal_col = resolve_prediction_column(calibration, "error", target)
        healthy = pd.to_numeric(validation[healthy_col], errors="coerce").abs().dropna()
        abnormal = pd.to_numeric(
            calibration.loc[alarm_points, abnormal_col], errors="coerce"
        ).abs().dropna()
        healthy_shape, healthy_scale = fit_weibull_mle(healthy)
        abnormal_shape, abnormal_scale = fit_weibull_mle(abnormal)
        rows.append({
            "target": target,
            "fit_method": "scipy.stats.weibull_min.fit with floc=0",
            "calibration_detector": "folded_normal_mle_shared_center",
            "calibration_setting": SETTING,
            "calibration_start": CALIBRATION_START,
            "calibration_end_exclusive": CALIBRATION_END_EXCLUSIVE,
            "healthy_source": "healthy validation point absolute residual",
            "abnormal_source": "folded-normal-MLE TA episode point absolute residual",
            "healthy_weibull_shape": healthy_shape,
            "healthy_weibull_scale": healthy_scale,
            "weibull_shape": abnormal_shape,
            "weibull_scale": abnormal_scale,
            "healthy_samples": int((healthy > 0).sum()),
            "abnormal_samples": int((abnormal > 0).sum()),
        })

    path = result_dir / PARAMETER_NAME
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print("Folded-MLE calibration true-alarm episodes:", int(episodes["in_fault_horizon"].sum()))
    print(pd.DataFrame(rows).to_string(index=False))
    print("Point-level Weibull parameters saved to:", path)


if __name__ == "__main__":
    main()
