from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from main_detect_residual_baseline import find_experiment, load_prediction_file
from main_predict_bcad_calibration import CALIBRATION_END_EXCLUSIVE, CALIBRATION_START
from scripts.experiments.main_fit_bcad_weibull_mle import ensure_calibration_predictions
from src.config.kelmarsh_config import TARGET_COLS
from src.detection.bcad import fit_weibull_mle
from src.detection.residuals import resolve_prediction_column


PARAMETER_NAME = "bcad_weibull_point_mle_parameters_calib2021_2022.csv"
OLD_EPISODE_NAME = "bcad_calibration_old_gaussian_ta_episodes_calib2021_2022_w72.csv"


def select_calibration_alarm_points(calibration: pd.DataFrame, episodes: pd.DataFrame) -> pd.Series:
    episodes = episodes.loc[episodes["in_fault_horizon"].fillna(False).astype(bool)].copy()
    episodes["start_time"] = pd.to_datetime(episodes["start_time"], errors="raise")
    episodes["end_time"] = pd.to_datetime(episodes["end_time"], errors="raise")
    timestamps = pd.to_datetime(calibration["Date and time"], errors="raise")
    selected = pd.Series(False, index=calibration.index)
    for turbine_id, turbine_episodes in episodes.groupby("turbine_id"):
        indices = calibration.index[calibration["turbine_id"].eq(turbine_id)]
        turbine_times = timestamps.loc[indices]
        turbine_selected = np.zeros(len(indices), dtype=bool)
        for episode in turbine_episodes.itertuples(index=False):
            turbine_selected |= turbine_times.between(episode.start_time, episode.end_time).to_numpy()
        selected.loc[indices] = turbine_selected
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit per-point Weibull H and A models for joint BCAD.")
    parser.add_argument("--run-id", help="Experiment run ID. Defaults to the newest experiment.")
    args = parser.parse_args()

    result_dir, _, metadata = find_experiment(PROJECT_ROOT, args.run_id)
    parameter_path = result_dir / PARAMETER_NAME
    validation = load_prediction_file(
        PROJECT_ROOT, metadata, path_key="predictions", fallback_name="healthy_val_predictions.csv"
    )
    calibration_path = ensure_calibration_predictions(PROJECT_ROOT, metadata["run_id"], result_dir)
    calibration = pd.read_csv(calibration_path, encoding="utf-8-sig")
    episode_path = result_dir / OLD_EPISODE_NAME
    if not episode_path.exists():
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "experiments" / "main_fit_bcad_weibull_mle.py"),
             "--run-id", metadata["run_id"], "--window-size", "72"],
            cwd=PROJECT_ROOT, check=True,
        )
    episodes = pd.read_csv(episode_path, encoding="utf-8-sig")
    alarm_points = select_calibration_alarm_points(calibration, episodes)
    if not alarm_points.any():
        raise ValueError("No true-alarm calibration points were selected")

    rows = []
    for target in metadata.get("targets", TARGET_COLS):
        healthy_col = resolve_prediction_column(validation, "error", target)
        abnormal_col = resolve_prediction_column(calibration, "error", target)
        healthy = pd.to_numeric(validation[healthy_col], errors="coerce").abs().dropna()
        abnormal = pd.to_numeric(calibration.loc[alarm_points, abnormal_col], errors="coerce").abs().dropna()
        healthy = healthy[healthy > 0]
        abnormal = abnormal[abnormal > 0]
        healthy_shape, healthy_scale = fit_weibull_mle(healthy)
        abnormal_shape, abnormal_scale = fit_weibull_mle(abnormal)
        rows.append({
            "target": target,
            "fit_method": "scipy.stats.weibull_min.fit with floc=0",
            "calibration_start": CALIBRATION_START,
            "calibration_end_exclusive": CALIBRATION_END_EXCLUSIVE,
            "healthy_source": "healthy validation point absolute residual",
            "abnormal_source": "old abs Gaussian TA episode point absolute residual",
            "healthy_weibull_shape": healthy_shape,
            "healthy_weibull_scale": healthy_scale,
            "weibull_shape": abnormal_shape,
            "weibull_scale": abnormal_scale,
            "healthy_samples": len(healthy),
            "abnormal_samples": len(abnormal),
        })

    parameters = pd.DataFrame(rows)
    parameters.to_csv(parameter_path, index=False, encoding="utf-8-sig")
    print(parameters.to_string(index=False))
    print("Point-level Weibull parameters saved to:", parameter_path)


if __name__ == "__main__":
    main()
