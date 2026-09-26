from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from main_detect_residual_baseline import find_experiment, load_prediction_file
from src.config.kelmarsh_config import TARGET_COLS
from src.detection.bcad import fit_folded_normal_mle
from src.detection.residuals import resolve_prediction_column


PARAMETER_NAME = "bcad_folded_normal_mle_healthy_parameters.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit healthy folded-normal BCAD models by MLE.")
    parser.add_argument("--run-id", help="Experiment run ID. Defaults to the newest experiment.")
    args = parser.parse_args()

    result_dir, _, metadata = find_experiment(PROJECT_ROOT, args.run_id)
    validation = load_prediction_file(
        PROJECT_ROOT, metadata, path_key="predictions", fallback_name="healthy_val_predictions.csv"
    )
    rows = []
    for target in metadata.get("targets", TARGET_COLS):
        error_col = resolve_prediction_column(validation, "error", target)
        values = pd.to_numeric(validation[error_col], errors="coerce").abs().dropna()
        center, scale = fit_folded_normal_mle(values)
        rows.append({
            "target": target,
            "healthy_center": center,
            "healthy_scale": scale,
            "foldnorm_shape": center / scale,
            "sample_count": len(values),
            "fit_method": "scipy.stats.foldnorm.fit with floc=0",
            "source": "healthy validation point absolute residual",
        })

    parameters = pd.DataFrame(rows)
    path = result_dir / PARAMETER_NAME
    parameters.to_csv(path, index=False, encoding="utf-8-sig")
    print(parameters.to_string(index=False))
    print("Healthy folded-normal MLE parameters saved to:", path)


if __name__ == "__main__":
    main()
