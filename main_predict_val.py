from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.config.kelmarsh_config import INPUT_COLS, TARGET_COL, SEQ_LEN
from src.detection.residuals import add_residual_columns, compute_regression_metrics
from src.features.scaling import load_scaler
from src.modeling.predict import (
    build_prediction_dataframe,
    load_trained_gru_model,
    predict_with_gru,
    prepare_scaled_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate validation predictions for a saved experiment.")
    parser.add_argument("--run-id", help="Experiment run ID. Defaults to the newest experiment.")
    return parser.parse_args()


def find_experiment(project_root: Path, requested_run_id: str | None) -> tuple[Path, dict]:
    experiments_dir = project_root / "results" / "kelmarsh" / "experiments"
    if requested_run_id:
        metadata_path = experiments_dir / requested_run_id / "metadata.json"
    else:
        candidates = sorted(
            experiments_dir.glob("*/metadata.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError("No experiment metadata found. Run main_train_gru.py first.")
        metadata_path = candidates[0]
    if not metadata_path.exists():
        raise FileNotFoundError(f"Experiment metadata not found: {metadata_path}")
    return metadata_path.parent, json.loads(metadata_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    result_dir, metadata = find_experiment(project_root, args.run_id)
    if metadata.get("scaling_pipeline_version") != 2:
        raise ValueError(
            "This experiment used the previous scaling pipeline and cannot be regenerated "
            "with the corrected prediction code. Keep its existing prediction file or retrain it."
        )
    turbine_name = metadata["turbine"]
    paths = {key: project_root / value for key, value in metadata["paths"].items()}

    val_path = (
        project_root / "data" / "processed" / "kelmarsh" / "train_val"
        / f"{turbine_name.lower()}_val_healthy.parquet"
    )
    val_df = pd.read_parquet(val_path).copy()
    val_df["Date and time"] = pd.to_datetime(val_df["Date and time"])
    val_df = val_df.sort_values("Date and time").dropna(subset=INPUT_COLS + [TARGET_COL]).reset_index(drop=True)

    x_scaler = load_scaler(paths["x_scaler"])
    y_scaler = load_scaler(paths["y_scaler"])
    X_val, y_val_scaled, meta_df = prepare_scaled_sequences(
        val_df, INPUT_COLS, TARGET_COL, SEQ_LEN, x_scaler, y_scaler, "Date and time"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_gru_model(
        paths["model"],
        input_size=len(INPUT_COLS),
        hidden_size=int(metadata["hidden_size"]),
        num_layers=int(metadata["num_layers"]),
        device=device,
    )
    y_pred_scaled = predict_with_gru(model, X_val, batch_size=512, device=device)
    pred_df = build_prediction_dataframe(meta_df, y_val_scaled, y_pred_scaled, y_scaler)
    pred_df = add_residual_columns(pred_df)
    pred_df.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")

    metrics = compute_regression_metrics(pred_df)
    metadata["metrics"] = metrics
    metadata["prediction_samples"] = len(pred_df)
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Run ID:", metadata["run_id"])
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print("Prediction file saved to:", paths["predictions"])


if __name__ == "__main__":
    main()
