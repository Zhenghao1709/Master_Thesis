from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from src.config.kelmarsh_config import INPUT_COLS, SCALER_TYPE_BY_SIGNAL, TARGET_COL, SEQ_LEN
from src.data.split_data import split_healthy_segments_by_time
from src.detection.residuals import add_residual_columns, compute_regression_metrics
from src.features.scaling import apply_scalers, fit_scalers, save_scaler
from src.modeling.predict import (
    build_prediction_dataframe,
    load_trained_gru_model,
    predict_with_gru,
    prepare_scaled_sequences,
)
from src.modeling.train_gru import train_gru_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one GRU normal behavior model experiment.")
    parser.add_argument("--turbine", default="Kelmarsh_1", help="Turbine name, e.g. Kelmarsh_1")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum number of training epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Stop after this many epochs without validation improvement; use 0 to disable",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible experiments")
    return parser.parse_args()


def make_run_id(turbine_name: str, epochs: int) -> str:
    return f"{turbine_name.lower()}_e{epochs}"


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.patience < 0:
        raise ValueError("--patience cannot be negative")

    set_random_seed(args.seed)

    project_root = Path(__file__).resolve().parent
    turbine_name = args.turbine
    run_id = make_run_id(turbine_name, args.epochs)

    healthy_segments_path = (
        project_root
        / "data" / "processed" / "kelmarsh" / "healthy_segments"
        / f"{turbine_name.lower()}_healthy_segments.parquet"
    )
    train_val_dir = project_root / "data" / "processed" / "kelmarsh" / "train_val"
    train_path = train_val_dir / f"{turbine_name.lower()}_train_healthy.parquet"
    val_path = train_val_dir / f"{turbine_name.lower()}_val_healthy.parquet"

    result_dir = project_root / "results" / "kelmarsh" / "experiments" / run_id
    model_dir = project_root / "models" / "kelmarsh" / "experiments" / run_id
    if result_dir.exists() or model_dir.exists():
        raise FileExistsError(f"Experiment already exists: {run_id}")

    train_log_path = result_dir / "train_log.csv"
    prediction_path = result_dir / "healthy_val_predictions.csv"
    metadata_path = result_dir / "metadata.json"
    model_path = model_dir / "model.pth"
    x_scaler_path = model_dir / "x_scaler.pkl"
    y_scaler_path = model_dir / "y_scaler.pkl"

    healthy_segments = pd.read_parquet(healthy_segments_path)
    required_cols = ["Date and time", "turbine_id", "segment_id"] + INPUT_COLS + [TARGET_COL]
    required_cols = list(dict.fromkeys(required_cols))
    train_df_full = healthy_segments[required_cols].dropna(subset=INPUT_COLS + [TARGET_COL]).copy()

    train_df, val_df = split_healthy_segments_by_time(
        train_df_full,
        time_col="Date and time",
        train_ratio=0.8,
    )
    train_val_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    print("Run ID:", run_id)
    print("Train rows:", len(train_df))
    print("Validation rows:", len(val_df))

    x_scaler, y_scaler = fit_scalers(train_df, INPUT_COLS, TARGET_COL)
    train_scaled = apply_scalers(train_df, INPUT_COLS, TARGET_COL, x_scaler, y_scaler)
    val_scaled = apply_scalers(val_df, INPUT_COLS, TARGET_COL, x_scaler, y_scaler)
    save_scaler(x_scaler, x_scaler_path)
    save_scaler(y_scaler, y_scaler_path)

    batch_size = 64
    hidden_size = 64
    num_layers = 1
    learning_rate = 1e-3
    _, log_df = train_gru_model(
        train_df=train_scaled,
        val_df=val_scaled,
        target_col=TARGET_COL,
        input_cols=INPUT_COLS,
        seq_len=SEQ_LEN,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        lr=learning_rate,
        epochs=args.epochs,
        early_stopping_patience=args.patience or None,
        model_save_path=model_path,
        log_save_path=train_log_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_val, y_val_scaled, meta_df = prepare_scaled_sequences(
        df=val_df,
        input_cols=INPUT_COLS,
        target_col=TARGET_COL,
        seq_len=SEQ_LEN,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        time_col="Date and time",
    )
    best_model = load_trained_gru_model(
        model_path=model_path,
        input_size=len(INPUT_COLS),
        hidden_size=hidden_size,
        num_layers=num_layers,
        device=device,
    )
    y_pred_scaled = predict_with_gru(best_model, X_val, batch_size=512, device=device)
    pred_df = build_prediction_dataframe(meta_df, y_val_scaled, y_pred_scaled, y_scaler)
    pred_df = add_residual_columns(pred_df)
    result_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    metrics = compute_regression_metrics(pred_df)
    best_row = log_df.loc[log_df["val_loss"].idxmin()]
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "turbine": turbine_name,
        "target": TARGET_COL,
        "epochs_requested": args.epochs,
        "epochs_completed": int(log_df["epoch"].max()),
        "early_stopping_patience": args.patience or None,
        "early_stopped": len(log_df) < args.epochs,
        "random_seed": args.seed,
        "best_epoch": int(best_row["epoch"]),
        "best_val_loss": float(best_row["val_loss"]),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "prediction_samples": len(pred_df),
        "sequence_length": SEQ_LEN,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "input_cols": INPUT_COLS,
        "scaler_type_by_signal": {
            signal: SCALER_TYPE_BY_SIGNAL[signal]
            for signal in list(dict.fromkeys(INPUT_COLS + [TARGET_COL]))
        },
        "prediction_units": "physical",
        "scaling_pipeline_version": 2,
        "metrics": metrics,
        "paths": {
            "train_log": str(train_log_path.relative_to(project_root)),
            "predictions": str(prediction_path.relative_to(project_root)),
            "model": str(model_path.relative_to(project_root)),
            "x_scaler": str(x_scaler_path.relative_to(project_root)),
            "y_scaler": str(y_scaler_path.relative_to(project_root)),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nValidation metrics (physical units):")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print("Best epoch:", int(best_row["epoch"]))
    print("Experiment saved to:", result_dir)


if __name__ == "__main__":
    main()
