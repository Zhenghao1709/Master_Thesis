from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from src.config.kelmarsh_config import INPUT_COLS, SCALER_TYPE_BY_SIGNAL, TARGET_COLS, SEQ_LEN
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

DEFAULT_TURBINES = [f"Kelmarsh_{i}" for i in range(1, 7)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GRU normal behavior model experiment.")
    parser.add_argument(
        "--turbine",
        "--turbines",
        dest="turbines",
        default="ALL",
        help="Turbine name, comma-separated names, or ALL. Defaults to ALL.",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Maximum number of training epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Stop after this many epochs without validation improvement; use 0 to disable",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible experiments")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--hidden-size", type=int, default=64, help="GRU hidden state size")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of stacked GRU layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="GRU dropout; only active when num_layers > 1")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--amsgrad", action="store_true", help="Use the AMSGrad variant of Adam")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="Input sequence length")
    return parser.parse_args()


def parse_turbine_selection(selection: str) -> list[str]:
    if selection.strip().upper() == "ALL":
        return DEFAULT_TURBINES
    turbines = [item.strip() for item in selection.split(",") if item.strip()]
    if not turbines:
        raise ValueError("--turbine must be ALL, one turbine name, or comma-separated turbine names.")
    return turbines


def format_hparam(value: float | int | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if value == 0:
        return "0"
    return f"{value:.0e}".replace("+", "")


def make_dataset_id(turbines: list[str]) -> str:
    if turbines == DEFAULT_TURBINES:
        return "all6"
    if len(turbines) == 1:
        return turbines[0].lower()
    return "custom" + "-".join(t.split("_")[-1] for t in turbines)


def make_run_id(args: argparse.Namespace, dataset_id: str) -> str:
    patience = args.patience if args.patience else "off"
    return (
        f"{dataset_id}_multi{len(TARGET_COLS)}"
        f"_seq{args.seq_len}"
        f"_h{args.hidden_size}"
        f"_l{args.num_layers}"
        f"_bs{args.batch_size}"
        f"_lr{format_hparam(args.learning_rate)}"
        f"_wd{format_hparam(args.weight_decay)}"
        f"_do{format_hparam(args.dropout)}"
        f"_e{args.epochs}"
        f"_p{patience}"
        f"_seed{args.seed}"
    )


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def interpolate_missing_values_within_segments(
    df: pd.DataFrame,
    signal_cols: list[str],
    time_col: str = "Date and time",
) -> pd.DataFrame:
    """Linearly fill missing signal values inside each healthy segment only."""
    group_cols = [col for col in ["turbine_id", "segment_id"] if col in df.columns]
    if not group_cols:
        raise ValueError("Expected turbine_id and segment_id columns before interpolation.")

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.sort_values(group_cols + [time_col]).reset_index(drop=True)
    signal_cols = [col for col in signal_cols if col in out.columns]

    for _, group_idx in out.groupby(group_cols, sort=False).groups.items():
        idx = list(group_idx)
        values = out.loc[idx, signal_cols].apply(pd.to_numeric, errors="coerce")
        interpolated = values.interpolate(
            method="linear",
            limit_direction="both",
            limit_area="inside",
        )
        out.loc[idx, signal_cols] = interpolated

    return out


def load_healthy_segments(
    project_root: Path,
    turbines: list[str],
    required_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    healthy_segments_dir = project_root / "data" / "processed" / "kelmarsh" / "healthy_segments"
    parts = []
    row_counts = {}
    paths = []

    for turbine_name in turbines:
        path = healthy_segments_dir / f"{turbine_name.lower()}_healthy_segments.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing healthy segments for {turbine_name}: {path}")
        df = pd.read_parquet(path)
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in {path.name}: {missing_cols}")
        df = df[required_cols].copy()
        df["turbine_id"] = turbine_name
        parts.append(df)
        row_counts[turbine_name] = len(df)
        paths.append(str(path.relative_to(project_root)))

    return pd.concat(parts, ignore_index=True), row_counts, paths


def load_and_validate_test_sets(
    project_root: Path,
    turbines: list[str],
    dataset_id: str,
) -> tuple[int, list[str], str | None]:
    train_val_dir = project_root / "data" / "processed" / "kelmarsh" / "train_val"
    combined_path = train_val_dir / f"{dataset_id}_test_2023_2024.parquet"
    parts = []
    paths = []

    if combined_path.exists():
        combined = pd.read_parquet(combined_path, columns=["Date and time"])
        combined["Date and time"] = pd.to_datetime(combined["Date and time"], errors="coerce")
        invalid_test_years = combined.loc[
            combined["Date and time"].notna()
            & ~combined["Date and time"].dt.year.isin([2023, 2024])
        ]
        if not invalid_test_years.empty:
            raise ValueError(f"Combined test set contains rows outside 2023-2024: {combined_path}")
        print(f"Existing combined test set found, skipped rebuild: {combined_path}")
        for turbine_name in turbines:
            path = train_val_dir / f"{turbine_name.lower()}_test_2023_2024.parquet"
            if path.exists():
                paths.append(str(path.relative_to(project_root)))
        return len(combined), paths, str(combined_path.relative_to(project_root))

    for turbine_name in turbines:
        path = train_val_dir / f"{turbine_name.lower()}_test_2023_2024.parquet"
        if not path.exists():
            print(f"[WARN] Test set not found yet for {turbine_name}: {path}")
            continue
        test_df = pd.read_parquet(path)
        test_df["Date and time"] = pd.to_datetime(test_df["Date and time"], errors="coerce")
        invalid_test_years = test_df.loc[
            test_df["Date and time"].notna()
            & ~test_df["Date and time"].dt.year.isin([2023, 2024])
        ]
        if not invalid_test_years.empty:
            raise ValueError(f"Test set contains rows outside 2023-2024: {path}")
        parts.append(test_df)
        paths.append(str(path.relative_to(project_root)))

    if not parts:
        return 0, paths, None

    combined = pd.concat(parts, ignore_index=True)
    combined.to_parquet(combined_path, index=False)
    return len(combined), paths, str(combined_path.relative_to(project_root))


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.patience < 0:
        raise ValueError("--patience cannot be negative")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.hidden_size < 1:
        raise ValueError("--hidden-size must be at least 1")
    if args.num_layers < 1:
        raise ValueError("--num-layers must be at least 1")
    if args.dropout < 0:
        raise ValueError("--dropout cannot be negative")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative")
    if args.seq_len < 1:
        raise ValueError("--seq-len must be at least 1")

    set_random_seed(args.seed)

    project_root = Path(__file__).resolve().parent
    turbines = parse_turbine_selection(args.turbines)
    dataset_id = make_dataset_id(turbines)
    run_id = make_run_id(args, dataset_id)

    train_val_dir = project_root / "data" / "processed" / "kelmarsh" / "train_val"
    train_path = train_val_dir / f"{dataset_id}_train_healthy.parquet"
    val_path = train_val_dir / f"{dataset_id}_val_healthy.parquet"

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

    required_cols = ["Date and time", "turbine_id", "segment_id"] + INPUT_COLS + TARGET_COLS
    required_cols = list(dict.fromkeys(required_cols))
    healthy_segments, healthy_segment_rows_by_turbine, healthy_segment_paths = load_healthy_segments(
        project_root,
        turbines,
        required_cols,
    )
    train_df_full = healthy_segments.copy()
    signal_cols = list(dict.fromkeys(INPUT_COLS + TARGET_COLS))
    missing_before = int(train_df_full[signal_cols].isna().sum().sum())
    train_df_full = interpolate_missing_values_within_segments(
        train_df_full,
        signal_cols=signal_cols,
        time_col="Date and time",
    )
    missing_after_interpolation = int(train_df_full[signal_cols].isna().sum().sum())
    train_df_full = train_df_full.dropna(subset=signal_cols).copy()
    missing_after_dropna = int(train_df_full[signal_cols].isna().sum().sum())

    train_df, val_df = split_healthy_segments_by_time(
        train_df_full,
        time_col="Date and time",
        train_ratio=0.8,
    )
    for split_name, split_df in [("train", train_df), ("validation", val_df)]:
        split_time = pd.to_datetime(split_df["Date and time"], errors="coerce")
        invalid_years = split_df.loc[
            split_time.notna() & ~split_time.dt.year.between(2016, 2022)
        ]
        if not invalid_years.empty:
            raise ValueError(f"{split_name} split contains rows outside 2016-2022.")

    train_val_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_rows, test_paths, combined_test_path = load_and_validate_test_sets(
        project_root,
        turbines,
        dataset_id,
    )
    print("Run ID:", run_id)
    print("Turbines:", ", ".join(turbines))
    print("Train rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Test rows (2023-2024, not used for training):", test_rows)
    print("Missing signal values before interpolation:", missing_before)
    print("Missing signal values after interpolation:", missing_after_interpolation)
    print("Missing signal values after final dropna:", missing_after_dropna)

    x_scaler, y_scaler = fit_scalers(train_df, INPUT_COLS, TARGET_COLS)
    train_scaled = apply_scalers(train_df, INPUT_COLS, TARGET_COLS, x_scaler, y_scaler)
    val_scaled = apply_scalers(val_df, INPUT_COLS, TARGET_COLS, x_scaler, y_scaler)
    save_scaler(x_scaler, x_scaler_path)
    save_scaler(y_scaler, y_scaler_path)

    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    dropout = args.dropout
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    _, log_df = train_gru_model(
        train_df=train_scaled,
        val_df=val_scaled,
        target_cols=TARGET_COLS,
        input_cols=INPUT_COLS,
        seq_len=args.seq_len,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=learning_rate,
        weight_decay=weight_decay,
        amsgrad=args.amsgrad,
        epochs=args.epochs,
        early_stopping_patience=args.patience or None,
        model_save_path=model_path,
        log_save_path=train_log_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_val, y_val_scaled, meta_df = prepare_scaled_sequences(
        df=val_df,
        input_cols=INPUT_COLS,
        target_cols=TARGET_COLS,
        seq_len=args.seq_len,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        time_col="Date and time",
    )
    best_model = load_trained_gru_model(
        model_path=model_path,
        input_size=len(INPUT_COLS),
        output_size=len(TARGET_COLS),
        hidden_size=hidden_size,
        num_layers=num_layers,
        device=device,
    )
    y_pred_scaled = predict_with_gru(best_model, X_val, batch_size=512, device=device)
    pred_df = build_prediction_dataframe(
        meta_df, y_val_scaled, y_pred_scaled, y_scaler, TARGET_COLS
    )
    pred_df = add_residual_columns(pred_df, TARGET_COLS)
    result_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    metrics = compute_regression_metrics(pred_df, TARGET_COLS)
    best_row = log_df.loc[log_df["val_loss"].idxmin()]
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": dataset_id,
        "turbines": turbines,
        "turbine": dataset_id,
        "targets": TARGET_COLS,
        "output_size": len(TARGET_COLS),
        "epochs_requested": args.epochs,
        "epochs_completed": int(log_df["epoch"].max()),
        "early_stopping_patience": args.patience or None,
        "early_stopped": len(log_df) < args.epochs,
        "random_seed": args.seed,
        "best_epoch": int(best_row["epoch"]),
        "best_val_loss": float(best_row["val_loss"]),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": test_rows,
        "healthy_segment_rows_by_turbine": healthy_segment_rows_by_turbine,
        "train_years": "2016-2022",
        "validation_years": "2016-2022",
        "test_years": "2023-2024",
        "train_ratio": 0.8,
        "missing_values_before_interpolation": missing_before,
        "missing_values_after_interpolation": missing_after_interpolation,
        "missing_values_after_final_dropna": missing_after_dropna,
        "missing_value_handling": (
            "Linear interpolation within each turbine_id and segment_id only; "
            "edge missing values are not extrapolated and remaining NaNs are dropped."
        ),
        "prediction_samples": len(pred_df),
        "sequence_length": args.seq_len,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "amsgrad": args.amsgrad,
        "optimizer": "Adam",
        "loss_function": "MSELoss",
        "input_cols": INPUT_COLS,
        "scaler_type_by_signal": {
            signal: SCALER_TYPE_BY_SIGNAL[signal]
            for signal in list(dict.fromkeys(INPUT_COLS + TARGET_COLS))
        },
        "prediction_units": "physical",
        "scaling_pipeline_version": 3,
        "metrics": metrics,
        "paths": {
            "train_log": str(train_log_path.relative_to(project_root)),
            "predictions": str(prediction_path.relative_to(project_root)),
            "model": str(model_path.relative_to(project_root)),
            "x_scaler": str(x_scaler_path.relative_to(project_root)),
            "y_scaler": str(y_scaler_path.relative_to(project_root)),
            "train_set": str(train_path.relative_to(project_root)),
            "validation_set": str(val_path.relative_to(project_root)),
            "test_set": combined_test_path,
            "test_sets_by_turbine": test_paths,
            "healthy_segments_by_turbine": healthy_segment_paths,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nValidation metrics (physical units):")
    for target_col, target_metrics in metrics["per_target"].items():
        print(target_col)
        for key, value in target_metrics.items():
            print(f"  {key}: {value:.6f}")
    print("Macro average")
    for key, value in metrics["macro_average"].items():
        print(f"  {key}: {value:.6f}")
    print("Best epoch:", int(best_row["epoch"]))
    print("Experiment saved to:", result_dir)


if __name__ == "__main__":
    main()
