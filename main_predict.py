from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.config.kelmarsh_config import FREQ_MINUTES, INPUT_COLS, TARGET_COLS, SEQ_LEN
from src.detection.residuals import add_residual_columns, compute_regression_metrics
from src.features.scaling import load_scaler
from src.modeling.predict import (
    build_prediction_dataframe,
    load_trained_gru_model,
    predict_with_gru,
    prepare_scaled_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 2023-2024 test predictions for a saved GRU NBM experiment."
    )
    parser.add_argument("--run-id", help="Experiment run ID. Defaults to the newest experiment.")
    parser.add_argument("--batch-size", type=int, default=512, help="Prediction batch size")
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


def resolve_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def load_test_set(project_root: Path, metadata: dict) -> pd.DataFrame:
    test_path_value = metadata.get("paths", {}).get("test_set")
    if test_path_value is None:
        dataset_id = metadata.get("dataset_id", metadata.get("turbine", "all6"))
        test_path = (
            project_root / "data" / "processed" / "kelmarsh" / "train_val"
            / f"{dataset_id}_test_2023_2024.parquet"
        )
    else:
        test_path = resolve_path(project_root, test_path_value)

    if not test_path.exists():
        raise FileNotFoundError(f"Missing test set: {test_path}")
    return pd.read_parquet(test_path).copy()


def add_continuous_prediction_segments(
    df: pd.DataFrame,
    time_col: str = "Date and time",
    turbine_col: str = "turbine_id",
    freq_minutes: int = FREQ_MINUTES,
) -> pd.DataFrame:
    expected_delta = pd.Timedelta(minutes=freq_minutes)
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out[out[time_col].notna()].copy()
    out = out.sort_values([turbine_col, time_col]).reset_index(drop=True)

    parts = []
    for turbine_id, group in out.groupby(turbine_col, sort=False):
        group = group.sort_values(time_col).copy()
        segment_start = group[time_col].diff().ne(expected_delta)
        group["segment_id"] = segment_start.cumsum().astype(int)
        group["segment_id"] = turbine_id + "_testseg_" + group["segment_id"].astype(str)
        parts.append(group)

    if not parts:
        raise ValueError("No test rows remain after timestamp filtering.")
    return pd.concat(parts, ignore_index=True)


def prepare_test_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    signal_cols = list(dict.fromkeys(INPUT_COLS + TARGET_COLS))
    required_cols = ["Date and time", "turbine_id"] + signal_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required test columns: {missing_cols}")

    out = df[required_cols].copy()
    out["Date and time"] = pd.to_datetime(out["Date and time"], errors="coerce")
    out = out.dropna(subset=signal_cols).copy()

    valid_years = out["Date and time"].dt.year.isin([2023, 2024])
    if not valid_years.all():
        raise ValueError("Test prediction input contains rows outside 2023-2024.")

    out = add_continuous_prediction_segments(out)
    return out.sort_values(["turbine_id", "segment_id", "Date and time"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    project_root = Path(__file__).resolve().parent
    result_dir, metadata = find_experiment(project_root, args.run_id)
    if metadata.get("scaling_pipeline_version") != 3:
        raise ValueError("This experiment is not compatible with the current prediction pipeline.")

    paths = {
        key: resolve_path(project_root, value)
        for key, value in metadata.get("paths", {}).items()
        if isinstance(value, str)
    }
    test_df = prepare_test_dataframe(load_test_set(project_root, metadata))

    x_scaler = load_scaler(paths["x_scaler"])
    y_scaler = load_scaler(paths["y_scaler"])
    seq_len = int(metadata.get("sequence_length", SEQ_LEN))
    X, y_scaled, meta_df = prepare_scaled_sequences(
        test_df,
        INPUT_COLS,
        TARGET_COLS,
        seq_len,
        x_scaler,
        y_scaler,
        "Date and time",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_gru_model(
        paths["model"],
        input_size=len(INPUT_COLS),
        output_size=len(TARGET_COLS),
        hidden_size=int(metadata["hidden_size"]),
        num_layers=int(metadata["num_layers"]),
        device=device,
    )
    y_pred_scaled = predict_with_gru(model, X, batch_size=args.batch_size, device=device)
    pred_df = build_prediction_dataframe(meta_df, y_scaled, y_pred_scaled, y_scaler, TARGET_COLS)
    pred_df = add_residual_columns(pred_df, TARGET_COLS)

    output_path = result_dir / "test_2023_2024_predictions.csv"
    pred_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    metrics = compute_regression_metrics(pred_df, TARGET_COLS)
    metadata["test_prediction"] = {
        "path": str(output_path.relative_to(project_root)),
        "input_rows": len(test_df),
        "prediction_samples": len(pred_df),
        "batch_size": args.batch_size,
        "metrics": metrics,
    }
    metadata.setdefault("paths", {})["test_predictions"] = str(output_path.relative_to(project_root))
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Run ID:", metadata["run_id"])
    print("Test input rows:", len(test_df))
    print("Prediction samples:", len(pred_df))
    print("Prediction file saved to:", output_path)
    print("\nTest prediction metrics in physical units:")
    for target_col, target_metrics in metrics["per_target"].items():
        print(target_col)
        for key, value in target_metrics.items():
            print(f"  {key}: {value:.6f}")
    print("Macro average")
    for key, value in metrics["macro_average"].items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
