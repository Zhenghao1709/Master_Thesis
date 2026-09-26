from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from main_predict import add_continuous_prediction_segments, find_experiment, resolve_path
from src.config.kelmarsh_config import INPUT_COLS, SEQ_LEN, TARGET_COLS
from src.detection.residuals import add_residual_columns, compute_regression_metrics
from src.features.scaling import load_scaler
from src.modeling.predict import (
    build_prediction_dataframe,
    load_trained_gru_model,
    predict_with_gru,
    prepare_scaled_sequences,
)


CALIBRATION_START = pd.Timestamp("2021-05-26 12:00:00")
CALIBRATION_END_EXCLUSIVE = pd.Timestamp("2023-01-01 00:00:00")
OUTPUT_NAME = "bcad_calibration_2021-05-26_2022_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict the complete validation-period SCADA timeline for BCAD A-distribution "
            "calibration. This includes both healthy and abnormal operating periods."
        )
    )
    parser.add_argument("--run-id", help="Experiment run ID. Defaults to the newest experiment.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--force", action="store_true", help="Regenerate an existing prediction file.")
    return parser.parse_args()


def load_calibration_scada(project_root: Path, turbines: list[str]) -> pd.DataFrame:
    signal_cols = list(dict.fromkeys(INPUT_COLS + TARGET_COLS))
    required_cols = ["Date and time", "turbine_id"] + signal_cols
    flags_dir = project_root / "data" / "interim" / "kelmarsh" / "flags"
    parts = []

    for turbine_id in turbines:
        path = flags_dir / f"{turbine_id.lower()}_with_flags.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing full SCADA flag file: {path}")
        frame = pd.read_parquet(path, columns=required_cols)
        frame["Date and time"] = pd.to_datetime(frame["Date and time"], errors="coerce")
        frame = frame[
            frame["Date and time"].notna()
            & (frame["Date and time"] >= CALIBRATION_START)
            & (frame["Date and time"] < CALIBRATION_END_EXCLUSIVE)
        ].copy()
        frame = frame.dropna(subset=signal_cols)
        parts.append(frame)

    if not parts:
        raise ValueError("No BCAD calibration SCADA rows were found.")
    combined = pd.concat(parts, ignore_index=True)
    return add_continuous_prediction_segments(combined)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    project_root = PROJECT_ROOT
    result_dir, metadata = find_experiment(project_root, args.run_id)
    output_path = result_dir / OUTPUT_NAME
    if output_path.exists() and not args.force:
        print("Existing BCAD calibration predictions found, skipped:", output_path)
        return

    paths = {
        key: resolve_path(project_root, value)
        for key, value in metadata.get("paths", {}).items()
        if isinstance(value, str)
    }
    calibration_df = load_calibration_scada(project_root, metadata["turbines"])
    x_scaler = load_scaler(paths["x_scaler"])
    y_scaler = load_scaler(paths["y_scaler"])
    seq_len = int(metadata.get("sequence_length", SEQ_LEN))
    X, y_scaled, meta_df = prepare_scaled_sequences(
        calibration_df,
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
    predictions = build_prediction_dataframe(
        meta_df,
        y_scaled,
        y_pred_scaled,
        y_scaler,
        TARGET_COLS,
    )
    predictions = add_residual_columns(predictions, TARGET_COLS)
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")

    metrics = compute_regression_metrics(predictions, TARGET_COLS)
    metadata.setdefault("bcad_calibration", {})["prediction"] = {
        "period_start": CALIBRATION_START.isoformat(),
        "period_end_exclusive": CALIBRATION_END_EXCLUSIVE.isoformat(),
        "source": "complete SCADA timeline, including abnormal periods",
        "input_rows": int(len(calibration_df)),
        "prediction_samples": int(len(predictions)),
        "path": str(output_path.relative_to(project_root)),
        "metrics": metrics,
    }
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Calibration period:", CALIBRATION_START, "to", CALIBRATION_END_EXCLUSIVE)
    print("Calibration input rows:", len(calibration_df))
    print("Prediction samples:", len(predictions))
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
