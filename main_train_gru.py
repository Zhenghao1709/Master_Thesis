# main_train_gru.py

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config.kelmarsh_config import (
    INPUT_COLS,
    TARGET_COL,
    SEQ_LEN,
)
from src.data.split_data import split_healthy_segments_by_time
from src.features.scaling import fit_scalers, apply_scalers, save_scaler
from src.modeling.train_gru import train_gru_model


def main():
    project_root = Path(__file__).resolve().parent
    turbine_name = "Kelmarsh_1"

    healthy_segments_path = (
        project_root
        / "data" / "processed" / "kelmarsh" / "healthy_segments"
        / f"{turbine_name.lower()}_healthy_segments.parquet"
    )

    train_path = (
        project_root
        / "data" / "processed" / "kelmarsh" / "train_val"
        / f"{turbine_name.lower()}_train_healthy.parquet"
    )

    val_path = (
        project_root
        / "data" / "processed" / "kelmarsh" / "train_val"
        / f"{turbine_name.lower()}_val_healthy.parquet"
    )

    model_path = (
        project_root
        / "models" / "kelmarsh"
        / "turbine_specific"
        / f"{turbine_name.lower()}_gru_gen_bearing_front.pth"
    )

    x_scaler_path = (
        project_root
        / "models" / "kelmarsh" / "scalers"
        / f"{turbine_name.lower()}_x_scaler.pkl"
    )

    y_scaler_path = (
        project_root
        / "models" / "kelmarsh" / "scalers"
        / f"{turbine_name.lower()}_y_scaler.pkl"
    )

    train_log_path = (
        project_root
        / "results" / "kelmarsh" / "logs"
        / f"{turbine_name.lower()}_train_log.csv"
    )

    # 1. 读取 healthy segments
    healthy_segments = pd.read_parquet(healthy_segments_path)

    # 2. 只保留训练所需列，并去掉关键列缺失
    required_cols = ["Date and time", "turbine_id", "segment_id"] + INPUT_COLS + [TARGET_COL]
    required_cols = list(dict.fromkeys(required_cols))

    train_df_full = healthy_segments[required_cols].dropna(subset=INPUT_COLS + [TARGET_COL]).copy()

    # 3. 按时间切 train / val
    train_df, val_df = split_healthy_segments_by_time(train_df_full, time_col="Date and time", train_ratio=0.8)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print("train rows:", len(train_df))
    print("val rows:", len(val_df))

    # 4. 标准化（只在训练集 fit）
    x_scaler, y_scaler = fit_scalers(train_df, INPUT_COLS, TARGET_COL)

    train_scaled = apply_scalers(train_df, INPUT_COLS, TARGET_COL, x_scaler, y_scaler)
    val_scaled = apply_scalers(val_df, INPUT_COLS, TARGET_COL, x_scaler, y_scaler)

    save_scaler(x_scaler, x_scaler_path)
    save_scaler(y_scaler, y_scaler_path)

    # 5. 训练模型
    model, log_df = train_gru_model(
        train_df=train_scaled,
        val_df=val_scaled,
        input_cols=INPUT_COLS,
        target_col=TARGET_COL,
        seq_len=SEQ_LEN,
        batch_size=64,
        hidden_size=64,
        num_layers=1,
        lr=1e-3,
        epochs=20,
        model_save_path=model_path,
        log_save_path=train_log_path,
    )

    print("Done.")
    print("Model saved to:", model_path)
    print("Train log saved to:", train_log_path)
    print(log_df.tail())


if __name__ == "__main__":
    main()