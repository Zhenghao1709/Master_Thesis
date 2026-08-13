# src/modeling/predict.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.modeling.dataset import create_sequences
from src.modeling.model_gru import GRUNBM
from src.features.scaling import load_scaler


def load_trained_gru_model(
    model_path: str | Path,
    input_size: int,
    output_size: int,
    hidden_size: int = 64,
    num_layers: int = 1,
    device: torch.device | None = None,
) -> GRUNBM:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRUNBM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def prepare_scaled_sequences(
    df: pd.DataFrame,
    input_cols: list[str],
    target_cols: list[str],
    seq_len: int,
    x_scaler,
    y_scaler,
    time_col: str = "Date and time",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    1. 对输入数据做 scaler transform
    2. 切成序列
    3. 返回：
       - X
       - y_scaled
       - 对应每条样本的 metadata（时间、segment_id、turbine_id）
    """
    out = df.copy()

    # Targets may also be autoregressive inputs. Keep their raw values so input
    # and output scaling are each applied to the original signals.
    raw_targets = out[target_cols].copy()
    out[input_cols] = x_scaler.transform(out[input_cols])
    out[target_cols] = y_scaler.transform(raw_targets)

    X, y_scaled = create_sequences(
        out,
        input_cols=input_cols,
        target_cols=target_cols,
        seq_len=seq_len,
    )

    meta_rows = []
    group_cols = [c for c in ["turbine_id", "segment_id"] if c in out.columns]

    for _, g in out.groupby(group_cols):
        g = g.sort_values(time_col).reset_index(drop=True)

        if len(g) <= seq_len:
            continue

        for i in range(len(g) - seq_len):
            meta_rows.append({
                "turbine_id": g.loc[i + seq_len, "turbine_id"] if "turbine_id" in g.columns else None,
                "segment_id": g.loc[i + seq_len, "segment_id"] if "segment_id" in g.columns else None,
                "Date and time": g.loc[i + seq_len, time_col],
            })

    meta_df = pd.DataFrame(meta_rows)
    return X, y_scaled, meta_df


def predict_with_gru(
    model,
    X: np.ndarray,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            xb = torch.tensor(X[start:end], dtype=torch.float32).to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)

    return np.concatenate(preds, axis=0)


def build_prediction_dataframe(
    meta_df: pd.DataFrame,
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    y_scaler,
    target_cols: list[str],
) -> pd.DataFrame:
    y_true = y_scaler.inverse_transform(y_true_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    pred_df = meta_df.copy()
    for index, target_col in enumerate(target_cols):
        pred_df[f"y_true::{target_col}"] = y_true[:, index]
        pred_df[f"y_pred::{target_col}"] = y_pred[:, index]
    return pred_df
