# src/data/split_data.py

from __future__ import annotations

import pandas as pd


def split_healthy_segments_by_time(
    df: pd.DataFrame,
    time_col: str = "Date and time",
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    按时间顺序切分 healthy segments。
    不打乱，不随机。
    """
    out = df.copy().sort_values(time_col).reset_index(drop=True)

    split_idx = int(len(out) * train_ratio)
    train_df = out.iloc[:split_idx].copy()
    val_df = out.iloc[split_idx:].copy()

    return train_df, val_df