# src/data/split_data.py

from __future__ import annotations

import pandas as pd


def partition_development_and_test_years(
    df: pd.DataFrame,
    time_col: str = "Date and time",
    development_start: str = "2016-01-01",
    test_start: str = "2023-01-01",
    test_end_exclusive: str = "2025-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep 2016-2022 for development and preserve 2023-2024 as test data."""
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out[out[time_col].notna()].sort_values(time_col).reset_index(drop=True)

    development_start_ts = pd.Timestamp(development_start)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end_exclusive)
    if not development_start_ts < test_start_ts < test_end_ts:
        raise ValueError("Expected development_start < test_start < test_end_exclusive.")

    development = out[
        (out[time_col] >= development_start_ts)
        & (out[time_col] < test_start_ts)
    ].copy()
    test = out[
        (out[time_col] >= test_start_ts)
        & (out[time_col] < test_end_ts)
    ].copy()
    return development.reset_index(drop=True), test.reset_index(drop=True)


def split_healthy_segments_by_time(
    df: pd.DataFrame,
    time_col: str = "Date and time",
    train_ratio: float = 0.8,
    development_start: str = "2016-01-01",
    development_end_exclusive: str = "2023-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split 2016-2022 healthy data without cutting a segment."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between 0 and 1.")

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out[
        out[time_col].notna()
        & (out[time_col] >= pd.Timestamp(development_start))
        & (out[time_col] < pd.Timestamp(development_end_exclusive))
    ].sort_values(time_col).reset_index(drop=True)
    if len(out) < 2:
        raise ValueError("At least two healthy development rows are required.")

    desired_idx = int(len(out) * train_ratio)
    split_idx = desired_idx
    segment_key_cols = [
        column for column in ("turbine_id", "segment_id") if column in out.columns
    ]
    if segment_key_cols:
        segment_change = out[segment_key_cols].ne(
            out[segment_key_cols].shift()
        ).any(axis=1)
        boundaries = segment_change[segment_change].index.to_numpy()
        boundaries = boundaries[(boundaries > 0) & (boundaries < len(out))]
        if len(boundaries):
            split_idx = int(boundaries[abs(boundaries - desired_idx).argmin()])

    train_df = out.iloc[:split_idx].copy()
    val_df = out.iloc[split_idx:].copy()

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
