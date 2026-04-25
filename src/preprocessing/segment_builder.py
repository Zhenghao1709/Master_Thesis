# src/preprocessing/segment_builder.py

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.kelmarsh_config import FREQ_MINUTES, MIN_SEGMENT_LEN


def split_into_healthy_segments(
    df: pd.DataFrame,
    time_col: str = "Date and time",
    id_col: str = "turbine_id",
    health_col: str = "is_healthy_candidate",
    freq_minutes: int = FREQ_MINUTES,
    min_segment_len: int = MIN_SEGMENT_LEN,
) -> pd.DataFrame:
    expected_delta = pd.Timedelta(minutes=freq_minutes)
    all_segments = []

    for turbine_id, g in df.groupby(id_col):
        g = g.sort_values(time_col).copy()
        g["time_diff"] = g[time_col].diff()

        segment_id = 0
        segment_ids = []
        prev_healthy = False

        for _, row in g.iterrows():
            if not row[health_col]:
                segment_ids.append(np.nan)
                prev_healthy = False
                continue

            if (not prev_healthy) or (row["time_diff"] != expected_delta):
                segment_id += 1

            segment_ids.append(segment_id)
            prev_healthy = True

        g["segment_id"] = segment_ids
        g = g[g["segment_id"].notna()].copy()

        if not g.empty:
            seg_sizes = (
                g.groupby("segment_id")
                .size()
                .rename("segment_len")
                .reset_index()
            )
            g = g.merge(seg_sizes, on="segment_id", how="left")
            g = g[g["segment_len"] >= min_segment_len].copy()

        all_segments.append(g)

    if not all_segments:
        return pd.DataFrame()

    out = pd.concat(all_segments, ignore_index=True)
    return out.sort_values([id_col, time_col]).reset_index(drop=True)