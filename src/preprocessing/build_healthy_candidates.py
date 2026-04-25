# src/preprocessing/build_healthy_candidates.py

from __future__ import annotations

import pandas as pd


def extract_healthy_candidates(df: pd.DataFrame, cols_to_keep: list[str]) -> pd.DataFrame:
    keep_cols = cols_to_keep + [
        "row_missing_ratio",
        "is_good_quality",
        "is_physically_valid",
        "is_normal_operating_condition",
        "is_dirty",
        "is_event_like",
        "is_healthy_candidate",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df.loc[df["is_healthy_candidate"], keep_cols].copy()