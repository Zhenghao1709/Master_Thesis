# src/preprocessing/heuristics.py

from __future__ import annotations

import pandas as pd

from src.config.kelmarsh_config import (
    PREPROCESSING_SIGNAL_COLS,
    PHYSICAL_LIMITS,
    MAX_MISSING_RATIO_PER_ROW,
)


def add_missing_quality_flags(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["row_missing_ratio"] = out[feature_cols].isna().mean(axis=1)
    out["is_good_quality"] = out["row_missing_ratio"] <= MAX_MISSING_RATIO_PER_ROW
    return out


def add_physical_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mask = pd.Series(True, index=out.index)

    for col, (low, high) in PHYSICAL_LIMITS.items():
        if col in out.columns:
            mask &= out[col].between(low, high, inclusive="both") | out[col].isna()

    out["is_physically_valid"] = mask
    return out


def add_dirty_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["is_dirty"] = (
        (~out["is_good_quality"])
        | (~out["is_physically_valid"])
        | (out.get("in_communication", False))
        | (out.get("in_curtailment", False))
    )

    return out


def add_event_buffer_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["is_event_like"] = (
        out.get("in_event", False)
        | out.get("in_maintenance", False)
        | out.get("in_manual_event", False)
        | out.get("in_extended_fault_window", False)
    )
    return out


def add_healthy_candidate_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["is_healthy_candidate"] = (
        (~out["is_dirty"])
        & (~out["is_event_like"])
    )

    return out


def build_all_heuristic_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    feature_cols = [c for c in PREPROCESSING_SIGNAL_COLS if c in out.columns]

    out = add_missing_quality_flags(out, feature_cols)
    out = add_physical_flags(out)
    out = add_dirty_flags(out)
    out = add_event_buffer_flags(out)
    out = add_healthy_candidate_flags(out)

    return out
