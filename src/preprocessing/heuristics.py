# src/preprocessing/heuristics.py

from __future__ import annotations

import pandas as pd

from src.config.kelmarsh_config import (
    INPUT_COLS,
    TARGET_COL,
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


def add_operating_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    定义一个保守的正常运行工况：
    - 风速 > 3 m/s
    - 功率 > 50 kW
    - 发电机转速 > 100 RPM
    可根据实际再调整。
    """
    out = df.copy()

    op_mask = pd.Series(True, index=out.index)

    if "Wind speed (m/s)" in out.columns:
        op_mask &= out["Wind speed (m/s)"] >= 3.0

    if "Power (kW)" in out.columns:
        op_mask &= out["Power (kW)"] >= 50.0

    if "Generator RPM (RPM)" in out.columns:
        op_mask &= out["Generator RPM (RPM)"] >= 100.0

    out["is_normal_operating_condition"] = op_mask
    return out


def add_dirty_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["is_dirty"] = (
        (~out["is_good_quality"])
        | (~out["is_physically_valid"])
        | (out.get("in_communication", False))
    )

    return out


def add_event_buffer_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["is_event_like"] = out.get("in_event", False) | out.get("in_maintenance", False)
    return out


def add_healthy_candidate_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["is_healthy_candidate"] = (
        (~out["is_dirty"])
        & (~out["is_event_like"])
        & out["is_normal_operating_condition"]
    )

    return out


def build_all_heuristic_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    feature_cols = [c for c in INPUT_COLS if c in out.columns]

    out = add_missing_quality_flags(out, feature_cols)
    out = add_physical_flags(out)
    out = add_operating_flags(out)
    out = add_dirty_flags(out)
    out = add_event_buffer_flags(out)
    out = add_healthy_candidate_flags(out)

    return out