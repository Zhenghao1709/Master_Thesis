# src/preprocessing/clean_scada.py

from __future__ import annotations

import pandas as pd


def clean_scada_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out[out["Date and time"].notna()].copy()
    out = out.sort_values(["turbine_id", "Date and time"] if "turbine_id" in out.columns else ["Date and time"]).reset_index(drop=True)

    return out


def keep_required_columns(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required SCADA columns: {missing}")
    return df[required_cols].copy()