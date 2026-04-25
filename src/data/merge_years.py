# src/data/merge_years.py

from __future__ import annotations

from pathlib import Path
import pandas as pd


def merge_and_save(df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def deduplicate_scada(df: pd.DataFrame, time_col: str = "Date and time") -> pd.DataFrame:
    subset = [c for c in ["turbine_id", time_col] if c in df.columns]
    if not subset:
        subset = [time_col]
    return df.drop_duplicates(subset=subset).sort_values(time_col).reset_index(drop=True)


def deduplicate_status(df: pd.DataFrame) -> pd.DataFrame:
    dedup_cols = [c for c in [
        "turbine_id",
        "Timestamp start",
        "Timestamp end",
        "Status",
        "Code",
        "Message",
        "IEC category",
    ] if c in df.columns]
    return df.drop_duplicates(subset=dedup_cols).sort_values(["Timestamp start", "Timestamp end"]).reset_index(drop=True)