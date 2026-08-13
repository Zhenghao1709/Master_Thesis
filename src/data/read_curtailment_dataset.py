# src/data/read_curtailment_dataset.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _detect_curtailment_header_row(csv_path: str | Path, max_scan_lines: int = 50) -> int:
    """
    Automatically detect the header row in the curtailment dataset.
    We assume the true header contains both 'Timestamp start' and 'Timestamp end'.
    """
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_scan_lines:
                break
            if "Timestamp start" in line and "Timestamp end" in line:
                return i
    raise ValueError(f"Could not detect header row in curtailment dataset: {csv_path}")


def read_curtailment_dataset(
    csv_path: str | Path,
    turbine_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read the manually prepared Kelmarsh curtailment dataset.
    """
    csv_path = Path(csv_path)
    header_row = _detect_curtailment_header_row(csv_path)

    df = pd.read_csv(
        csv_path,
        header=header_row,
        encoding="utf-8-sig",
        engine="python",
    )

    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    datetime_cols = ["Timestamp start", "Timestamp end"]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Turbine" in df.columns:
        df["Turbine"] = df["Turbine"].astype(str).str.strip()

    if turbine_filter is not None and "Turbine" in df.columns:
        turbine_num = turbine_filter.split("_")[-1]

        turbine_mask = (
            df["Turbine"].str.contains(turbine_filter, case=False, na=False)
            | df["Turbine"].str.contains(turbine_num, case=False, na=False)
        )
        df = df.loc[turbine_mask].copy()

    numeric_cols = ["Curtailment", "Max observed rate"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sort_cols = [c for c in ["Timestamp start", "Timestamp end"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    df["source_file"] = csv_path.name
    return df
