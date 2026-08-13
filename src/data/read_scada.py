# src/data/read_scada.py

from __future__ import annotations

from pathlib import Path
from typing import Optional
import re

import pandas as pd

TIME_COL = "Date and time"
DATA_AVAILABILITY_COL = "Data Availability"
DEFAULT_CHUNK_SIZE = 500_000
CSV_ENGINE = "c"


def _detect_scada_header_row(csv_path: str | Path, max_scan_lines: int = 50) -> int:
    """Find the real Greenbyte SCADA header row."""
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_scan_lines:
                break
            if TIME_COL in line:
                return i
    raise ValueError(f"Could not detect SCADA header row in file: {csv_path}")


def _extract_year_from_filename(filename: str) -> int | None:
    match = re.search(r"_(20\d{2})-\d{2}-\d{2}_", filename)
    if match:
        return int(match.group(1))
    return None


def _read_clean_header(csv_path: str | Path, header_row: int) -> tuple[list[str], dict[str, str]]:
    header_df = pd.read_csv(
        csv_path,
        sep=",",
        header=header_row,
        nrows=0,
        encoding="utf-8-sig",
        engine=CSV_ENGINE,
    )
    raw_cols = list(header_df.columns)
    cleaned_cols = [str(c).strip().lstrip("#").strip() for c in raw_cols]
    clean_to_raw = dict(zip(cleaned_cols, raw_cols))
    return cleaned_cols, clean_to_raw


def _clean_scada_chunk(
    df: pd.DataFrame,
    source_file: str,
    turbine_id: Optional[str],
    keep_cols: Optional[list[str]],
) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip().lstrip("#").strip() for c in df.columns]

    if TIME_COL not in df.columns:
        raise ValueError(f"Missing '{TIME_COL}' column in {source_file}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df[df[TIME_COL].notna()].copy()

    if DATA_AVAILABILITY_COL in df.columns:
        availability = pd.to_numeric(df[DATA_AVAILABILITY_COL], errors="coerce")
        df = df[availability == 1].copy()
        df = df.drop(columns=[DATA_AVAILABILITY_COL])

    if keep_cols is not None:
        keep_existing = [c for c in keep_cols if c in df.columns and c != DATA_AVAILABILITY_COL]
        df = df[keep_existing].copy()

    if turbine_id is not None:
        df["turbine_id"] = turbine_id

    df["source_file"] = source_file

    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")

    return df


def read_scada_csv(
    csv_path: str | Path,
    turbine_id: Optional[str] = None,
    keep_cols: Optional[list[str]] = None,
    chunksize: int | None = None,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    header_row = _detect_scada_header_row(csv_path)
    cleaned_cols, clean_to_raw = _read_clean_header(csv_path, header_row)

    if TIME_COL not in clean_to_raw:
        print("Actual SCADA columns:", cleaned_cols)
        raise ValueError(f"Missing '{TIME_COL}' column in {csv_path.name}")

    usecols_raw = None
    if keep_cols is not None:
        requested_cols = list(dict.fromkeys([TIME_COL, DATA_AVAILABILITY_COL] + keep_cols))
        usecols_raw = [clean_to_raw[c] for c in requested_cols if c in clean_to_raw]

    reader = pd.read_csv(
        csv_path,
        sep=",",
        header=header_row,
        usecols=usecols_raw,
        encoding="utf-8-sig",
        engine=CSV_ENGINE,
        chunksize=chunksize,
    )

    if chunksize is None:
        df = _clean_scada_chunk(reader, csv_path.name, turbine_id, keep_cols)
        return df.sort_values(TIME_COL).reset_index(drop=True)

    chunks = []
    total_rows = 0
    kept_rows = 0
    for chunk_idx, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)
        df = _clean_scada_chunk(chunk, csv_path.name, turbine_id, keep_cols)
        if not df.empty:
            chunks.append(df)
            kept_rows += len(df)
        print(
            f"  {csv_path.name}: chunks={chunk_idx:,} | "
            f"read rows={total_rows:,} | kept rows={kept_rows:,}",
            flush=True,
        )

    if chunks:
        out = pd.concat(chunks, ignore_index=True)
    else:
        out_cols = [c for c in (keep_cols or cleaned_cols) if c != DATA_AVAILABILITY_COL]
        if turbine_id is not None:
            out_cols.append("turbine_id")
        out_cols.append("source_file")
        out = pd.DataFrame(columns=list(dict.fromkeys(out_cols)))

    return out.sort_values(TIME_COL).reset_index(drop=True)


def read_scada_folder(
    folder_path: str | Path,
    turbine_id: Optional[str] = None,
    pattern: str = "*.csv",
    keep_cols: Optional[list[str]] = None,
    year_min: int | None = None,
    year_max: int | None = None,
    chunksize: int | None = DEFAULT_CHUNK_SIZE,
) -> pd.DataFrame:
    folder_path = Path(folder_path)
    files = sorted(folder_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No SCADA files found in {folder_path}")

    dfs = []
    for fp in files:
        year = _extract_year_from_filename(fp.name)
        if year_min is not None and year is not None and year < year_min:
            continue
        if year_max is not None and year is not None and year > year_max:
            continue

        print(f"Reading SCADA: {fp.name}")
        dfs.append(
            read_scada_csv(
                fp,
                turbine_id=turbine_id,
                keep_cols=keep_cols,
                chunksize=chunksize,
            )
        )

    if not dfs:
        raise FileNotFoundError(
            f"No SCADA files matched the requested year range in {folder_path}"
        )

    out = pd.concat(dfs, ignore_index=True)
    return out.sort_values(TIME_COL).reset_index(drop=True)
