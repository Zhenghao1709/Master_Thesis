# src/data/read_scada.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _detect_scada_header_row(csv_path: str | Path, max_scan_lines: int = 50) -> int:
    """
    Greenbyte SCADA 文件常有注释头，自动寻找真正表头。
    规则：找到包含 'Date and time' 的那一行。
    """
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_scan_lines:
                break
            if "Date and time" in line:
                return i
    raise ValueError(f"Could not detect SCADA header row in file: {csv_path}")


def read_scada_csv(
    csv_path: str | Path,
    turbine_id: Optional[str] = None,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    header_row = _detect_scada_header_row(csv_path)

    df = pd.read_csv(
        csv_path,
        sep=",",
        header=header_row,
        encoding="utf-8-sig",
        engine="python",
    )

    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip().lstrip("#").strip() for c in df.columns]

    if "Date and time" not in df.columns:
        print("实际读取到的列名:", df.columns.tolist())
        raise ValueError(f"Missing 'Date and time' column in {csv_path.name}")

    df["Date and time"] = pd.to_datetime(df["Date and time"], errors="coerce")
    df = df[df["Date and time"].notna()].copy()

    if turbine_id is not None:
        df["turbine_id"] = turbine_id

    df["source_file"] = csv_path.name
    df = df.sort_values("Date and time").reset_index(drop=True)
    return df


def read_scada_folder(
    folder_path: str | Path,
    turbine_id: Optional[str] = None,
    pattern: str = "*.csv",
) -> pd.DataFrame:
    folder_path = Path(folder_path)
    files = sorted(folder_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No SCADA files found in {folder_path}")

    dfs = []
    for fp in files:
        dfs.append(read_scada_csv(fp, turbine_id=turbine_id))

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values("Date and time").reset_index(drop=True)
    return out