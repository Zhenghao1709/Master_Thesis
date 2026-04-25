from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


EXPECTED_STATUS_COLUMNS = [
    "Timestamp start",
    "Timestamp end",
    "Duration",
    "Status",
    "Code",
    "Message",
    "Comment",
    "Service contract category",
    "IEC category",
]


def _detect_header_row(csv_path: str | Path, max_scan_lines: int = 50) -> int:
    """
    扫描前若干行，找到真正的表头行。
    规则：包含 'Timestamp start' 和 'Timestamp end' 即认为是表头。
    """
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_scan_lines:
                break
            if "Timestamp start" in line and "Timestamp end" in line:
                return i
    raise ValueError(f"Could not detect header row in status file: {csv_path}")


def read_status_csv(
    csv_path: str | Path,
    turbine_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    读取单个 status csv。
    自动寻找表头行，解析时间列，并附加 turbine_id。
    """
    csv_path = Path(csv_path)
    header_row = _detect_header_row(csv_path)

    df = pd.read_csv(
        csv_path,
        sep=",",
        header=header_row,
        encoding="utf-8-sig",
        engine="python",
    )

    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in ["Timestamp start", "Timestamp end", "Status"] if c not in df.columns]
    if missing:
        print("实际读取到的列名:", df.columns.tolist())
        raise ValueError(f"Missing required status columns in {csv_path.name}: {missing}")

    df["Timestamp start"] = pd.to_datetime(df["Timestamp start"], errors="coerce")
    df["Timestamp end"] = pd.to_datetime(df["Timestamp end"], errors="coerce")

    if "Duration" in df.columns:
        try:
            df["Duration"] = pd.to_timedelta(df["Duration"], errors="coerce")
        except Exception:
            pass

    if turbine_id is not None:
        df["turbine_id"] = turbine_id

    df["source_file"] = csv_path.name
    df = df.sort_values(["Timestamp start", "Timestamp end"]).reset_index(drop=True)
    return df


def read_status_folder(
    folder_path: str | Path,
    turbine_id: Optional[str] = None,
    pattern: str = "*.csv",
) -> pd.DataFrame:
    """
    读取一个 turbine 某目录下的所有 status csv，并合并。
    """
    folder_path = Path(folder_path)
    files = sorted(folder_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No status files found in {folder_path}")

    dfs = []
    for fp in files:
        dfs.append(read_status_csv(fp, turbine_id=turbine_id))

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values(["Timestamp start", "Timestamp end"]).reset_index(drop=True)
    return out