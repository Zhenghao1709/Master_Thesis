from __future__ import annotations

from pathlib import Path
import pandas as pd

TIME_COL = "Date and time"

KEEP_COLS = [
    "Date and time",
    "Wind speed (m/s)",
    "Power (kW)",
    "Nacelle ambient temperature (°C)",
    "Nacelle temperature (°C)",
    "Generator RPM (RPM)",
    "Rotor speed (RPM)",
    "Stator temperature 1 (°C)",
    "Generator bearing front temperature (°C)",
]


def detect_header_row(csv_path: Path, max_scan_lines: int = 50) -> int:
    with csv_path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_scan_lines:
                break
            if "Date and time" in line:
                return i
    raise ValueError(f"Could not detect header row in {csv_path}")


def extract_one_file(csv_path: Path, out_path: Path, turbine_name: str):
    print(f"Reading: {csv_path.name}")
    header_row = detect_header_row(csv_path)

    # 先只读表头，拿到真实列名
    header_df = pd.read_csv(
        csv_path,
        header=header_row,
        nrows=0,
        encoding="utf-8-sig",
        engine="python",
    )
    cols = [str(c).strip().lstrip("#").strip() for c in header_df.columns]

    usecols = [c for c in KEEP_COLS if c in cols]
    if "Date and time" not in usecols:
        raise ValueError(f"'Date and time' not found in {csv_path.name}")

    df = pd.read_csv(
        csv_path,
        header=header_row,
        usecols=usecols,
        encoding="utf-8-sig",
        engine="python",
    )

    df.columns = [str(c).strip().lstrip("#").strip() for c in df.columns]
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df[df[TIME_COL].notna()].copy()

    # 压缩内存
    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")

    df["turbine_id"] = turbine_name
    df["source_file"] = csv_path.name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path} | rows={len(df):,}")


def main():
    project_root = Path(__file__).resolve().parent
    turbine_name = "Kelmarsh_1"

    in_folder = project_root / "data" / "raw" / "kelmarsh" / "scada" / turbine_name
    out_folder = project_root / "data" / "interim" / "kelmarsh" / "light_scada_parts" / turbine_name

    files = sorted(in_folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No SCADA files found in {in_folder}")

    for fp in files:
        out_path = out_folder / f"{fp.stem}.parquet"
        extract_one_file(fp, out_path, turbine_name)


if __name__ == "__main__":
    main()