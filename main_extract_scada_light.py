from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

TIME_COL = "Date and time"
DATA_AVAILABILITY_COL = "Data Availability"
CHUNK_SIZE = 500_000

KEEP_COLS = [
    "Date and time",
    "Data Availability",
    "Wind speed (m/s)",
    "Power (kW)",
    "Nacelle ambient temperature (°C)",
    "Nacelle temperature (°C)",
    "Generator RPM (RPM)",
    "Rotor speed (RPM)",
    "Gearbox speed (RPM)",
    "Ambient temperature (converter) (°C)",
    "Stator temperature 1 (°C)",
    "Generator bearing front temperature (°C)",
    "Rear bearing temperature (°C)",
]

YEAR_MIN = 2016
YEAR_MAX = 2024


def detect_header_row(csv_path: Path, max_scan_lines: int = 50) -> int:
    with csv_path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_scan_lines:
                break
            if "Date and time" in line:
                return i
    raise ValueError(f"Could not detect header row in {csv_path}")


def extract_year_from_filename(filename: str) -> int | None:
    """
    Example:
    Turbine_Data_Kelmarsh_1_2016-01-03_-_2017-01-01_228.csv
    -> 2016
    """
    m = re.search(r"_(20\d{2})-\d{2}-\d{2}_", filename)
    if m:
        return int(m.group(1))
    return None


def extract_one_file(csv_path: Path, out_path: Path, turbine_name: str):
    print(f"Reading: {csv_path.name}")
    header_row = detect_header_row(csv_path)

    # Read raw header first
    header_df = pd.read_csv(
        csv_path,
        header=header_row,
        nrows=0,
        encoding="utf-8-sig",
        engine="python",
    )

    raw_cols = list(header_df.columns)
    cleaned_cols = [str(c).strip().lstrip("#").strip() for c in raw_cols]

    raw_to_clean = dict(zip(raw_cols, cleaned_cols))
    clean_to_raw = {v: k for k, v in raw_to_clean.items()}

    usecols_raw = [clean_to_raw[c] for c in KEEP_COLS if c in clean_to_raw]

    if "Date and time" not in clean_to_raw:
        raise ValueError(f"'Date and time' not found in cleaned header of {csv_path.name}")

    chunks = []
    total_rows = 0
    kept_rows = 0

    reader = pd.read_csv(
        csv_path,
        header=header_row,
        usecols=usecols_raw,
        encoding="utf-8-sig",
        engine="python",
        chunksize=CHUNK_SIZE,
    )

    for chunk_idx, df in enumerate(reader, start=1):
        total_rows += len(df)
        df.columns = [str(c).strip().lstrip("#").strip() for c in df.columns]

        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df[df[TIME_COL].notna()].copy()

        if DATA_AVAILABILITY_COL in df.columns:
            availability = pd.to_numeric(df[DATA_AVAILABILITY_COL], errors="coerce")
            df = df[availability == 1].copy()
            df = df.drop(columns=[DATA_AVAILABILITY_COL])

        # Reduce memory after filtering each chunk.
        float_cols = df.select_dtypes(include=["float64"]).columns
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].astype("float32")

        if not df.empty:
            kept_rows += len(df)
            chunks.append(df)

        if chunk_idx % 10 == 0:
            print(f"  chunks={chunk_idx:,} | read rows={total_rows:,} | kept rows={kept_rows:,}")

    if chunks:
        df = pd.concat(chunks, ignore_index=True)
        df = df.sort_values(TIME_COL).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=[c for c in KEEP_COLS if c != DATA_AVAILABILITY_COL])

    df["turbine_id"] = turbine_name
    df["source_file"] = csv_path.name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path} | read rows={total_rows:,} | kept rows={len(df):,}")


def main():
    project_root = Path(__file__).resolve().parent

    turbine_list = [f"Kelmarsh_{i}" for i in range(1, 7)]

    for turbine_name in turbine_list:
        print(f"\n=== Extracting light SCADA for {turbine_name} ({YEAR_MIN}-{YEAR_MAX}) ===")
        in_folder = project_root / "data" / "raw" / "kelmarsh" / "scada" / turbine_name
        out_folder = project_root / "data" / "interim" / "kelmarsh" / "light_scada_parts" / turbine_name

        files = sorted(in_folder.glob("*.csv"))
        if not files:
            print(f"[WARN] No SCADA files found in {in_folder}")
            continue

        for fp in files:
            year = extract_year_from_filename(fp.name)
            if year is None:
                print(f"[WARN] Could not detect year from filename, skipped: {fp.name}")
                continue

            if not (YEAR_MIN <= year <= YEAR_MAX):
                print(f"Skipping (out of range): {fp.name}")
                continue

            out_path = out_folder / f"{fp.stem}.parquet"
            if out_path.exists():
                print(f"Already exists, skipped: {out_path.name}")
                continue

            try:
                extract_one_file(fp, out_path, turbine_name)
            except Exception as e:
                print(f"[ERROR] Failed for {fp.name}: {e}")


if __name__ == "__main__":
    main()
