from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

TIME_COL = "Date and time"
YEAR_MIN = 2016
YEAR_MAX = 2024


def extract_year_from_filename(filename: str) -> int | None:
    m = re.search(r"_(20\d{2})-\d{2}-\d{2}_", filename)
    if m:
        return int(m.group(1))
    return None


def merge_one_turbine(project_root: Path, turbine_name: str) -> pd.DataFrame | None:
    in_folder = project_root / "data" / "interim" / "kelmarsh" / "light_scada_parts" / turbine_name
    out_path = (
        project_root
        / "data" / "interim" / "kelmarsh" / "merged_scada"
        / f"{turbine_name.lower()}_scada_merged_raw_2016_2024.parquet"
    )

    parts = []
    for p in sorted(in_folder.glob("*.parquet")):
        year = extract_year_from_filename(p.name)
        if year is not None and YEAR_MIN <= year <= YEAR_MAX:
            parts.append(p)

    if not parts:
        print(f"[WARN] No light_scada_parts found for {turbine_name} ({YEAR_MIN}-{YEAR_MAX})")
        return None

    print(f"\n=== Merging light SCADA parts for {turbine_name} ({YEAR_MIN}-{YEAR_MAX}) ===")
    dfs = []

    for p in parts:
        print(f"Loading part: {p.name}")
        df = pd.read_parquet(p)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged[TIME_COL] = pd.to_datetime(merged[TIME_COL], errors="coerce")
    merged = merged[merged[TIME_COL].notna()].copy()
    merged = merged.sort_values(TIME_COL).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    print(f"Saved merged file: {out_path}")
    print(f"Rows: {len(merged):,}")
    return merged


def main():
    project_root = Path(__file__).resolve().parent
    turbine_list = [f"Kelmarsh_{i}" for i in range(1, 7)]

    all_parts = []

    for turbine_name in turbine_list:
        merged = merge_one_turbine(project_root, turbine_name)
        if merged is not None:
            all_parts.append(merged)

    if all_parts:
        print(f"\n=== Building all-turbines merged file ({YEAR_MIN}-{YEAR_MAX}) ===")
        all_df = pd.concat(all_parts, ignore_index=True)
        all_df = all_df.sort_values(["turbine_id", TIME_COL]).reset_index(drop=True)

        out_path = (
            project_root
            / "data" / "interim" / "kelmarsh" / "merged_scada"
            / f"all_turbines_scada_merged_raw_{YEAR_MIN}_{YEAR_MAX}.parquet"
        )
        all_df.to_parquet(out_path, index=False)

        print(f"Saved all-turbines merged file: {out_path}")
        print(f"Rows: {len(all_df):,}")


if __name__ == "__main__":
    main()
