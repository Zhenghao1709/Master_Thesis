# main_build_healthy_data.py

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.config.kelmarsh_config import (
    TIME_COL,
    TARGET_COLS,
    PREPROCESSING_SIGNAL_COLS,
    TURBINE_ID_COL,
)
from src.data.read_status import read_status_folder
from src.data.read_scada import read_scada_folder
from src.data.read_event_dataset import read_event_dataset
from src.data.read_curtailment_dataset import read_curtailment_dataset
from src.data.merge_years import deduplicate_scada, deduplicate_status, merge_and_save
from src.data.align_scada_status import align_status_to_scada
from src.data.align_interval_flags import add_interval_flag_to_scada
from src.data.split_data import partition_development_and_test_years
from src.config.status_mapping import (
    DEFAULT_EVENT_POST_MONTHS,
    DEFAULT_EVENT_PRE_MONTHS,
)
from src.preprocessing.clean_scada import clean_scada_df
from src.preprocessing.clean_status import clean_status_df
from src.preprocessing.heuristics import build_all_heuristic_flags
from src.preprocessing.build_healthy_candidates import extract_healthy_candidates
from src.preprocessing.segment_builder import split_into_healthy_segments


YEAR_MIN = 2016
YEAR_MAX = 2024


def get_required_scada_cols():
    return list(dict.fromkeys(
        [TIME_COL, TURBINE_ID_COL] + PREPROCESSING_SIGNAL_COLS + TARGET_COLS + ["source_file"]
    ))


def main(turbine_name: str):
    project_root = Path(__file__).resolve().parent

    scada_folder = project_root / "data" / "raw" / "kelmarsh" / "scada" / turbine_name
    status_folder = project_root / "data" / "raw" / "kelmarsh" / "status" / turbine_name
    auxiliary_folder = project_root / "data" / "raw" / "kelmarsh" / "auxiliary"
    event_dataset_path = auxiliary_folder / "kelmarsh_event_dataset.csv"
    curtailment_dataset_path = auxiliary_folder / "kelmarsh_curtailment_dataset.csv"

    interim_root = project_root / "data" / "interim" / "kelmarsh"
    processed_root = project_root / "data" / "processed" / "kelmarsh"

    cleaned_scada_path = interim_root / "cleaned_scada" / f"{turbine_name.lower()}_scada_clean_2016_2024.parquet"
    merged_status_path = interim_root / "merged_status" / f"{turbine_name.lower()}_status_2016_2024.parquet"
    aligned_path = interim_root / "aligned" / f"{turbine_name.lower()}_scada_with_status.parquet"
    flags_path = interim_root / "flags" / f"{turbine_name.lower()}_with_flags.parquet"

    healthy_candidates_path = processed_root / "healthy_candidates" / f"{turbine_name.lower()}_healthy_candidates.parquet"
    healthy_segments_path = processed_root / "healthy_segments" / f"{turbine_name.lower()}_healthy_segments.parquet"
    test_path = (
        processed_root / "train_val"
        / f"{turbine_name.lower()}_test_2023_2024.parquet"
    )

    print(f"\n=== Processing {turbine_name} ===")

    # 1. Read raw SCADA directly for this turbine. This replaces the previous
    # extract-light-parts-then-merge step.
    print("1/9 Reading raw SCADA directly for one turbine...")
    scada_keep_cols = [
        c for c in get_required_scada_cols()
        if c not in {TURBINE_ID_COL, "source_file"}
    ]
    scada_df = read_scada_folder(
        scada_folder,
        turbine_id=turbine_name,
        keep_cols=scada_keep_cols,
        year_min=YEAR_MIN,
        year_max=YEAR_MAX,
    )
    if TURBINE_ID_COL not in scada_df.columns:
        scada_df[TURBINE_ID_COL] = turbine_name

    # Keep only required columns as early as possible
    missing_scada_cols = [c for c in get_required_scada_cols() if c not in scada_df.columns and c != "source_file"]
    if missing_scada_cols:
        raise ValueError(f"Missing required SCADA columns for {turbine_name}: {missing_scada_cols}")
    required_scada_cols = [c for c in get_required_scada_cols() if c in scada_df.columns]
    scada_df = scada_df[required_scada_cols].copy()

    # 2. Read status
    print("2/9 Reading status and manual interval labels...")
    status_df = read_status_folder(status_folder, turbine_id=turbine_name)
    event_df = read_event_dataset(event_dataset_path, turbine_filter=turbine_name)
    curtailment_df = read_curtailment_dataset(curtailment_dataset_path, turbine_filter=turbine_name)

    # 3. Clean + deduplicate
    print("3/9 Cleaning and deduplicating...")
    scada_df = clean_scada_df(scada_df)
    scada_df = deduplicate_scada(scada_df)

    status_df = clean_status_df(status_df)
    status_df = deduplicate_status(status_df)

    # Save per-turbine interim files.
    merge_and_save(scada_df, cleaned_scada_path)
    merge_and_save(status_df, merged_status_path)

    print(f"   SCADA rows: {len(scada_df):,}")
    print(f"   Status rows: {len(status_df):,}")

    # 4. Align status to SCADA
    print("4/9 Aligning status to SCADA...")
    aligned_df = align_status_to_scada(
        scada_df=scada_df,
        status_df=status_df,
        scada_time_col=TIME_COL,
        scada_turbine_col=TURBINE_ID_COL,
        status_turbine_col=TURBINE_ID_COL,
    )
    aligned_df = add_interval_flag_to_scada(
        scada_df=aligned_df,
        intervals_df=event_df,
        flag_col="in_manual_event",
        scada_time_col=TIME_COL,
    )
    if "Category" in event_df.columns:
        confirmed_fault_events = event_df[
            event_df["Category"].astype(str).str.strip().str.casefold().eq("fault")
        ].copy()
    else:
        confirmed_fault_events = event_df.copy()
    aligned_df = add_interval_flag_to_scada(
        scada_df=aligned_df,
        intervals_df=confirmed_fault_events,
        flag_col="in_extended_fault_window",
        scada_time_col=TIME_COL,
        pre_months=DEFAULT_EVENT_PRE_MONTHS,
        post_months=DEFAULT_EVENT_POST_MONTHS,
    )
    aligned_df = add_interval_flag_to_scada(
        scada_df=aligned_df,
        intervals_df=curtailment_df,
        flag_col="in_curtailment",
        scada_time_col=TIME_COL,
    )
    merge_and_save(aligned_df, aligned_path)

    # 5. Build heuristic flags
    print("5/9 Building heuristic flags...")
    flagged_df = build_all_heuristic_flags(aligned_df)
    merge_and_save(flagged_df, flags_path)

    # Only 2016-2022 may supply healthy training/validation rows. The complete
    # 2023-2024 timeline (including event and quality flags) is retained as test.
    development_df, test_df = partition_development_and_test_years(
        flagged_df,
        time_col=TIME_COL,
    )
    merge_and_save(test_df, test_path)

    # Print key ratios
    summary_cols = [
        "is_good_quality",
        "is_physically_valid",
        "is_dirty",
        "is_event_like",
        "is_healthy_candidate",
        "in_manual_event",
        "in_extended_fault_window",
        "in_curtailment",
    ]
    existing_summary_cols = [c for c in summary_cols if c in flagged_df.columns]
    print(flagged_df[existing_summary_cols].mean())

    # 6. Extract healthy candidates
    print("6/9 Extracting healthy candidates...")
    cols_to_keep = [TIME_COL, TURBINE_ID_COL] + PREPROCESSING_SIGNAL_COLS + TARGET_COLS
    cols_to_keep = list(dict.fromkeys(cols_to_keep))
    healthy_candidates = extract_healthy_candidates(development_df, cols_to_keep)
    merge_and_save(healthy_candidates, healthy_candidates_path)

    print(f"   Healthy candidate rows: {len(healthy_candidates):,}")

    # 7. Build healthy segments
    print("7/9 Building healthy segments...")
    healthy_segments = split_into_healthy_segments(
        development_df,
        time_col=TIME_COL,
        id_col=TURBINE_ID_COL,
        health_col="is_healthy_candidate",
    )
    merge_and_save(healthy_segments, healthy_segments_path)

    print(f"   Healthy segment rows: {len(healthy_segments):,}")
    print(f"   Test rows (2023-2024, unfiltered): {len(test_df):,}")
    if not healthy_segments.empty and "segment_id" in healthy_segments.columns:
        n_segments = healthy_segments["segment_id"].nunique()
        print(f"   Number of segments: {n_segments:,}")

    # 8. Done
    print("8/9 Saved outputs:")
    print("   raw_scada_folder:", scada_folder)
    print("   cleaned_scada:", cleaned_scada_path)
    print("   status:", merged_status_path)
    print("   aligned:", aligned_path)
    print("   flags:", flags_path)
    print("   healthy_candidates:", healthy_candidates_path)
    print("   healthy_segments:", healthy_segments_path)
    print("   test_2023_2024:", test_path)

    print(f"=== Done: {turbine_name} ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build healthy data for one Kelmarsh turbine.")
    parser.add_argument(
        "--turbine",
        type=str,
        default="Kelmarsh_1",
        help="Turbine name, e.g. Kelmarsh_1. Defaults to Kelmarsh_1.",
    )
    args = parser.parse_args()

    main(turbine_name=args.turbine)
