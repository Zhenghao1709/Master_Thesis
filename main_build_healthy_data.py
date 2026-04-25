# main_build_healthy_data.py

from __future__ import annotations

from pathlib import Path

from src.config.kelmarsh_config import TIME_COL, TARGET_COL, INPUT_COLS, TURBINE_ID_COL
from src.data.read_scada import read_scada_folder
from src.data.read_status import read_status_folder
from src.data.merge_years import deduplicate_scada, deduplicate_status, merge_and_save
from src.data.align_scada_status import align_status_to_scada
from src.preprocessing.clean_scada import clean_scada_df
from src.preprocessing.clean_status import clean_status_df
from src.preprocessing.heuristics import build_all_heuristic_flags
from src.preprocessing.build_healthy_candidates import extract_healthy_candidates
from src.preprocessing.segment_builder import split_into_healthy_segments


def main():
    project_root = Path(__file__).resolve().parent

    turbine_name = "Kelmarsh_1"

    scada_folder = project_root / "data" / "raw" / "kelmarsh" / "scada" / turbine_name
    status_folder = project_root / "data" / "raw" / "kelmarsh" / "status" / turbine_name

    interim_root = project_root / "data" / "interim" / "kelmarsh"
    processed_root = project_root / "data" / "processed" / "kelmarsh"

    merged_scada_path = interim_root / "merged_scada" / f"{turbine_name.lower()}_scada_2016_2024.parquet"
    merged_status_path = interim_root / "merged_status" / f"{turbine_name.lower()}_status_2016_2024.parquet"
    aligned_path = interim_root / "aligned" / f"{turbine_name.lower()}_scada_with_status.parquet"
    flags_path = interim_root / "flags" / f"{turbine_name.lower()}_with_flags.parquet"

    healthy_candidates_path = processed_root / "healthy_candidates" / f"{turbine_name.lower()}_healthy_candidates.parquet"
    healthy_segments_path = processed_root / "healthy_segments" / f"{turbine_name.lower()}_healthy_segments.parquet"

    # 1. 读多年度 SCADA 和 Status
    scada_df = read_scada_folder(scada_folder, turbine_id=turbine_name)
    status_df = read_status_folder(status_folder, turbine_id=turbine_name)

    # 2. 清洗 + 去重
    scada_df = clean_scada_df(scada_df)
    scada_df = deduplicate_scada(scada_df)

    status_df = clean_status_df(status_df)
    status_df = deduplicate_status(status_df)

    # 保存 merged
    merge_and_save(scada_df, merged_scada_path)
    merge_and_save(status_df, merged_status_path)

    # 3. 对齐 status 到 SCADA 时间轴
    aligned_df = align_status_to_scada(scada_df, status_df, scada_time_col=TIME_COL)
    merge_and_save(aligned_df, aligned_path)

    # 4. 规则打标
    flagged_df = build_all_heuristic_flags(aligned_df)

    print(flagged_df[[
        "is_good_quality",
        "is_physically_valid",
        "is_normal_operating_condition",
        "is_dirty",
        "is_event_like",
        "is_healthy_candidate"
    ]].mean())


    merge_and_save(flagged_df, flags_path)

    # 5. 提取 healthy candidates
    cols_to_keep = [TIME_COL, TURBINE_ID_COL] + INPUT_COLS + [TARGET_COL]
    cols_to_keep = list(dict.fromkeys(cols_to_keep))
    healthy_candidates = extract_healthy_candidates(flagged_df, cols_to_keep)
    merge_and_save(healthy_candidates, healthy_candidates_path)

    # 6. 切连续健康片段
    healthy_segments = split_into_healthy_segments(
        flagged_df,
        time_col=TIME_COL,
        id_col=TURBINE_ID_COL,
        health_col="is_healthy_candidate",
    )
    merge_and_save(healthy_segments, healthy_segments_path)

    print("Done.")
    print("SCADA merged:", merged_scada_path)
    print("Status merged:", merged_status_path)
    print("Aligned:", aligned_path)
    print("Flags:", flags_path)
    print("Healthy candidates:", healthy_candidates_path)
    print("Healthy segments:", healthy_segments_path)
    print("Healthy candidate rows:", len(healthy_candidates))
    print("Healthy segment rows:", len(healthy_segments))


if __name__ == "__main__":
    main()