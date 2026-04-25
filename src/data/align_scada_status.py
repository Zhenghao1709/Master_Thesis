# src/data/align_scada_status.py

from __future__ import annotations

import pandas as pd


def align_status_to_scada(
    scada_df: pd.DataFrame,
    status_df: pd.DataFrame,
    scada_time_col: str = "Date and time",
) -> pd.DataFrame:
    """
    对每个 SCADA 时间点，标记它是否落在某类 status 区间中。
    输出列：
    - has_any_status
    - in_event
    - in_maintenance
    - in_communication
    - in_normal_ref
    """
    scada = scada_df.copy().sort_values(scada_time_col).reset_index(drop=True)
    status = status_df.copy().sort_values(["Timestamp start", "Timestamp end"]).reset_index(drop=True)

    scada["has_any_status"] = False
    scada["in_event"] = False
    scada["in_maintenance"] = False
    scada["in_communication"] = False
    scada["in_normal_ref"] = False

    for _, row in status.iterrows():
        start = row["Timestamp start"]
        end = row["Timestamp end"]
        bucket = row.get("status_bucket", "other")

        mask = (scada[scada_time_col] >= start) & (scada[scada_time_col] <= end)
        if not mask.any():
            continue

        scada.loc[mask, "has_any_status"] = True

        if bucket == "event":
            scada.loc[mask, "in_event"] = True
        elif bucket == "maintenance":
            scada.loc[mask, "in_maintenance"] = True
        elif bucket == "communication":
            scada.loc[mask, "in_communication"] = True
        elif bucket == "normal_ref":
            scada.loc[mask, "in_normal_ref"] = True

    return scada