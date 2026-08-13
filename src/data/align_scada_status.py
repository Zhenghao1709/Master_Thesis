# src/data/align_scada_status.py

from __future__ import annotations

import pandas as pd

from src.config.status_mapping import (
    DEFAULT_COMM_POST_DAYS,
    DEFAULT_COMM_PRE_DAYS,
)


def align_status_to_scada(
    scada_df: pd.DataFrame,
    status_df: pd.DataFrame,
    scada_time_col: str = "Date and time",
    scada_turbine_col: str = "turbine_id",
    status_turbine_col: str = "turbine_id",
    comm_pre_days: int = DEFAULT_COMM_PRE_DAYS,
    comm_post_days: int = DEFAULT_COMM_POST_DAYS,
) -> pd.DataFrame:
    """
    Align interval-based status logs to point-wise SCADA timestamps.

    Output columns added:
    - has_any_status
    - in_event
    - in_maintenance
    - in_communication
    - in_normal_ref

    Notes
    -----
    This version is safer for larger data than the previous one because:
    1. It processes one turbine at a time.
    2. It uses datetime index slicing instead of full-table boolean scan for each row.
    3. Communication intervals are expanded with a configurable day-level buffer.
    """
    if comm_pre_days < 0 or comm_post_days < 0:
        raise ValueError("comm_pre_days and comm_post_days must be non-negative.")

    scada = scada_df.copy()
    status = status_df.copy()

    if scada_time_col not in scada.columns:
        raise ValueError(f"Missing SCADA time column: {scada_time_col}")
    if "Timestamp start" not in status.columns or "Timestamp end" not in status.columns:
        raise ValueError("Status dataframe must contain 'Timestamp start' and 'Timestamp end'.")

    scada[scada_time_col] = pd.to_datetime(scada[scada_time_col], errors="coerce")
    status["Timestamp start"] = pd.to_datetime(status["Timestamp start"], errors="coerce")
    status["Timestamp end"] = pd.to_datetime(status["Timestamp end"], errors="coerce")

    scada = scada[scada[scada_time_col].notna()].copy()
    status = status[
        status["Timestamp start"].notna() &
        status["Timestamp end"].notna() &
        (status["Timestamp end"] >= status["Timestamp start"])
    ].copy()

    # initialize flags
    for col in ["has_any_status", "in_event", "in_maintenance", "in_communication", "in_normal_ref"]:
        scada[col] = False

    # if no turbine column exists in status, align everything together
    if scada_turbine_col not in scada.columns or status_turbine_col not in status.columns:
        return _align_one_group(
            scada=scada,
            status=status,
            scada_time_col=scada_time_col,
            comm_pre_days=comm_pre_days,
            comm_post_days=comm_post_days,
        )

    all_parts = []

    for turbine_id, scada_g in scada.groupby(scada_turbine_col, sort=False):
        scada_g = scada_g.sort_values(scada_time_col).copy()

        status_g = status[status[status_turbine_col] == turbine_id].copy()
        if status_g.empty:
            all_parts.append(scada_g)
            continue

        aligned_g = _align_one_group(
            scada=scada_g,
            status=status_g,
            scada_time_col=scada_time_col,
            comm_pre_days=comm_pre_days,
            comm_post_days=comm_post_days,
        )
        all_parts.append(aligned_g)

    out = pd.concat(all_parts, ignore_index=True)
    return out.sort_values([scada_turbine_col, scada_time_col]).reset_index(drop=True)


def _align_one_group(
    scada: pd.DataFrame,
    status: pd.DataFrame,
    scada_time_col: str = "Date and time",
    comm_pre_days: int = DEFAULT_COMM_PRE_DAYS,
    comm_post_days: int = DEFAULT_COMM_POST_DAYS,
) -> pd.DataFrame:
    scada = scada.copy().sort_values(scada_time_col).reset_index(drop=True)
    status = status.copy().sort_values(["Timestamp start", "Timestamp end"]).reset_index(drop=True)

    # Set datetime index for faster interval assignment
    scada = scada.set_index(scada_time_col, drop=False)

    for _, row in status.iterrows():
        start = row["Timestamp start"]
        end = row["Timestamp end"]
        bucket = row.get("status_bucket", "other")
        flag_start = start
        flag_end = end
        if bucket == "communication":
            flag_start = start - pd.DateOffset(days=comm_pre_days)
            flag_end = end + pd.DateOffset(days=comm_post_days)

        # skip obviously non-overlapping intervals
        if flag_end < scada.index.min() or flag_start > scada.index.max():
            continue

        try:
            idx_slice = scada.loc[flag_start:flag_end]
        except KeyError:
            continue

        if idx_slice.empty:
            continue

        scada.loc[start:end, "has_any_status"] = True

        if bucket == "event":
            scada.loc[flag_start:flag_end, "in_event"] = True
        elif bucket == "maintenance":
            scada.loc[flag_start:flag_end, "in_maintenance"] = True
        elif bucket == "communication":
            scada.loc[flag_start:flag_end, "in_communication"] = True
        elif bucket == "normal_ref":
            scada.loc[flag_start:flag_end, "in_normal_ref"] = True

    return scada.reset_index(drop=True)
