from __future__ import annotations

import pandas as pd


def add_interval_flag_to_scada(
    scada_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    flag_col: str,
    scada_time_col: str = "Date and time",
    start_col: str = "Timestamp start",
    end_col: str = "Timestamp end",
    pre_months: int = 0,
    post_months: int = 0,
) -> pd.DataFrame:
    """
    Add a boolean flag for SCADA rows that fall inside labelled time intervals.

    This is intended for manually labelled datasets such as fault events and
    curtailment periods, after those files have already been filtered to one
    turbine. Optional calendar-month offsets expand the interval before flags
    are assigned.
    """
    if pre_months < 0 or post_months < 0:
        raise ValueError("pre_months and post_months must be non-negative.")

    scada = scada_df.copy()
    intervals = intervals_df.copy()

    if scada_time_col not in scada.columns:
        raise ValueError(f"Missing SCADA time column: {scada_time_col}")

    if intervals.empty:
        scada[flag_col] = False
        return scada

    for col in [start_col, end_col]:
        if col not in intervals.columns:
            raise ValueError(f"Interval dataframe must contain '{col}'.")

    scada[scada_time_col] = pd.to_datetime(scada[scada_time_col], errors="coerce")
    intervals[start_col] = pd.to_datetime(intervals[start_col], errors="coerce")
    intervals[end_col] = pd.to_datetime(intervals[end_col], errors="coerce")

    scada = scada[scada[scada_time_col].notna()].copy()
    intervals = intervals[
        intervals[start_col].notna()
        & intervals[end_col].notna()
        & (intervals[end_col] >= intervals[start_col])
    ].copy()

    scada[flag_col] = False
    if intervals.empty or scada.empty:
        return scada

    scada = scada.sort_values(scada_time_col).set_index(scada_time_col, drop=False)
    scada_min = scada.index.min()
    scada_max = scada.index.max()

    for _, row in intervals.sort_values([start_col, end_col]).iterrows():
        start = row[start_col] - pd.DateOffset(months=pre_months)
        end = row[end_col] + pd.DateOffset(months=post_months)

        if end < scada_min or start > scada_max:
            continue

        scada.loc[start:end, flag_col] = True

    return scada.reset_index(drop=True)
