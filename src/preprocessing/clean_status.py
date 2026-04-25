from __future__ import annotations

import re
import pandas as pd

from src.config.status_mapping import (
    EVENT_STATUS_KEYWORDS,
    MAINTENANCE_KEYWORDS,
    COMMUNICATION_KEYWORDS,
    IEC_EVENT_CATEGORIES,
    IEC_MAINTENANCE_CATEGORIES,
    IEC_NORMAL_CATEGORIES,
)


def _normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _lower_text(x) -> str:
    return _normalize_text(x).lower()


def clean_status_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 标准化文本列
    text_cols = [
        "Status",
        "Code",
        "Message",
        "Comment",
        "Service contract category",
        "IEC category",
    ]
    for c in text_cols:
        if c in out.columns:
            out[c] = out[c].map(_normalize_text)

    # 去除无效时间记录
    out = out[out["Timestamp start"].notna()].copy()
    if "Timestamp end" in out.columns:
        out = out[out["Timestamp end"].notna()].copy()
        out = out[out["Timestamp end"] >= out["Timestamp start"]].copy()

    # duration_seconds
    if "Duration" in out.columns and pd.api.types.is_timedelta64_dtype(out["Duration"]):
        out["duration_seconds"] = out["Duration"].dt.total_seconds()
    else:
        out["duration_seconds"] = (out["Timestamp end"] - out["Timestamp start"]).dt.total_seconds()

    # 统一小写辅助列
    out["status_lower"] = out["Status"].map(_lower_text) if "Status" in out.columns else ""
    out["message_lower"] = out["Message"].map(_lower_text) if "Message" in out.columns else ""
    out["iec_lower"] = out["IEC category"].map(_lower_text) if "IEC category" in out.columns else ""

    # 粗分类
    out["is_event_status"] = out["status_lower"].apply(
        lambda s: any(k in s for k in [x.lower() for x in EVENT_STATUS_KEYWORDS])
    )

    out["is_maintenance_status"] = (
        out["status_lower"].apply(lambda s: any(k in s for k in [x.lower() for x in MAINTENANCE_KEYWORDS]))
        | out["message_lower"].apply(lambda s: any(k in s for k in [x.lower() for x in MAINTENANCE_KEYWORDS]))
        | out["IEC category"].isin(IEC_MAINTENANCE_CATEGORIES)
    )

    out["is_comm_status"] = (
        out["status_lower"].apply(lambda s: any(k in s for k in [x.lower() for x in COMMUNICATION_KEYWORDS]))
        | out["message_lower"].apply(lambda s: any(k in s for k in [x.lower() for x in COMMUNICATION_KEYWORDS]))
    )

    out["is_forced_outage"] = out["IEC category"].isin(IEC_EVENT_CATEGORIES)
    out["is_full_performance"] = out["IEC category"].isin(IEC_NORMAL_CATEGORIES)

    # 统一大类标签
    def map_status_bucket(row) -> str:
        if row["is_comm_status"]:
            return "communication"
        if row["is_forced_outage"] or row["is_event_status"]:
            return "event"
        if row["is_maintenance_status"]:
            return "maintenance"
        if row["is_full_performance"]:
            return "normal_ref"
        return "other"

    out["status_bucket"] = out.apply(map_status_bucket, axis=1)

    return out.sort_values(["Timestamp start", "Timestamp end"]).reset_index(drop=True)