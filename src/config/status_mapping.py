from __future__ import annotations

EVENT_STATUS_KEYWORDS = [
    "stop",
    "warning",
    "fault",
    "outage",
    "trip",
    "emergency",
    "shutdown",
]

MAINTENANCE_KEYWORDS = [
    "maintenance",
    "service",
    "manual stop",
    "inspection",
]

COMMUNICATION_KEYWORDS = [
    "communication",
    "comms",
    "signal lost",
    "data link",
    "controller communication",
]

# IEC category 级别的粗映射
IEC_EVENT_CATEGORIES = {
    "Forced outage",
    "Forced Outage",
    "External conditions",
}

IEC_MAINTENANCE_CATEGORIES = {
    "Scheduled Maintenance",
    "Scheduled maintenance",
    "Technical Standby",
}

IEC_NORMAL_CATEGORIES = {
    "Full Performance",
    "Full performance",
}

# buffer 设置，后面 heuristics 会用到
DEFAULT_EVENT_PRE_DAYS = 14
DEFAULT_EVENT_POST_DAYS = 3

DEFAULT_MAINTENANCE_PRE_DAYS = 1
DEFAULT_MAINTENANCE_POST_DAYS = 2

DEFAULT_COMM_PRE_HOURS = 1
DEFAULT_COMM_POST_HOURS = 1