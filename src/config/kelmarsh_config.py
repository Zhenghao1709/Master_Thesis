from __future__ import annotations

TIME_COL = "Date and time"
TURBINE_ID_COL = "turbine_id"

TARGET_COL = "Generator bearing front temperature (°C)"

INPUT_COLS = [
    "Wind speed (m/s)",
    "Power (kW)",
    "Nacelle ambient temperature (°C)",
    "Nacelle temperature (°C)",
    "Generator RPM (RPM)",
    "Rotor speed (RPM)",
    "Stator temperature 1 (°C)",
    "Generator bearing front temperature (°C)",   # autoregressive
]

# 数据分辨率通常是 10 分钟
FREQ_MINUTES = 10
SEQ_LEN = 12                  # 过去 2 小时
MIN_SEGMENT_LEN = 24          # 至少 4 小时连续片段
MAX_MISSING_RATIO_PER_ROW = 0.2

# 物理合理性粗筛，第一版保守即可
PHYSICAL_LIMITS = {
    "Wind speed (m/s)": (0.0, 40.0),
    "Power (kW)": (-100.0, 3000.0),
    "Generator RPM (RPM)": (0.0, 2500.0),
    "Rotor speed (RPM)": (0.0, 30.0),
    "Generator bearing front temperature (°C)": (-20.0, 150.0),
    "Stator temperature 1 (°C)": (-20.0, 180.0),
    "Nacelle ambient temperature (°C)": (-40.0, 60.0),
    "Nacelle temperature (°C)": (-40.0, 80.0),
}