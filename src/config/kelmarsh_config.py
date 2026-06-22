from __future__ import annotations

TIME_COL = "Date and time"
TURBINE_ID_COL = "turbine_id"

TARGET_COL = "Generator bearing front temperature (°C)"

TARGET_COLS = [
    "Generator bearing front temperature (°C)",
    "Stator temperature 1 (°C)",
    "Rear bearing temperature (°C)",
]

# Broad candidate set used for preprocessing, quality filtering, physical
# checks, and healthy segment construction. Final model input selection can be
# narrower and should happen in the training scripts.
PREPROCESSING_SIGNAL_COLS = [
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

# Per-signal scaling selected from the distribution diagnostics in
# signal_viewer.ipynb. The three rotational-speed signals intentionally
# override the notebook's automatic suggestion and use MinMaxScaler.
SCALER_TYPE_BY_SIGNAL = {
    PREPROCESSING_SIGNAL_COLS[0]: "robust",    # Wind speed
    PREPROCESSING_SIGNAL_COLS[1]: "standard",  # Power
    PREPROCESSING_SIGNAL_COLS[2]: "standard",  # Nacelle ambient temperature
    PREPROCESSING_SIGNAL_COLS[3]: "standard",  # Nacelle temperature
    PREPROCESSING_SIGNAL_COLS[4]: "minmax",   # Generator RPM
    PREPROCESSING_SIGNAL_COLS[5]: "minmax",   # Rotor speed
    PREPROCESSING_SIGNAL_COLS[6]: "minmax",   # Gearbox speed
    PREPROCESSING_SIGNAL_COLS[7]: "standard",  # Converter ambient temperature
    PREPROCESSING_SIGNAL_COLS[8]: "robust",    # Stator temperature 1
    PREPROCESSING_SIGNAL_COLS[9]: "standard",  # Generator bearing front temperature
    PREPROCESSING_SIGNAL_COLS[10]: "robust",   # Rear bearing temperature
}

# Baseline training input set. Highly correlated alternatives such as
# Rotor speed/Gearbox speed and converter ambient temperature remain available
# in PREPROCESSING_SIGNAL_COLS for later ablation experiments.
INPUT_COLS = [
    "Wind speed (m/s)",
    "Power (kW)",
    "Nacelle ambient temperature (°C)",
    "Nacelle temperature (°C)",
    "Generator RPM (RPM)",
    "Stator temperature 1 (°C)",  # target history
    "Generator bearing front temperature (°C)",  # target history
    "Rear bearing temperature (°C)",  # target history
]

# SCADA resolution is usually 10 minutes.
FREQ_MINUTES = 10
SEQ_LEN = 12
MIN_SEGMENT_LEN = 24
MAX_MISSING_RATIO_PER_ROW = 0.2

# Conservative first-pass validity limits used by heuristic filtering.
PHYSICAL_LIMITS = {
    "Wind speed (m/s)": (3.0, 40.0),
    "Power (kW)": (50.0, 3000.0),
    "Generator RPM (RPM)": (100.0, 2500.0),
    "Rotor speed (RPM)": (0.0, 30.0),
    "Generator bearing front temperature (°C)": (-20.0, 150.0),
    "Stator temperature 1 (°C)": (-20.0, 180.0),
    "Nacelle ambient temperature (°C)": (-40.0, 60.0),
    "Nacelle temperature (°C)": (-40.0, 80.0),
    "Rear bearing temperature (°C)": (-20.0, 150.0),
    "Ambient temperature (converter) (°C)": (-40.0, 100.0),
    "Gearbox speed (RPM)": (0.0, 3000.0),
}
