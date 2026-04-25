# src/features/scaling.py

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_scalers(
    train_df: pd.DataFrame,
    input_cols: list[str],
    target_col: str,
) -> tuple[StandardScaler, StandardScaler]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(train_df[input_cols])
    y_scaler.fit(train_df[[target_col]])

    return x_scaler, y_scaler


def apply_scalers(
    df: pd.DataFrame,
    input_cols: list[str],
    target_col: str,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
) -> pd.DataFrame:
    out = df.copy()

    out[input_cols] = x_scaler.transform(out[input_cols])
    out[[target_col]] = y_scaler.transform(out[[target_col]])

    return out


def save_scaler(scaler, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: str | Path):
    return joblib.load(path)