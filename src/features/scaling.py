from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.config.kelmarsh_config import SCALER_TYPE_BY_SIGNAL


class ColumnwiseScaler:
    """Fit and apply one independently configured scaler per signal column."""

    def __init__(self, scaler_types: dict[str, str]):
        self.scaler_types = dict(scaler_types)

    @staticmethod
    def _make_scaler(scaler_type: str):
        factories = {
            "standard": StandardScaler,
            "robust": RobustScaler,
            "minmax": MinMaxScaler,
        }
        try:
            return factories[scaler_type]()
        except KeyError as exc:
            raise ValueError(f"Unsupported scaler type: {scaler_type}") from exc

    def fit(self, data: pd.DataFrame) -> "ColumnwiseScaler":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("ColumnwiseScaler.fit requires a pandas DataFrame with signal names.")

        missing_config = [column for column in data.columns if column not in self.scaler_types]
        if missing_config:
            raise KeyError(f"No scaler configured for signal(s): {missing_config}")

        self.feature_names_in_ = np.asarray(data.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.scalers_ = {}
        for column in self.feature_names_in_:
            scaler = self._make_scaler(self.scaler_types[column])
            scaler.fit(data[[column]].to_numpy(dtype=float))
            self.scalers_[column] = scaler
        return self

    def _as_array(self, data) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            missing = [column for column in self.feature_names_in_ if column not in data.columns]
            if missing:
                raise KeyError(f"Missing signal column(s): {missing}")
            array = data[list(self.feature_names_in_)].to_numpy(dtype=float)
        else:
            array = np.asarray(data, dtype=float)
            if array.ndim == 1:
                array = array.reshape(-1, 1)
        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} feature(s), received {array.shape[1]}."
            )
        return array

    def transform(self, data) -> np.ndarray:
        array = self._as_array(data)
        transformed = np.empty_like(array, dtype=float)
        for index, column in enumerate(self.feature_names_in_):
            transformed[:, index] = self.scalers_[column].transform(array[:, [index]]).ravel()
        return transformed

    def inverse_transform(self, data) -> np.ndarray:
        array = self._as_array(data)
        restored = np.empty_like(array, dtype=float)
        for index, column in enumerate(self.feature_names_in_):
            restored[:, index] = self.scalers_[column].inverse_transform(array[:, [index]]).ravel()
        return restored

    def scaler_name(self, column: str) -> str:
        return type(self.scalers_[column]).__name__


def fit_scalers(
    train_df: pd.DataFrame,
    input_cols: list[str],
    target_col: str,
) -> tuple[ColumnwiseScaler, ColumnwiseScaler]:
    x_scaler = ColumnwiseScaler(SCALER_TYPE_BY_SIGNAL)
    y_scaler = ColumnwiseScaler(SCALER_TYPE_BY_SIGNAL)

    x_scaler.fit(train_df[input_cols])
    y_scaler.fit(train_df[[target_col]])
    return x_scaler, y_scaler


def apply_scalers(
    df: pd.DataFrame,
    input_cols: list[str],
    target_col: str,
    x_scaler: ColumnwiseScaler,
    y_scaler: ColumnwiseScaler,
) -> pd.DataFrame:
    out = df.copy()

    # The autoregressive target can also be an input feature. Preserve its raw
    # values so it is not transformed once as input and again as output.
    raw_target = out[[target_col]].copy()
    out[input_cols] = x_scaler.transform(out[input_cols])
    out[[target_col]] = y_scaler.transform(raw_target)
    return out


def save_scaler(scaler, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: str | Path):
    return joblib.load(path)
