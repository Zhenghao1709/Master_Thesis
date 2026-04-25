# src/modeling/dataset.py

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def create_sequences(
    df,
    input_cols,
    target_col,
    seq_len=12,
    id_cols=("turbine_id", "segment_id"),
    time_col="Date and time",
):
    X, y = [], []

    group_cols = [c for c in id_cols if c in df.columns]

    for _, g in df.groupby(group_cols):
        g = g.sort_values(time_col)

        x_values = g[input_cols].values
        y_values = g[target_col].values

        if len(g) <= seq_len:
            continue

        for i in range(len(g) - seq_len):
            X.append(x_values[i:i + seq_len])
            y.append(y_values[i + seq_len])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]