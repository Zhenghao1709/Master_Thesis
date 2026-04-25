# src/modeling/model_gru.py

from __future__ import annotations

import torch.nn as nn


class GRUNBM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)       # out: [B, T, H]
        last_hidden = out[:, -1, :]  # 取最后一个时间步
        pred = self.fc(last_hidden)  # [B, 1]
        return pred