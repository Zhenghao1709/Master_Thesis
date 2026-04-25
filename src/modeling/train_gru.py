# src/modeling/train_gru.py

from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.modeling.dataset import create_sequences, SequenceDataset
from src.modeling.model_gru import GRUNBM


def train_gru_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    input_cols: list[str],
    target_col: str,
    seq_len: int,
    batch_size: int = 64,
    hidden_size: int = 64,
    num_layers: int = 1,
    lr: float = 1e-3,
    epochs: int = 20,
    model_save_path: str | Path | None = None,
    log_save_path: str | Path | None = None,
):
    # 1. 构造序列
    X_train, y_train = create_sequences(
        train_df,
        input_cols=input_cols,
        target_col=target_col,
        seq_len=seq_len,
    )

    X_val, y_val = create_sequences(
        val_df,
        input_cols=input_cols,
        target_col=target_col,
        seq_len=seq_len,
    )

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)

    # 2. Dataset / DataLoader
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUNBM(
        input_size=len(input_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logs = []
    best_val_loss = float("inf")

    # 4. 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss_sum += loss.item() * xb.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)

        logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if model_save_path is not None:
                model_save_path = Path(model_save_path)
                model_save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_save_path)

    log_df = pd.DataFrame(logs)

    if log_save_path is not None:
        log_save_path = Path(log_save_path)
        log_save_path.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(log_save_path, index=False)

    return model, log_df