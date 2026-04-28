
import os
import io
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

import shm_utils
import plot_utils

# --- Copied Logic from Train_SHM_Neural_Network.py to ensure standalone execution ---

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, win: int = 100, stride: int = 10):
        self.X = X
        self.y = y
        self.win = win
        self.stride = stride
        self.idxs = np.arange(0, len(X) - win, stride, dtype=np.int64)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        s = self.idxs[i]
        e = s + self.win
        return self.X[s:e], self.y[e - 1]

@dataclass
class TrainConfig:
    win: int = 100
    stride: int = 10
    batch: int = 512
    epochs: int = 10
    lr: float = 1e-3
    seed: int = 2026
    test_fraction: float = 0.15
    scaler_q_low: int = 10
    scaler_q_high: int = 90

def train_model(df: pd.DataFrame, input_cols: List[str], target_cols: List[str], cfg: TrainConfig) -> Tuple[bytes, pd.DataFrame]:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    X_raw = df[input_cols].to_numpy(np.float32)
    y_raw = df[target_cols].to_numpy(np.float32)

    split_idx = int(len(X_raw) * (1.0 - cfg.test_fraction))
    X_train, X_test = X_raw[:split_idx], X_raw[split_idx:]
    y_train, y_test = y_raw[:split_idx], y_raw[split_idx:]

    x_scaler = RobustScaler(quantile_range=(cfg.scaler_q_low, cfg.scaler_q_high))
    y_scaler = RobustScaler(quantile_range=(cfg.scaler_q_low, cfg.scaler_q_high))

    X_train_s = x_scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = x_scaler.transform(X_test).astype(np.float32)
    y_train_s = y_scaler.fit_transform(y_train).astype(np.float32)
    y_test_s = y_scaler.transform(y_test).astype(np.float32)

    train_ds = WindowDataset(X_train_s, y_train_s, win=cfg.win, stride=cfg.stride)
    test_ds = WindowDataset(X_test_s, y_test_s, win=cfg.win, stride=cfg.stride)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = shm_utils.CNN1D(in_ch=len(input_cols), out_dim=len(target_cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=1.0)

    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item() * xb.size(0)
        
        train_loss = running_loss / len(train_ds)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                test_loss += loss_fn(pred, yb).item() * xb.size(0)
        test_loss /= len(test_ds)
        
        history.append({"epoch": epoch, "train_huber_scaled": train_loss, "test_huber_scaled": test_loss})
        if epoch % 10 == 0 or epoch == cfg.epochs:
            print(f"  Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "x_center": x_scaler.center_, "x_scale": x_scaler.scale_,
        "y_center": y_scaler.center_, "y_scale": y_scaler.scale_,
        "input_cols": input_cols, "target_cols": target_cols,
        "win": cfg.win, "stride": cfg.stride, "batch": cfg.batch, "epochs": cfg.epochs, "lr": cfg.lr
    }
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    return buffer.getvalue(), pd.DataFrame(history)

# --- Experiment Execution ---

def main():
    df = pd.read_csv("Test_Data_1.csv")
    df.columns = df.columns.str.strip()
    input_cols = ["accel_x", "accel_y", "accel_z", "ang_rate_x", "ang_rate_y", "ang_rate_z"]
    target_cols = ["gauge_1", "gauge_2", "gauge_3", "gauge_4", "gauge_5", "gauge_6"]

    configs = [
        {"win": 110, "stride": 2, "batch": 16, "epochs": 100, "lr": 0.0001, "name": "WL110_S2_B16_E100_LR0.0001"},
        {"win": 125, "stride": 2, "batch": 24, "epochs": 100, "lr": 0.0005, "name": "WL125_S2_B24_E100_LR0.0005"},
        {"win": 75, "stride": 1, "batch": 16, "epochs": 50, "lr": 0.0005, "name": "WL75_S1_B16_E50_LR0.0005"}
    ]

    all_histories = {}
    all_predictions = {col: {} for col in target_cols}
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"custom_comparison_{timestamp}"

    for c in configs:
        print(f"\n>>> Running: {c['name']}")
        cfg = TrainConfig(win=c['win'], stride=c['stride'], batch=c['batch'], epochs=c['epochs'], lr=c['lr'])
        model_bytes, hist_df = train_model(df, input_cols, target_cols, cfg)
        
        all_histories[c['name']] = hist_df
        
        # Inference
        ckpt = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=False)
        estimator = shm_utils.GenericTimeSeriesEstimator(ckpt)
        out_df = shm_utils.build_output_dataframe(df, estimator, time_col="time", include_inputs=False, include_measured_outputs=True, quiet=True)
        
        for col in target_cols:
            all_predictions[col][c['name']] = out_df[f"pred_{col}"].values

    # Plotting
    print("\n>>> Generating Comparison Plots...")
    plot_utils.auto_save_batch_training_histories(all_histories, "config", experiment_name)
    
    time_col = "time" if "time" in df.columns else "row_index"
    for col in target_cols:
        plot_utils.auto_save_multi_comparison_plot(all_predictions[col], col, time_col, df, "config", experiment_name)
    
    print(f"\nDone! Results saved in plots/{experiment_name}")

if __name__ == "__main__":
    main()
