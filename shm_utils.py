import io
import time
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn

class CNN1D(nn.Module):
    def __init__(self, in_ch: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        z = self.net(x)
        return self.head(z)

class GenericTimeSeriesEstimator:
    def __init__(self, ckpt: dict):
        required_keys = [
            "model_state_dict",
            "input_cols",
            "target_cols",
            "win",
            "x_center",
            "x_scale",
            "y_center",
            "y_scale",
        ]
        missing = [k for k in required_keys if k not in ckpt]
        if missing:
            raise KeyError(
                f"Checkpoint is missing required keys: {missing}."
            )

        self.input_cols: List[str] = list(ckpt["input_cols"])
        self.target_cols: List[str] = list(ckpt["target_cols"])
        self.win: int = int(ckpt["win"])
        self.stride: int = int(ckpt["stride"]) if "stride" in ckpt else 10
        self.batch: int = int(ckpt["batch"]) if "batch" in ckpt else 512
        self.epochs: int = int(ckpt["epochs"]) if "epochs" in ckpt else 10
        self.lr: float = float(ckpt["lr"]) if "lr" in ckpt else 0.001
        self.fs: Optional[float] = float(ckpt["fs"]) if "fs" in ckpt else None

        self.x_center = np.asarray(ckpt["x_center"], dtype=np.float32)
        self.x_scale = np.asarray(ckpt["x_scale"], dtype=np.float32)
        self.y_center = np.asarray(ckpt["y_center"], dtype=np.float32)
        self.y_scale = np.asarray(ckpt["y_scale"], dtype=np.float32)

        self.model = CNN1D(in_ch=len(self.input_cols), out_dim=len(self.target_cols))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.buffer = np.zeros((self.win, len(self.input_cols)), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_filled = False

    def reset(self):
        self.buffer[:] = 0.0
        self.buffer_index = 0
        self.buffer_filled = False

    def scale_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_center) / (self.x_scale + 1e-8)

    def unscale_y(self, y: np.ndarray) -> np.ndarray:
        return y * (self.y_scale + 1e-8) + self.y_center

    def update(self, sample: np.ndarray) -> Optional[np.ndarray]:
        sample = np.asarray(sample, dtype=np.float32).reshape(-1)
        if sample.shape[0] != len(self.input_cols):
            raise ValueError(f"Expected {len(self.input_cols)} inputs, got {sample.shape[0]}.")

        self.buffer[self.buffer_index] = sample
        self.buffer_index = (self.buffer_index + 1) % self.win

        if self.buffer_index == 0:
            self.buffer_filled = True

        if not self.buffer_filled:
            return None

        window = np.concatenate(
            [self.buffer[self.buffer_index:], self.buffer[:self.buffer_index]],
            axis=0,
        )

        window_scaled = self.scale_x(window)
        tensor = torch.tensor(window_scaled[None, :, :], dtype=torch.float32)

        with torch.no_grad():
            y_scaled = self.model(tensor).numpy()[0]

        return self.unscale_y(y_scaled)

def build_output_dataframe(
    df: pd.DataFrame,
    estimator: GenericTimeSeriesEstimator,
    time_col: Optional[str],
    include_inputs: bool,
    include_measured_outputs: bool,
    progress_bar=None,
    progress_text=None,
    quiet=False
) -> pd.DataFrame:
    output_rows = []
    has_time = time_col is not None and time_col in df.columns
    has_measured_outputs = all(c in df.columns for c in estimator.target_cols)

    estimator.reset()
    total_rows = len(df)

    for row_idx, row in df.iterrows():
        sample = row[estimator.input_cols].to_numpy(dtype=np.float32)
        pred = estimator.update(sample)

        out = {"row_index": row_idx}
        if has_time:
            out[time_col] = row[time_col]
        if include_inputs:
            for col in estimator.input_cols:
                out[col] = row[col]
        if include_measured_outputs and has_measured_outputs:
            for col in estimator.target_cols:
                out[f"measured_{col}"] = row[col]

        if pred is None:
            for col in estimator.target_cols:
                out[f"pred_{col}"] = np.nan
        else:
            for col, val in zip(estimator.target_cols, pred):
                out[f"pred_{col}"] = float(val)

        output_rows.append(out)

        if not quiet and ((row_idx + 1) % 500 == 0 or (row_idx + 1) == total_rows):
            if progress_bar is not None:
                percent = int(((row_idx + 1) / total_rows) * 100)
                progress_bar.progress(percent)
            if progress_text is not None:
                progress_text.text(f"Processed {row_idx + 1} / {total_rows} rows")

    out_df = pd.DataFrame(output_rows)
    ordered_cols = []
    if has_time:
        ordered_cols.append(time_col)
    ordered_cols.append("row_index")
    if include_inputs:
        ordered_cols.extend(estimator.input_cols)
    if include_measured_outputs and has_measured_outputs:
        ordered_cols.extend([f"measured_{c}" for c in estimator.target_cols])
    ordered_cols.extend([f"pred_{c}" for c in estimator.target_cols])
    ordered_cols = [c for c in ordered_cols if c in out_df.columns]
    return out_df[ordered_cols]
