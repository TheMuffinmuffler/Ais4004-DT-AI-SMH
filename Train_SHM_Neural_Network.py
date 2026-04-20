"""
Streamlit GUI for training a 1D CNN neural network from CSV data.

Changes from the previous version:
- no input/output columns are selected by default
- supports either:
  1) CSV upload, or
  2) loading from a local/server CSV path
- if a CSV path is used, the trained .pt file can be saved automatically
  in the same folder as the CSV file
- if a CSV is uploaded through the browser, Streamlit cannot access the
  user's local folder path, so the model is offered as a download instead

Run:
    streamlit run Train_SHM_Neural_Network.py

Requirements:
    pip install streamlit torch pandas numpy scikit-learn
"""

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

import plot_utils
import shm_utils


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
    model_filename: str = "trained_timeseries_model.pt"


def train_model(
    df: pd.DataFrame,
    input_cols: List[str],
    target_cols: List[str],
    cfg: TrainConfig,
    status_box,
) -> Tuple[bytes, pd.DataFrame]:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    X_raw = df[input_cols].to_numpy(np.float32)
    y_raw = df[target_cols].to_numpy(np.float32)
    X = X_raw.copy()

    split_idx = int(len(X) * (1.0 - cfg.test_fraction))
    split_idx = max(cfg.win + 1, min(split_idx, len(X) - cfg.win - 1))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_raw[:split_idx], y_raw[split_idx:]

    x_scaler = RobustScaler(quantile_range=(cfg.scaler_q_low, cfg.scaler_q_high))
    y_scaler = RobustScaler(quantile_range=(cfg.scaler_q_low, cfg.scaler_q_high))

    X_train_s = x_scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = x_scaler.transform(X_test).astype(np.float32)
    y_train_s = y_scaler.fit_transform(y_train).astype(np.float32)
    y_test_s = y_scaler.transform(y_test).astype(np.float32)

    train_ds = WindowDataset(X_train_s, y_train_s, win=cfg.win, stride=cfg.stride)
    test_ds = WindowDataset(X_test_s, y_test_s, win=cfg.win, stride=cfg.stride)

    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError(
            "Not enough rows for the selected window length and test split. "
            "Reduce the window length or use more data."
        )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch, shuffle=False, drop_last=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = shm_utils.CNN1D(in_ch=len(input_cols), out_dim=len(target_cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=1.0)

    history = []
    progress = st.progress(0.0)

    def eval_loss():
        model.eval()
        tot = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb).item()
                tot += loss * xb.size(0)
                n += xb.size(0)
        return tot / max(n, 1)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item() * xb.size(0)
            seen += xb.size(0)

        train_loss = running / max(seen, 1)
        test_loss = eval_loss()
        history.append(
            {
                "epoch": epoch,
                "train_huber_scaled": train_loss,
                "test_huber_scaled": test_loss,
            }
        )
        status_box.text(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"inputs: {len(input_cols)} | outputs: {len(target_cols)} | "
            f"train Huber (scaled): {train_loss:.6f} | "
            f"test Huber (scaled): {test_loss:.6f}"
        )
        progress.progress(epoch / cfg.epochs)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "x_center": x_scaler.center_,
        "x_scale": x_scaler.scale_,
        "y_center": y_scaler.center_,
        "y_scale": y_scaler.scale_,
        "input_cols": input_cols,
        "target_cols": target_cols,
        "win": cfg.win,
        "stride": cfg.stride,
        "batch": cfg.batch,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "fs": 100.0,
        "preprocess": {
            "input_handling": "raw_csv_values",
            "scaler": f"RobustScaler({cfg.scaler_q_low}-{cfg.scaler_q_high})",
        },
    }

    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)

    hist_df = pd.DataFrame(history)
    return buffer.getvalue(), hist_df


def load_dataframe(csv_path_text: str, uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[Path], str]:
    if csv_path_text.strip():
        path = Path(csv_path_text.strip())
        if not path.exists():
            raise FileNotFoundError(f"CSV path does not exist: {path}")
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df, path, f"Loaded from path: {path}"
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        return df, None, f"Uploaded file: {uploaded_file.name}"
    return None, None, ""


st.set_page_config(page_title="Train Neural Network", layout="wide")
st.title("Time-Series Neural Network Trainer")
st.write(
    "Upload a CSV file or provide a local/server CSV path, then select any numeric "
    "columns as inputs and outputs and train a 1D CNN model."
)

uploaded_file = st.file_uploader("Upload training CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    csv_path = None
    st.caption(f"Uploaded file: {uploaded_file.name}")
else:
    df = None

if df is not None:


    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    st.subheader("Preview")
    st.dataframe(df.head(20), width="stretch")

    if non_numeric_cols:
        st.info(f"Non-numeric columns excluded from training selection: {non_numeric_cols}")

    st.sidebar.header("Column Selection")

    input_cols = st.sidebar.multiselect(
        "Select input columns",
        numeric_cols,
        default=[],
        help="Choose any numeric columns to use as model inputs.",
    )

    remaining_for_targets = [c for c in numeric_cols if c not in input_cols]
    target_cols = st.sidebar.multiselect(
        "Select output columns",
        remaining_for_targets,
        default=[],
        help="Choose any numeric columns to predict.",
    )

    st.sidebar.header("Training Parameters")
    win = st.sidebar.number_input("Window length (samples)", min_value=2, max_value=5000, value=100, step=1)
    stride = st.sidebar.number_input("Stride (samples)", min_value=1, max_value=1000, value=10, step=1)
    batch = st.sidebar.number_input("Batch size", min_value=1, max_value=8192, value=512, step=1)
    epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=1000, value=10, step=1)
    lr = st.sidebar.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=1e-3, format="%.6f")
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000_000, value=2026, step=1)
    test_fraction = st.sidebar.slider("Test fraction", min_value=0.05, max_value=0.40, value=0.15, step=0.01)
    scaler_q_low, scaler_q_high = st.sidebar.slider(
        "RobustScaler quantiles",
        min_value=0,
        max_value=100,
        value=(10, 90),
        step=1,
    )

    default_model_name = "trained_timeseries_model.pt"
    if csv_path is not None:
        default_model_name = csv_path.stem + ".pt"
    elif uploaded_file is not None:
        default_model_name = Path(uploaded_file.name).stem + ".pt"

    model_filename = st.sidebar.text_input("Output model filename", value=default_model_name)

    st.subheader("Current Selection")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.metric("Number of inputs", len(input_cols))
    with c2:
        st.metric("Number of outputs", len(target_cols))
    with c3:
        st.metric("Numeric columns in CSV", len(numeric_cols))

    a, b = st.columns(2)
    with a:
        st.write("**Inputs**")
        st.write(input_cols)
    with b:
        st.write("**Outputs**")
        st.write(target_cols)

    st.info("The trained model will be available for download after training.")

    valid = True
    problems = []

    if len(input_cols) < 1:
        valid = False
        problems.append("Select at least one input column.")
    if len(target_cols) < 1:
        valid = False
        problems.append("Select at least one output column.")
    if set(input_cols) & set(target_cols):
        valid = False
        problems.append("Input and output columns must not overlap.")
    if len(df) <= int(win):
        valid = False
        problems.append("CSV has too few rows for the chosen window length.")

    if not valid:
        for msg in problems:
            st.error(msg)

    cfg = TrainConfig(
        win=int(win),
        stride=int(stride),
        batch=int(batch),
        epochs=int(epochs),
        lr=float(lr),
        seed=int(seed),
        test_fraction=float(test_fraction),
        scaler_q_low=int(scaler_q_low),
        scaler_q_high=int(scaler_q_high),
        model_filename=model_filename,
    )

    if st.button("Train Model", type="primary", disabled=not valid):
        status_box = st.empty()
        try:
            model_bytes, hist_df = train_model(df, input_cols, target_cols, cfg, status_box)
            st.success("Training finished.")

            st.subheader("Training History")
            st.dataframe(hist_df, width="stretch")
            
            fig = plot_utils.create_training_history_plot(hist_df)
            st.plotly_chart(fig, width="stretch")

            # Create a descriptive parameter string for the folder name
            run_params = f"win{cfg.win}_st{cfg.stride}_bs{cfg.batch}_Eps{cfg.epochs}_Lr{cfg.lr}"
            
            # Auto-save image using matplotlib (no Chrome needed)
            model_stem = Path(model_filename).stem
            saved_path = plot_utils.auto_save_training_history(hist_df, model_stem, run_params)
            st.info(f"Training history image automatically saved to: {saved_path}")

            metadata = {
                "input_cols": input_cols,
                "target_cols": target_cols,
                "num_inputs": len(input_cols),
                "num_outputs": len(target_cols),
                "win": cfg.win,
                "stride": cfg.stride,
                "batch": cfg.batch,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "seed": cfg.seed,
                "test_fraction": cfg.test_fraction,
                "scaler_quantiles": [cfg.scaler_q_low, cfg.scaler_q_high],
            }

            model_path = Path(uploaded_file.name).stem + ".pt"
            model_path = Path(model_path)

            model_path.write_bytes(model_bytes)
            st.write(model_path)

            st.success(f"Saved model as: {model_path.resolve()}")


        except Exception as e:
            st.error(f"Training failed: {e}")

    st.divider()
    st.header("Batch Experiment Mode")
    st.write("Vary one parameter across multiple training runs and compare the results on the input data.")
    
    enable_batch = st.checkbox("Enable Batch Experiment")
    
    if enable_batch:
        param_to_vary = st.selectbox("Parameter to vary", ["stride", "win", "epochs", "lr", "batch"])
        values_str = st.text_input("Values (comma-separated)", value="1, 2, 5, 10")
        
        try:
            if param_to_vary == "lr":
                values = [float(x.strip()) for x in values_str.split(",")]
            else:
                values = [int(x.strip()) for x in values_str.split(",")]
        except:
            st.error("Invalid values format. Use comma-separated numbers.")
            values = []

        if st.button("Run Batch Experiment", type="secondary", disabled=not valid or not values):
            batch_status = st.empty()
            batch_progress = st.progress(0.0)
            
            all_results = {} # {value: pred_series}
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            # Create a more descriptive folder name: batch_stride_1_2_5_10_2026...
            values_label = "_".join([str(v) for v in values[:5]]) # limit to first 5 values for name brevity
            experiment_name = f"batch_{param_to_vary}_{values_label}_{timestamp}"
            
            for i, val in enumerate(values):
                batch_status.text(f"Running experiment {i+1}/{len(values)}: {param_to_vary}={val}")
                
                # Update config for this run
                local_cfg = TrainConfig(
                    win=int(win),
                    stride=int(stride),
                    batch=int(batch),
                    epochs=int(epochs),
                    lr=float(lr),
                    seed=int(seed),
                    test_fraction=float(test_fraction),
                    scaler_q_low=int(scaler_q_low),
                    scaler_q_high=int(scaler_q_high),
                    model_filename=f"batch_{val}_{model_filename}"
                )
                setattr(local_cfg, param_to_vary, val)
                
                # 1. Train
                train_status = st.empty()
                model_bytes, hist_df = train_model(df, input_cols, target_cols, local_cfg, train_status)
                
                # Save training history for this specific batch run
                batch_hist_params = f"{param_to_vary}_{val}"
                plot_utils.auto_save_training_history(
                    hist_df, 
                    f"run_{val}", 
                    batch_hist_params, 
                    base_dir=f"plots/{experiment_name}/histories"
                )
                
                # 2. Run Inference
                batch_status.text(f"Running inference for {param_to_vary}={val}...")
                ckpt = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=False)
                estimator = shm_utils.GenericTimeSeriesEstimator(ckpt)
                
                out_df = shm_utils.build_output_dataframe(
                    df=df,
                    estimator=estimator,
                    time_col="time" if "time" in df.columns else None,
                    include_inputs=False,
                    include_measured_outputs=True,
                    quiet=True
                )
                
                # Store predictions for each target column
                for col in target_cols:
                    if col not in all_results:
                        all_results[col] = {}
                    all_results[col][val] = out_df[f"pred_{col}"].values
                
                batch_progress.progress((i + 1) / len(values))
            
            batch_status.success(f"Batch experiment finished! Folder: `plots/{experiment_name}/`")
            
            # 3. Generate Multi-Comparison Plots
            st.subheader("Batch Comparison Results")
            st.info(f"All results saved in: `{Path('plots').resolve() / experiment_name}`")
            
            time_col_name = "time" if "time" in df.columns else "row_index"
            
            saved_files = []
            for col in target_cols:
                with st.expander(f"View Plot: {col}", expanded=(col == target_cols[0])):
                    fig_path = plot_utils.auto_save_multi_comparison_plot(
                        out_dfs=all_results[col],
                        output_name=col,
                        x_col=time_col_name,
                        measured_df=df,
                        run_params_label=param_to_vary,
                        experiment_name=experiment_name
                    )
                    st.image(str(fig_path))
                    saved_files.append(fig_path)
            
            st.write("### Saved Comparison Files:")
            for f in saved_files:
                st.write(f"- `{f.name}`")
            
            st.write(f"### Individual training histories saved in: `plots/{experiment_name}/histories/`")

else:
    st.info("Provide a CSV path or upload a CSV file to begin.")
