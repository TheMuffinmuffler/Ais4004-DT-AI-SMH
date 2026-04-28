
import os
import io
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Mock streamlit for the training script to work
import sys
from unittest.mock import MagicMock
sys.modules["streamlit"] = MagicMock()

import Train_SHM_Neural_Network as trainer
import plot_utils
import shm_utils

def run_custom_experiment():
    # Load data
    df = pd.read_csv("Test_Data_1.csv")
    df.columns = df.columns.str.strip()
    
    input_cols = ["accel_x", "accel_y", "accel_z", "ang_rate_x", "ang_rate_y", "ang_rate_z"]
    target_cols = ["gauge_1", "gauge_2", "gauge_3", "gauge_4", "gauge_5", "gauge_6"]
    
    # Define configurations
    configs = [
        {"win": 110, "stride": 2, "batch": 16, "epochs": 100, "lr": 0.0001, "label": "WL110_S2_B16_E100_LR0.0001"},
        {"win": 125, "stride": 2, "batch": 24, "epochs": 100, "lr": 0.0005, "label": "WL125_S2_B24_E100_LR0.0005"},
        {"win": 75, "stride": 1, "batch": 16, "epochs": 50, "lr": 0.0005, "label": "WL75_S1_B16_E50_LR0.0005"}
    ]
    
    all_histories = {}
    all_predictions = {col: {} for col in target_cols}
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"custom_batch_comparison_{timestamp}"
    
    status_box = MagicMock()
    
    for cfg_dict in configs:
        label = cfg_dict["label"]
        print(f"Running experiment: {label}")
        
        cfg = trainer.TrainConfig(
            win=cfg_dict["win"],
            stride=cfg_dict["stride"],
            batch=cfg_dict["batch"],
            epochs=cfg_dict["epochs"],
            lr=cfg_dict["lr"],
            model_filename=f"{label}.pt"
        )
        
        # 1. Train
        model_bytes, hist_df = trainer.train_model(df, input_cols, target_cols, cfg, status_box)
        all_histories[label] = hist_df
        
        # 2. Inference
        print(f"Running inference for: {label}")
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
        
        # Store predictions
        for col in target_cols:
            all_predictions[col][label] = out_df[f"pred_{col}"].values
            
    # 3. Save Consolidated History Plots
    print("Generating history plots...")
    # We need to adapt the label to the plot_utils function which expects param_to_vary as a key
    # but here we vary many things, so we'll just pass 'config' as the param name
    train_plot, test_plot = plot_utils.auto_save_batch_training_histories(
        all_histories, "config", experiment_name
    )
    print(f"Saved history plots: {train_plot}, {test_plot}")
    
    # 4. Generate Multi-Comparison Plots for each gauge
    print("Generating gauge comparison plots...")
    time_col_name = "time" if "time" in df.columns else "row_index"
    for col in target_cols:
        fig_path = plot_utils.auto_save_multi_comparison_plot(
            out_dfs=all_predictions[col],
            output_name=col,
            x_col=time_col_name,
            measured_df=df,
            run_params_label="config",
            experiment_name=experiment_name
        )
        print(f"Saved plot for {col}: {fig_path}")

if __name__ == "__main__":
    run_custom_experiment()
