import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
import streamlit as st
import io
import matplotlib.pyplot as plt
import re

def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be used as a valid folder/file name."""
    return re.sub(r'[\\/*?:"<>|]', "_", name).replace(" ", "_")

def auto_save_training_history(hist_df: pd.DataFrame, model_name: str, run_params: str, base_dir: str = "plots"):
    """
    Automatically saves the training history as a PNG using Matplotlib in a parameter-specific folder.
    """
    run_folder = sanitize_filename(run_params)
    out_path = Path(base_dir) / run_folder
    out_path.mkdir(parents=True, exist_ok=True)
    
    file_path = out_path / f"training_history_{model_name}.png"
    
    plt.figure(figsize=(10, 6))
    if "train_huber_scaled" in hist_df.columns:
        plt.plot(hist_df["epoch"], hist_df["train_huber_scaled"], label="Train Loss")
    if "test_huber_scaled" in hist_df.columns:
        plt.plot(hist_df["epoch"], hist_df["test_huber_scaled"], label="Test Loss")
    
    plt.title(f"Training History: {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss (scaled)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(file_path)
    plt.close()
    return file_path

def auto_save_comparison_plot(out_df: pd.DataFrame, output_name: str, x_col: str, model_name: str, run_params: str, base_dir: str = "plots"):
    """
    Automatically saves a comparison plot as a PNG in a parameter-specific folder.
    """
    run_folder = sanitize_filename(run_params)
    out_path = Path(base_dir) / run_folder
    out_path.mkdir(parents=True, exist_ok=True)
    
    file_path = out_path / f"comparison_{output_name}.png"
    
    plt.figure(figsize=(12, 6))
    
    measured_col = f"measured_{output_name}"
    pred_col = f"pred_{output_name}"
    
    if measured_col in out_df.columns:
        plt.plot(out_df[x_col], out_df[measured_col], label=f"Measured {output_name}", alpha=0.7)
    
    if pred_col in out_df.columns:
        plt.plot(out_df[x_col], out_df[pred_col], label=f"Predicted {output_name}", alpha=0.8, linestyle="--")
    
    plt.title(f"Measured vs Predicted: {output_name}\n({run_params})")
    plt.xlabel(x_col)
    plt.ylabel(output_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    return file_path

def auto_save_multi_comparison_plot(
    out_dfs: dict, 
    output_name: str, 
    x_col: str, 
    measured_df: pd.DataFrame, 
    run_params_label: str,
    experiment_name: str,
    base_dir: str = "plots"
):
    """
    Saves a plot comparing multiple model runs for a single gauge.
    out_dfs: dict of {label: pred_series}
    """
    out_path = Path(base_dir) / sanitize_filename(experiment_name)
    out_path.mkdir(parents=True, exist_ok=True)
    
    file_path = out_path / f"multi_comparison_{output_name}.png"
    
    plt.figure(figsize=(15, 7))
    
    measured_col = output_name # Assuming measured_df has the raw names
    if measured_col in measured_df.columns:
        plt.plot(measured_df[x_col], measured_df[measured_col], label=f"Measured {output_name}", color='black', linewidth=2, alpha=0.6)
    
    for label, pred_series in out_dfs.items():
        plt.plot(measured_df[x_col], pred_series, label=f"Pred ({run_params_label}={label})", alpha=0.8, linestyle="--")
    
    plt.title(f"Multi-Model Comparison: {output_name}\nVarying {run_params_label}")
    plt.xlabel(x_col)
    plt.ylabel(output_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    return file_path

def create_training_history_plot(hist_df: pd.DataFrame) -> go.Figure:
    """Creates a Plotly figure for training history visualization."""
    fig = go.Figure()
    if "train_huber_scaled" in hist_df.columns:
        fig.add_trace(go.Scatter(x=hist_df["epoch"], y=hist_df["train_huber_scaled"], name="Train Loss"))
    if "test_huber_scaled" in hist_df.columns:
        fig.add_trace(go.Scatter(x=hist_df["epoch"], y=hist_df["test_huber_scaled"], name="Test Loss"))
    fig.update_layout(
        title="Training History",
        xaxis_title="Epoch",
        yaxis_title="Huber Loss (scaled)",
        template="plotly_white"
    )
    return fig
