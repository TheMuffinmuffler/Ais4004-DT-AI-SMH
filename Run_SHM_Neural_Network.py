"""
Streamlit GUI for generic checkpoint-driven time-series inference.

Features
- Upload or point to a trained .pt checkpoint
- Upload an input CSV
- Reads input_cols and target_cols from the checkpoint
- Uses the checkpoint-selected output columns, if present in the CSV, as measured outputs
- Runs causal windowed inference
- Plots measured vs predicted output curves with Plotly
- Lets the user download the prediction CSV

Run:
    streamlit run Run_SHM_Neural_Network.py

Requirements:
    pip install streamlit torch pandas numpy plotly
"""

import io
import time
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from torch import nn

import plot_utils
import shm_utils

def load_checkpoint_from_uploaded(uploaded_model) -> dict:
    if uploaded_model is None:
        return None
    model_bytes = uploaded_model.read()
    bio = io.BytesIO(model_bytes)
    ckpt = torch.load(bio, map_location="cpu", weights_only=False)
    return ckpt


def load_input_csv(uploaded_csv) -> pd.DataFrame:
    if uploaded_csv is None:
        return None
    df = pd.read_csv(uploaded_csv)
    df.columns = df.columns.str.strip()
    return df


def make_comparison_plot(out_df: pd.DataFrame, output_name: str, x_col: str) -> go.Figure:
    fig = go.Figure()

    measured_col = f"measured_{output_name}"
    pred_col = f"pred_{output_name}"

    if measured_col in out_df.columns:
        fig.add_trace(
            go.Scatter(
                x=out_df[x_col],
                y=out_df[measured_col],
                mode="lines",
                name=f"measured {output_name}",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=out_df[x_col],
            y=out_df[pred_col],
            mode="lines",
            name=f"predicted {output_name}",
        )
    )

    fig.update_layout(
        title=f"Measured vs Predicted: {output_name}",
        xaxis_title=x_col,
        yaxis_title=output_name,
        legend_title="Series",
        height=500,
    )
    return fig


st.set_page_config(page_title="Execute Neural Network", layout="wide")
st.title("Time-Series Neural Network execution")
st.write(
    "Upload a trained .pt and an input CSV file used in training (must have the same structure). The app reads the required "
    "input and output columns from the input CSV, runs the Neural Network, and plots measured "
    "vs predicted strain outputs"
)

uploaded_model = st.file_uploader("Upload trained Neural Network .pt file", type=["pt"], 
                    help="Select the trained Neural Network .pt file")
uploaded_csv = st.file_uploader("Upload input CSV with MRU data", type=["csv"], 
                    help="A file identical to or having the same structure as the CSV used for training")


include_inputs              = False
include_measured_outputs    = True


time_col = st.text_input(
    "Time column name",
    value="time",
    help="If this column exists in the CSV, it will be used for plotting and preserved in the output CSV.",
)

strict_numeric = True

if uploaded_model is not None and uploaded_csv is not None:
    try:
        ckpt = load_checkpoint_from_uploaded(uploaded_model)
        estimator = shm_utils.GenericTimeSeriesEstimator(ckpt)
        df = load_input_csv(uploaded_csv)

        missing = [c for c in estimator.input_cols if c not in df.columns]
        if missing:
            st.error(
                f"Missing required input columns: {missing}\n\n"
                f"Found columns: {list(df.columns)}"
            )
            st.stop()

        if strict_numeric:
            non_numeric = [c for c in estimator.input_cols if not pd.api.types.is_numeric_dtype(df[c])]
            if non_numeric:
                st.error(f"These required input columns are not numeric: {non_numeric}")
                st.stop()

        st.subheader("Checkpoint Summary")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Inputs", len(estimator.input_cols))
        with s2:
            st.metric("Outputs", len(estimator.target_cols))
        with s3:
            st.metric("Window length", estimator.win)

        a, b = st.columns(2)
        with a:
            st.write("**Expected measured MRU input columns**")
            st.write(estimator.input_cols)
        with b:
            st.write("**AI Predicted output strain columns**")
            st.write(estimator.target_cols)

        measured_available = [c for c in estimator.target_cols if c in df.columns]
        if measured_available:
            st.success(f"Measured strain outputs found in input CSV: {measured_available}")
        else:
            st.info("No measured strain output columns were found in the CSV.")

        progress_bar = st.progress(0)
        progress_text = st.empty()
        if st.button("Run Neural Network", type="primary"):

            progress_bar = st.progress(0)
            progress_text = st.empty()

            out_df = shm_utils.build_output_dataframe(
                df=df,
                estimator=estimator,
                time_col=time_col if time_col else None,
                include_inputs=include_inputs,
                include_measured_outputs=include_measured_outputs,
                progress_bar=progress_bar,
                progress_text=progress_text,
            )

            progress_bar.progress(100)
            progress_text.text("Inference finished.")

            st.session_state["inference_output_df"] = out_df
            st.session_state["inference_target_cols"] = estimator.target_cols
            st.session_state["inference_time_col"] = time_col if (time_col and time_col in out_df.columns) else "row_index"
            st.session_state["model_name"] = Path(uploaded_model.name).stem

            # Create a descriptive parameter string for the folder name
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            params_str = f"win{estimator.win}_st{estimator.stride}_bs{estimator.batch}_Eps{estimator.epochs}_Lr{estimator.lr}"
            run_params = f"model_{st.session_state['model_name']}_{params_str}_{timestamp}"

            # Auto-save all plots
            st.write(f"### Automatically saving comparison plots to folder: `{run_params}`")
            for col in estimator.target_cols:
                saved_path = plot_utils.auto_save_comparison_plot(
                    out_df, col, st.session_state["inference_time_col"], st.session_state["model_name"], run_params
                )
                st.write(f"- Saved: {saved_path}")

            st.success(f"Neural Network execution finished. Plots are in `plots/{run_params}/`")

    except Exception as e:
        st.error(f"Neural Network failed: {e}")

if "inference_output_df" in st.session_state:
    out_df = st.session_state["inference_output_df"]
    target_cols = st.session_state["inference_target_cols"]
    x_col = st.session_state["inference_time_col"]

    st.subheader("Preview of measured versus AI predicted strains") # skip the warm up strains!!
    pred_cols = [c for c in out_df.columns if c.startswith("pred_")]
    preview_df = out_df.dropna(subset=pred_cols, how="all").head(50)
    st.dataframe(preview_df, width="stretch")

    st.subheader("Plot measured and simulated data for comparison")
    selectable_outputs = [c for c in target_cols if f"pred_{c}" in out_df.columns]
    selected_output = st.selectbox("Select output to plot", selectable_outputs)

    fig = make_comparison_plot(out_df, selected_output, x_col=x_col)
    st.plotly_chart(fig, width="stretch")

    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download prediction CSV",
        data=csv_bytes,
        file_name="predicted_output.csv",
        mime="text/csv",
    )
else:
    st.info("Upload the trained Neural Network .pt and an input CSV, then execute Neural Network.")
