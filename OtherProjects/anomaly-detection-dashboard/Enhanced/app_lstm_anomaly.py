# app_lstm_anomaly.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
import os
from train_lstm_autoencoder import LSTMAutoencoder, TimeSeriesDataset
from data_utils import generate_sine_data
import plotly.express as px

st.set_page_config(page_title="LSTM Autoencoder Anomaly Detection", layout="wide")
st.title("⚠️ LSTM Autoencoder — Time Series Anomaly Detection")

# Sidebar options
st.sidebar.header("Configuration")
use_synthetic = st.sidebar.checkbox("Use synthetic demo data", value=True)
seq_len = st.sidebar.number_input("Sequence length (window)", min_value=10, max_value=500, value=50)
threshold_k = st.sidebar.slider("Threshold (mean + k * std)", min_value=0.5, max_value=6.0, value=3.0, step=0.1)
model_dir = st.sidebar.text_input("Model directory (to load pretrained)", value="model")
use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=False)

if use_synthetic:
    df = generate_sine_data(length=2000, seq_len=seq_len, noise=0.05, anomaly_ratio=0.02)
else:
    uploaded = st.file_uploader("Upload CSV (with 'value' column or first numeric column)", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV or use synthetic demo data.")
        st.stop()
    df = pd.read_csv(uploaded)
    if 'value' not in df.columns:
        df['value'] = df.select_dtypes(include=np.number).iloc[:,0]

st.subheader("Data preview")
st.write(df.head())

# Load model + scaler if present
model_path = os.path.join(model_dir, "lstm_ae.pt")
scaler_path = os.path.join(model_dir, "scaler.joblib")
if os.path.exists(model_path) and os.path.exists(scaler_path):
    st.info(f"Loading model from {model_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    scaler = joblib.load(scaler_path)
    model = LSTMAutoencoder(
        seq_len=seq_len,
        n_features=1,
        latent_dim=8,        # or args.latent_dim if you're passing via CLI
        hidden_size=32,      # or args.hidden_size
        num_layers=1
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
else:
    st.warning("Pretrained model not found in the model directory. You can train local model by running train_lstm_autoencoder.py or check 'Use synthetic demo' for default.")
    model = None
    scaler = None

# If user wants to run train in-app (optional small training)
if st.button("Train quick model on this data (small)"):
    st.info("Training small model — this may take time depending on CPU/GPU and data size.")
    # For safety keep training quick: small epochs, small hidden size
    from train_lstm_autoencoder import train
    # Run training with synthetic or uploaded CSV; store to model_dir
    import subprocess, sys
    cmd = [
        sys.executable, "train_lstm_autoencoder.py",
        "--out_dir", model_dir,
        "--seq_len", str(seq_len),
        "--epochs", "5",
        "--batch_size", "64",
        "--latent_dim", "8",
        "--hidden_size", "32",
    ]
    if use_gpu:
        cmd.append("--use_cuda")
    if use_synthetic:
        cmd.append("--synthetic")
    # run synchronously (blocks UI) — okay for small jobs
    with st.spinner("Training..."):
        subprocess.check_call(cmd)
    st.success("Training finished. Reloading model.")

# If model is available, run inference
if model is not None and scaler is not None:
    series = df['value'].values
    scaled = scaler.transform(series.reshape(-1,1)).flatten()
    # Create windows (sliding)
    windows = []
    index_windows = []
    for i in range(len(scaled) - seq_len + 1):
        windows.append(scaled[i:i+seq_len])
        index_windows.append(i + seq_len - 1)  # align anomaly score at window end index
    windows = np.array(windows).astype(np.float32)
    # run through model in batches
    device = next(model.parameters()).device
    batch_size = 256
    reconstructions = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.tensor(windows[i:i+batch_size]).unsqueeze(-1).to(device)  # (B, seq_len, 1)
            out = model(batch)  # (B, seq_len, 1)
            out = out.cpu().numpy().squeeze(-1)
            reconstructions.append(out)
    reconstructions = np.vstack(reconstructions)
    # compute reconstruction error per window (MSE over sequence)
    errors = ((reconstructions - windows) ** 2).mean(axis=1)
    # create series of error aligned with end index
    error_series = np.zeros(len(series))
    error_series[:] = np.nan
    for idx, e in zip(index_windows, errors):
        error_series[idx] = e
    # threshold
    mu = np.nanmean(errors)
    sigma = np.nanstd(errors)
    thresh = mu + threshold_k * sigma
    anomaly_flags = error_series > thresh

    # visualization
    fig = px.line(x=df['timestamp'] if 'timestamp' in df.columns else np.arange(len(series)),
                  y=series, labels={'x':'timestamp','y':'value'}, title="Time Series with Anomalies Highlighted")
    anomaly_indices = np.where(anomaly_flags)[0]
    if len(anomaly_indices) > 0:
        fig.add_scatter(x=anomaly_indices, y=series[anomaly_indices], mode='markers', marker=dict(color='red', size=8), name='Anomaly')
    st.plotly_chart(fig, use_container_width=True)

    # show error plot
    fig2 = px.line(x=np.arange(len(error_series)), y=error_series, labels={'x':'index','y':'reconstruction_error'}, title="Reconstruction Error (per sliding window)")
    fig2.add_hline(y=thresh, line_dash="dash", annotation_text=f"threshold={thresh:.3f}")
    st.plotly_chart(fig2, use_container_width=True)

    # summary
    num = np.nansum(anomaly_flags)
    st.write(f"Detected {int(num)} anomaly points (windows) with threshold k={threshold_k} (mu + k*std).")
    st.write("You can adjust threshold slider to be more/less sensitive.")
