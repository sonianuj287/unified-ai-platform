# src/evaluate.py
import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from model_lstm_ae import LSTMAutoencoder

def create_sequences(array, seq_len):
    sequences = []
    for i in range(len(array) - seq_len + 1):
        sequences.append(array[i:i+seq_len])
    return np.stack(sequences)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold_k", type=float, default=3.0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv).select_dtypes(include=[np.number])
    arr = df.iloc[:,0].values.astype(np.float32)

    scaler = joblib.load(os.path.join(args.model_dir, "scaler.joblib"))
    arr_scaled = scaler.transform(arr.reshape(-1,1)).flatten()
    seq_len = 50
    sequences = create_sequences(arr_scaled, seq_len)[..., np.newaxis]
    X = torch.tensor(sequences, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(seq_len, 1, latent_dim=8, hidden_size=32)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "lstm_ae.pt"), map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        recon = model(X.to(device)).cpu().numpy()
    errors = ((recon - sequences) ** 2).mean(axis=(1,2))
    mu, sigma = errors.mean(), errors.std()
    thresh = mu + args.threshold_k * sigma
    num_anom = int((errors > thresh).sum())
    metrics = {
        "reconstruction_error_mean": float(mu),
        "reconstruction_error_std": float(sigma),
        "threshold": float(thresh),
        "num_anomalies": num_anom,
        "num_windows": int(len(errors))
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    run_id = None
    try:
        run_file = os.path.join(args.model_dir, "latest_run.txt")
        if os.path.exists(run_file):
            with open(run_file, "r") as f:
                run_id = f.read().strip()
    except Exception:
        run_id = None

    metrics["run_id"] = run_id


    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] wrote metrics to {args.out}")

    print(f"[evaluate] wrote metrics to {args.out} with run_id={run_id}")
