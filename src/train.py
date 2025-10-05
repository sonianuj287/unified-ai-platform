# src/train.py
import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import joblib

from model_lstm_ae import LSTMAutoencoder
from sklearn.preprocessing import StandardScaler

def create_sequences(array, seq_len):
    sequences = []
    for i in range(len(array) - seq_len + 1):
        sequences.append(array[i:i+seq_len])
    return np.stack(sequences)

def load_and_prepare(csv_path, seq_len):
    df = pd.read_csv(csv_path)
    # numeric-only (should already be processed)
    df = df.select_dtypes(include=[np.number])
    if df.empty:
        raise ValueError("No numeric columns found")
    # For simplicity use first numeric column as series
    arr = df.iloc[:, 0].values.astype(np.float32)
    # scale
    scaler = StandardScaler()
    arr_scaled = scaler.fit_transform(arr.reshape(-1,1)).flatten()
    sequences = create_sequences(arr_scaled, seq_len)
    # result shape: (num_samples, seq_len, n_features=1)
    return sequences[..., np.newaxis], scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--mlflow_experiment", type=str, default="LSTM_Anomaly_Detector")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sequences, scaler = load_and_prepare(args.csv, args.seq_len)
    X = torch.tensor(sequences, dtype=torch.float32)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(args.seq_len, 1, args.latent_dim, args.hidden_size, args.num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_params({
            "seq_len": args.seq_len,
            "latent_dim": args.latent_dim,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr
        })

        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            for batch in loader:
                x = batch[0].to(device)
                optimizer.zero_grad()
                recon = model(x)
                loss = criterion(recon, x)
                loss.backward()
                optimizer.step()
                total += loss.item() * x.size(0)
            avg = total / len(loader.dataset)
            print(f"Epoch {epoch}/{args.epochs} - Loss: {avg:.6f}")
            mlflow.log_metric("loss", avg, step=epoch)

        # save scaler and model artifact locally
        scaler_path = os.path.join(args.out_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        # log model to mlflow
        mlflow.pytorch.log_model(model, "lstm_ae_model", input_example=X[:1].numpy())
        # optionally save model checkpoint locally
        torch.save(model.state_dict(), os.path.join(args.out_dir, "lstm_ae.pt"))
        print(f"Saved model and artifacts under {args.out_dir}")

        # capture run id
        run = mlflow.active_run()
        run_id = run.info.run_id

        # Save scaler and model artifact locally
        scaler_path = os.path.join(args.out_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)

        # log model to mlflow and include input_example
        mlflow.pytorch.log_model(model, "lstm_ae_model", input_example=X[:1].numpy())

        # save model checkpoint locally
        torch.save(model.state_dict(), os.path.join(args.out_dir, args.model_name if hasattr(args, "model_name") else "lstm_ae.pt"))

        # write run id to a file for later use by evaluate / infra
        latest_run_path = os.path.join(args.out_dir, "latest_run.txt")
        with open(latest_run_path, "w") as f:
            f.write(run_id + "\n")

        # Optionally store run_id as an MLflow tag
        mlflow.set_tag("latest_run_id", run_id)

        print(f"Saved model and artifacts under {args.out_dir}. run_id={run_id}")
