# train_lstm_autoencoder.py
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib

# -- Dataset windowing
class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len=50):
        # series: 1-D numpy array
        self.seq_len = seq_len
        self.series = series
        self.windows = []
        for i in range(len(series) - seq_len + 1):
            self.windows.append(series[i:i+seq_len])
        self.windows = np.array(self.windows).astype(np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]

# -- LSTM Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, latent_dim, hidden_size, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Project to latent dimension
        self.to_latent = nn.Linear(hidden_size, latent_dim)

        # Project back from latent to hidden
        self.from_latent = nn.Linear(latent_dim, hidden_size)

        # Decoder
        self.decoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Final output layer
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        # --- Encoder ---
        enc_out, (h, c) = self.encoder(x)
        # Only take final hidden state (last layer)
        h_last = h[-1]  # shape: (batch, hidden_size)
        z = self.to_latent(h_last)  # latent vector

        # --- Decoder ---
        # Map latent vector back to hidden space
        dec_hidden = self.from_latent(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
        dec_cell = torch.zeros_like(dec_hidden)

        # Start decoding with zero input sequence
        dec_input = torch.zeros(x.size(0), self.seq_len, self.n_features, device=x.device)

        # Decode
        dec_out, _ = self.decoder(dec_input, (dec_hidden, dec_cell))
        reconstructed = self.output_layer(dec_out)
        return reconstructed


def train(args):
    # Load CSV (expects columns timestamp, value OR a single column of values)
    if args.synthetic:
        from data_utils import generate_sine_data
        df = generate_sine_data(length=args.length, seq_len=args.seq_len, noise=args.noise, anomaly_ratio=0.0)
        series = df['value'].values
    else:
        df = pd.read_csv(args.csv)
        if 'value' in df.columns:
            series = df['value'].values
        else:
            # take first numeric column
            series = df.select_dtypes(include=np.number).iloc[:,0].values

    # Normalize
    scaler = StandardScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1,1)).flatten()

    dataset = TimeSeriesDataset(series_scaled, seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    model = LSTMAutoencoder(seq_len=args.seq_len, n_features=1, latent_dim=args.latent_dim,
                            hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, args.epochs+1):
        total_loss = 0.0
        for batch in loader:
            x = batch.unsqueeze(-1).to(device)  # (B, seq_len, 1)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg = total_loss / len(loader.dataset)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg:.6f}")

    # Save model and scaler
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "lstm_ae.pt")
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.joblib"))
    print("Saved model to", model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV (if not using synthetic)")
    parser.add_argument("--out_dir", type=str, default="model", help="Where to save model/scaler")
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--length", type=int, default=5000)
    parser.add_argument("--noise", type=float, default=0.05)
    args = parser.parse_args()
    train(args)
