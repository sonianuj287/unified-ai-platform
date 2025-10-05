# src/model_lstm_ae.py
import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, latent_dim, hidden_size, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.latent = nn.Linear(hidden_size, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_size)
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.decoder_output = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        # x: (B, seq_len, n_features)
        enc_out, (h, c) = self.encoder(x)
        latent = self.latent(enc_out[:, -1, :])  # (B, latent_dim)

        # decode using input seq length (dynamic)
        dec_in = self.decoder_input(latent).unsqueeze(1).repeat(1, x.size(1), 1)  # (B, seq_len, hidden_size)
        dec_out, _ = self.decoder(dec_in, (h, torch.zeros_like(h)))
        out = self.decoder_output(dec_out)
        return out
