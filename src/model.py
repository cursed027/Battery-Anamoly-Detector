import torch
import torch.nn as nn
from src.config import SEQ_LEN

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, n_features=3, embedding_dim=128, num_layers=2, bottleneck_dim=64):
        super().__init__()
        self.seq_len = seq_len

        # --- Encoder ---
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.encoder_norm = nn.LayerNorm(embedding_dim)

        # Bottleneck: compress representation
        self.bottleneck = nn.Linear(embedding_dim, bottleneck_dim)

        # --- Decoder ---
        self.decoder_lstm = nn.LSTM(
            input_size=bottleneck_dim,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.decoder_norm = nn.LayerNorm(embedding_dim)

        self.output_layer = nn.Linear(embedding_dim, n_features)

    def forward(self, x, return_embedding=False):
        # --- Encode ---
        _, (hidden, _) = self.encoder_lstm(x)
        hidden_last = hidden[-1]                     # (batch, embed_dim)
        hidden_norm = self.encoder_norm(hidden_last) # normalize
        compressed = self.bottleneck(hidden_norm)    # bottleneck (batch, bottleneck_dim)

        # --- Repeat compressed for seq reconstruction ---
        repeated = compressed.unsqueeze(1).repeat(1, self.seq_len, 1)

        # --- Decode ---
        decoded, _ = self.decoder_lstm(repeated)
        decoded_norm = self.decoder_norm(decoded)
        out = self.output_layer(decoded_norm)

        if return_embedding:
            return out, compressed
        else:
            return out
