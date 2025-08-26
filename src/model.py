import torch.nn as nn
from src.config import SEQ_LEN

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, n_features=3, embedding_dim=128, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_lstm = nn.LSTM(
            input_size=n_features, hidden_size=embedding_dim,
            num_layers=num_layers, batch_first=True
        )
        self.decoder_lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=embedding_dim,
            num_layers=num_layers, batch_first=True
        )
        self.output_layer = nn.Linear(embedding_dim, n_features)

    def forward(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        hidden_last = hidden[-1]
        hidden_rep = hidden_last.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded, _ = self.decoder_lstm(hidden_rep)
        out = self.output_layer(decoded)
        return out, hidden_last
