import numpy as np
import torch
from src.model import LSTMAutoencoder
from src.errors import reconstruction_errors_and_recons  # type: ignore

def test_model_forward_shapes_cpu():
    torch.manual_seed(0)
    B, T, F = 4, 20, 3
    model = LSTMAutoencoder(seq_len=T, n_features=F, embedding_dim=16, num_layers=1)
    x = torch.randn(B, T, F)
    recon, emb = model(x,return_embedding=True)
    assert recon.shape == x.shape
    assert emb.shape == (B, 16)

def test_reconstruction_errors_and_recons_shapes():
    torch.manual_seed(0)
    B, T, F = 4, 20, 3
    model = LSTMAutoencoder(seq_len=T, n_features=F, embedding_dim=16, num_layers=1)
    X = torch.randn(B, T, F)
    errs, recons, embs = reconstruction_errors_and_recons(model, X, device=torch.device("cpu"), batch_size=2)
    assert errs.shape == (B,)
    assert recons.shape == (B, T, F)
    assert embs.shape == (B, 16)
    # errors should be non-negative
    assert np.all(errs >= 0.0)
