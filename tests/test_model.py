from src.model import LSTMAutoencoder
import torch

def test_build_model_forward_pass():
    model = LSTMAutoencoder()
    x = torch.randn(2, 300, 3)   # (batch, seq_len, features)
    out = model(x)    
    if isinstance(out, tuple):
        out = out[0]   # take the reconstructed output
    assert out.shape[0] == 2
        # batch dimension preserved
