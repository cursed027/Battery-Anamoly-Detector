import torch
import pandas as pd
import numpy as np

def fake_dataset(num_samples=10, seq_len=20, n_features=3):
    """Return fake numpy arrays shaped like processed battery data."""
    X = np.random.randn(num_samples, seq_len, n_features).astype(np.float32)
    meta = pd.DataFrame({
        "battery_id": ["FAKE"] * num_samples,
        "cycle_count": list(range(1, num_samples + 1))
    })
    return X, meta

def fake_tensor_dataset(batch_size=4, seq_len=20, n_features=3):
    """Return torch tensors shaped like training batches."""
    X = torch.randn(batch_size, seq_len, n_features)
    return X
