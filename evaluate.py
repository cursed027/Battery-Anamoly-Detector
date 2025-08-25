import numpy as np
import torch
from errors import reconstruction_errors_and_recons
from config import MODEL_PATH

def evaluate_model(model, X_val_tensor, X_test_tensor, device):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    val_errors, val_recons, val_embs = reconstruction_errors_and_recons(model, X_val_tensor, device)
    test_errors, test_recons, test_embs = reconstruction_errors_and_recons(model, X_test_tensor, device)

    thresh = np.percentile(val_errors, 99.0)
    test_anom_mask = test_errors > thresh
    print(f"Threshold {thresh:.4f}, anomalies detected: {test_anom_mask.sum()} / {len(test_errors)}")
    return val_errors, val_recons, val_embs, test_errors, test_recons, test_embs, test_anom_mask, thresh
