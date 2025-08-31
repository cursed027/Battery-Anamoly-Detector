# app/fastapi_app.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import numpy as np
import joblib

from src.model import LSTMAutoencoder
from src.config import SEQ_LEN, FEATURES, ARTIFACTS_DIR, SCALER_PATH, MODEL_PATH, EMBED_DIM, NUM_LAYERS

app = FastAPI(title="Battery Anomaly Detector API")

# --- Define input schema ---
class CycleInput(BaseModel):
    data: List[List[float]]  # SEQ_LEN x n_features

# --- Load scaler, thresholds and model ---
scaler = joblib.load(SCALER_PATH)
thresholds = joblib.load(os.path.join(ARTIFACTS_DIR, "thresholds.pkl"))  # dict {feature: thr}

model = LSTMAutoencoder(
    seq_len=SEQ_LEN,
    n_features=len(FEATURES),
    embedding_dim=EMBED_DIM,
    num_layers=NUM_LAYERS
)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
device = torch.device("cpu")
model.to(device)


@app.post("/predict")
def predict_anomaly(input: CycleInput):
    try:
        x = np.array(input.data, dtype=np.float32)

        if x.shape != (SEQ_LEN, len(FEATURES)):
            return {"error": f"Expected shape ({SEQ_LEN}, {len(FEATURES)}), got {x.shape}"}

        # Scale
        x_scaled = scaler.transform(x)
        x_scaled = x_scaled.reshape(1, SEQ_LEN, len(FEATURES))
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)

        # Forward pass
        with torch.no_grad():
            recon, _ = model(x_tensor, return_embedding=True)
            recon_np = recon.squeeze(0).cpu().numpy()

        # Compute per-feature reconstruction error
        error_matrix = (recon_np - x_scaled.squeeze(0))**2
        feat_errors = error_matrix.mean(axis=0)  # per feature
        mean_error = feat_errors.mean()

        # Anomaly decisions
        global_thr = np.mean(list(thresholds.values()))
        global_anom = mean_error > global_thr

        feature_anoms = {
            feat: float(err > thresholds[feat])
            for feat, err in zip(FEATURES, feat_errors)
        }

        # Compute metrics
        mse = float(np.mean(error_matrix))
        mae = float(np.mean(np.abs(recon_np - x_scaled.squeeze(0))))

        return {
            "mean_reconstruction_error": float(mean_error),
            "global_threshold": float(global_thr),
            "global_anomaly": bool(global_anom),
            "feature_errors": dict(zip(FEATURES, feat_errors.tolist())),
            "feature_thresholds": thresholds,
            "feature_anomalies": feature_anoms,
            "reconstructed": recon_np.tolist(),
            "mae":mae,
            "mse":mse
        }

    except Exception as e:
        return {"error": str(e)}
