import os, joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES, SEQ_LEN, BATCH_SIZE, ARTIFACTS_DIR, SCALER_PATH
from src.data_utils import clean_and_sort, build_sequences, pick_normal_cycles
from src.model import LSTMAutoencoder
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import plot_losses,plot_latent_pca,plot_error_hist,plot_error_vs_cycle,plot_reconstruction

def run_pipeline(train_csv, val_csv, test_csv, device_str=None):
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))

    # --- Load CSVs ---
    train_raw = clean_and_sort(pd.read_csv(train_csv))
    val_raw   = clean_and_sort(pd.read_csv(val_csv))
    test_raw  = clean_and_sort(pd.read_csv(test_csv))
    X_tr_all, meta_tr = build_sequences(train_raw)
    X_va, meta_va = build_sequences(val_raw)
    X_te, meta_te = build_sequences(test_raw)

    # --- Pick normal cycles ---
    normal_mask = pick_normal_cycles(meta_tr)
    X_tr = X_tr_all[normal_mask]

    # --- Scale ---
    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(-1, X_tr.shape[-1]))
    X_tr = scaler.transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
    X_va = scaler.transform(X_va.reshape(-1, X_va.shape[-1])).reshape(X_va.shape)
    X_te = scaler.transform(X_te.reshape(-1, X_te.shape[-1])).reshape(X_te.shape)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    # --- Tensors ---
    X_tr_tensor, X_va_tensor, X_te_tensor = map(lambda x: torch.tensor(x, dtype=torch.float32), [X_tr, X_va, X_te])
    train_loader = DataLoader(TensorDataset(X_tr_tensor, X_tr_tensor), batch_size=BATCH_SIZE, shuffle=True)

    # --- Model ---
    model = LSTMAutoencoder(seq_len=SEQ_LEN, n_features=len(FEATURES), embedding_dim=256, num_layers=2).to(device)

    # --- Train ---
    train_losses, val_losses = train_model(model, train_loader, X_va_tensor, device)

    # --- Evaluate ---
    val_errors, val_recons, val_embs, test_errors, test_recons, test_embs, test_anom_mask, thresh = \
        evaluate_model(model, X_va_tensor, X_te_tensor, device)

    # --- Visualize ---
    plot_losses(train_losses, val_losses)
    plot_reconstruction(X_te, test_recons, test_anom_mask, SEQ_LEN)
    plot_error_hist(test_errors, thresh)
    plot_error_vs_cycle(test_errors, test_anom_mask, thresh)
    plot_latent_pca(test_embs, test_errors)

    print("Pipeline finished. Artifacts saved to", ARTIFACTS_DIR)

if __name__ == "__main__":
    run_pipeline("datasets\train_dataset.csv", "datasets\val_dataset.csv", "datasets\test_dataset.csv")
