import os, joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch  # for saving torch models

from src.config import FEATURES, SEQ_LEN, BATCH_SIZE, ARTIFACTS_DIR, SCALER_PATH, LR, EPOCHS, PATIENCE, EMBED_DIM, NUM_LAYERS
from src.data_utils import clean_and_sort, build_sequences, pick_normal_cycles
from src.model import LSTMAutoencoder
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import plot_losses, plot_latent_pca, plot_error_hist, plot_error_vs_cycle, plot_reconstruction

# --- FIX: make sure MLflow logs locally ---
mlruns_path = os.path.abspath("mlruns").replace("\\", "/")
mlflow.set_tracking_uri(f"file:///{mlruns_path.lstrip('/')}")
mlflow.set_experiment("default")



def run_pipeline(train_csv, val_csv, test_csv, device_str=None):
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Start MLflow experiment
    with mlflow.start_run():
        # --- Log parameters ---
        mlflow.log_param("seq_len", SEQ_LEN)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LR)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("patience", PATIENCE)
        mlflow.log_param("embedding_dim", 16)
        mlflow.log_param("num_layers", 1)

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
        X_tr_tensor, X_va_tensor, X_te_tensor = map(
            lambda x: torch.tensor(x, dtype=torch.float32), [X_tr, X_va, X_te]
        )
        train_loader = DataLoader(
            TensorDataset(X_tr_tensor, X_tr_tensor),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        # --- Model ---
        model = LSTMAutoencoder(
            seq_len=SEQ_LEN,
            n_features=len(FEATURES),
            embedding_dim=EMBED_DIM,
            num_layers=NUM_LAYERS
        ).to(device)

        # --- Train ---
        train_losses, val_losses = train_model(model, train_loader, X_va_tensor, device)

        # Log final losses
        mlflow.log_metric("final_train_loss", train_losses[-1])
        mlflow.log_metric("final_val_loss", val_losses[-1])

        # --- Evaluate ---
        (
            val_errors,
            val_recons,
            val_embs,
            test_errors,
            test_recons,
            test_embs,
            per_feature_mask,
            raw_cycle_mask,
            global_cycle_mask,
            persisted_mask,
            thresholds,
            thresh_dict,
            global_thr,
        ) = evaluate_model(model, X_va_tensor, X_te_tensor, device, method="quantile", persistence=2)

        # --- Log metrics ---
        mlflow.log_metric("val_error_mean", float(np.mean(val_errors)))
        mlflow.log_metric("test_error_mean", float(np.mean(test_errors)))

        # Anomaly rates for different definitions
        mlflow.log_metric("raw_anomaly_rate", float(raw_cycle_mask.mean()))
        mlflow.log_metric("global_anomaly_rate", float(global_cycle_mask.mean()))
        mlflow.log_metric("persisted_anomaly_rate", float(persisted_mask.mean()))

        # Log thresholds
        mlflow.log_metric("global_threshold", float(global_thr))
        for feat, thr in thresh_dict.items():
            mlflow.log_metric(f"feature_threshold_{feat}", thr)

        # --- Visualize ---
        plot_losses(train_losses, val_losses)
        plot_reconstruction(X_te, test_recons, persisted_mask, SEQ_LEN)  # use persisted mask for plots
        plot_error_hist(test_errors, thresholds)
        plot_error_vs_cycle(test_errors, persisted_mask, thresholds)
        plot_latent_pca(test_embs, test_errors)


        # Log artifacts (plots, scaler, etc.)
        for file in os.listdir(ARTIFACTS_DIR):
            mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, file))

        # --- Save model ---
        # Prepare input example for MLflow (must NOT be torch.Tensor directly)
        cpu_model = model.to("cpu")
        example_input = torch.randn(BATCH_SIZE, SEQ_LEN, len(FEATURES), dtype=torch.float32)
        example_input_np = example_input.numpy().astype(np.float32)  

        mlflow.pytorch.log_model(
            cpu_model,
            name="BatteryLSTM",
            input_example=example_input_np
        )


        print("Pipeline finished. Artifacts and run tracked with MLflow.")


if __name__ == "__main__":
    run_pipeline(
        r"C:\Users\27mil\OneDrive\Desktop\Battery Anamoly Detector\Battery-Anamoly-Detector\datasets\processed\train_dataset.csv", \
        r"C:\Users\27mil\OneDrive\Desktop\Battery Anamoly Detector\Battery-Anamoly-Detector\datasets\processed\val_dataset.csv",\
        r"C:\Users\27mil\OneDrive\Desktop\Battery Anamoly Detector\Battery-Anamoly-Detector\datasets\processed\test_dataset.csv"
    )
