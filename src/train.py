import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from src.config import ARTIFACTS_DIR, MODEL_PATH, LR, EPOCHS, PATIENCE

def train_model(model, train_loader, X_val_tensor, device):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val, best_epoch = np.inf, -1
    patience_counter, train_losses, val_losses = 0, [], []

    # --- MLflow: start run ---
    with mlflow.start_run():
        mlflow.log_params({
            "lr": LR,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "optimizer": "Adam",
            "loss_fn": "MSELoss"
        })

        for epoch in range(1, EPOCHS+1):
            model.train()
            running, batches = 0.0, 0
            for xb, _ in train_loader:
                xb = xb.to(device)
                optimizer.zero_grad()
                recon, _ = model(xb)
                loss = criterion(recon, xb)
                loss.backward()
                optimizer.step()
                running += loss.item(); batches += 1
            train_loss = running / max(1, batches)
            train_losses.append(train_loss)

            model.eval()
            with torch.no_grad():
                X_val_gpu = X_val_tensor.to(device)
                recon_val, _ = model(X_val_gpu)
                val_loss = criterion(recon_val, X_val_gpu).item()
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            # --- MLflow: log metrics ---
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            print(f"Epoch {epoch}: Train {train_loss:.4f} | Val {val_loss:.4f}")

            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"  -> saved {MODEL_PATH} (epoch {epoch})")
                patience_counter = 0

                # --- MLflow: save model ---
                mlflow.pytorch.log_model(model, "model")

            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("Early stopping triggered!")
                    break

        mlflow.log_metric("best_val", best_val)
        mlflow.set_tag("best_epoch", best_epoch)

    print(f"Training finished. Best val: {best_val:.4f} at epoch {best_epoch}")
    return train_losses, val_losses
