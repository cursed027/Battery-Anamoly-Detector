import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.config import ARTIFACTS_DIR

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("Train/Val Loss")
    plt.legend(); plt.grid(True)
    plt.savefig(f"{ARTIFACTS_DIR}/loss_curve.png", dpi=150)
    plt.close()

def plot_reconstruction(X_te, test_recons, test_anom_mask, seq_len):
    os.makedirs(f"{ARTIFACTS_DIR}/plots", exist_ok=True)
    normal_idx = np.where(~test_anom_mask)[0]
    anom_idx = np.where(test_anom_mask)[0]
    choose = []
    if len(normal_idx) > 0: choose.append(normal_idx[len(normal_idx)//2])
    if len(anom_idx) > 0: choose.append(anom_idx[len(anom_idx)//2])
    if not choose: choose = [0]

    for idx in choose:
        orig, recon = X_te[idx], test_recons[idx]
        plt.figure(figsize=(10,4))
        t = np.arange(seq_len)
        for f in range(orig.shape[1]):
            plt.plot(t, orig[:,f], label=f"orig_f{f}")
            plt.plot(t, recon[:,f], "--", label=f"recon_f{f}")
        plt.title(f"Test Cycle {idx} | anomaly={bool(test_anom_mask[idx])}")
        plt.legend(); plt.grid(True)
        plt.savefig(f"{ARTIFACTS_DIR}/plots/recon_cycle_{idx}.png", dpi=150)
        plt.close()

def plot_error_hist(test_errors, thresh):
    plt.figure(figsize=(6,4))
    plt.hist(test_errors, bins=80, alpha=0.8)
    plt.axvline(thresh, color='r', linestyle='--')
    plt.title("Test reconstruction error histogram")
    plt.savefig(f"{ARTIFACTS_DIR}/plots/error_hist.png", dpi=150)
    plt.close()

def plot_error_vs_cycle(test_errors, test_anom_mask, thresh):
    plt.figure(figsize=(10,4))
    plt.plot(test_errors, label="test error")
    plt.scatter(np.where(test_anom_mask)[0], test_errors[test_anom_mask], color='r', s=8, label="anomaly")
    plt.axhline(thresh, color='r', linestyle='--')
    plt.title("Error vs Test Cycle")
    plt.savefig(f"{ARTIFACTS_DIR}/plots/error_vs_cycle.png", dpi=150)
    plt.close()

def plot_latent_pca(test_embs, test_errors):
    try:
        pca = PCA(n_components=2).fit_transform(test_embs)
        plt.figure(figsize=(6,6))
        plt.scatter(pca[:,0], pca[:,1], c=test_errors, cmap="coolwarm", s=8)
        plt.colorbar(label="reconstruction error")
        plt.title("PCA of latent embeddings")
        plt.savefig(f"{ARTIFACTS_DIR}/plots/latent_pca.png", dpi=150)
        plt.close()
    except Exception as e:
        print("PCA plot failed:", e)
