import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.config import ARTIFACTS_DIR, FEATURES  # <- import feature names

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("HuberLOSS"); plt.title("Train/Val Loss")
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
            feat_name = FEATURES[f] if f < len(FEATURES) else f"f{f}"
            plt.plot(t, orig[:,f], label=f"{feat_name} (orig)")
            plt.plot(t, recon[:,f], "--", label=f"{feat_name} (recon)")
        plt.title(f"Test Cycle {idx} | anomaly={bool(test_anom_mask[idx])}")
        plt.legend(); plt.grid(True)
        plt.savefig(f"{ARTIFACTS_DIR}/plots/recon_cycle_{idx}.png", dpi=150)
        plt.close()

def plot_error_hist(test_errors, thresh):
    plt.figure(figsize=(6,4))
    plt.hist(test_errors.flatten(), bins=80, alpha=0.8)

    # Handle scalar vs array thresholds
    if np.ndim(thresh) == 0 or np.size(thresh) == 1:
        plt.axvline(float(np.array(thresh).item()), color='r', linestyle='--', label="Threshold")
    else:
        for f, t in enumerate(np.array(thresh).flatten()):
            feat_name = FEATURES[f] if f < len(FEATURES) else f"f{f}"
            plt.axvline(float(t), linestyle="--", alpha=0.6, label=f"{feat_name} thr")
        plt.axvline(np.mean(thresh), color="r", linestyle="--", label="Mean threshold")

    plt.title("Test reconstruction error histogram")
    plt.legend()
    plt.savefig(f"{ARTIFACTS_DIR}/plots/error_hist.png", dpi=150)
    plt.close()

def plot_error_vs_cycle(test_errors, test_anom_mask, thresh):
    plt.figure(figsize=(10,4))
    mean_err = test_errors.mean(axis=1) if test_errors.ndim > 1 else test_errors
    plt.plot(mean_err, label="mean test error")

    # scatter anomalies
    plt.scatter(
        np.where(test_anom_mask)[0],
        mean_err[test_anom_mask],
        color='r', s=8, label="anomaly"
    )

    # Handle scalar vs array thresholds
    if np.ndim(thresh) == 0 or np.size(thresh) == 1:
        plt.axhline(float(np.array(thresh).item()), color='r', linestyle='--', label="Threshold")
    else:
        # Draw per-feature thresholds
        for f, t in enumerate(np.array(thresh).flatten()):
            feat_name = FEATURES[f] if f < len(FEATURES) else f"f{f}"
            plt.axhline(float(t), linestyle="--", alpha=0.5, label=f"{feat_name} thr")
        # Also draw mean threshold
        plt.axhline(np.mean(thresh), color="k", linestyle="--", label="Mean threshold")

    plt.title("Error vs Test Cycle")
    plt.xlabel("Cycle index")
    plt.ylabel("Reconstruction error")
    plt.legend()
    plt.savefig(f"{ARTIFACTS_DIR}/plots/error_vs_cycle.png", dpi=150)
    plt.close()


def plot_latent_pca(test_embs, test_errors):
    try:
        pca = PCA(n_components=2).fit_transform(test_embs)
        plt.figure(figsize=(6,6))
        plt.scatter(
            pca[:,0], pca[:,1],
            c=test_errors.mean(axis=1) if test_errors.ndim > 1 else test_errors,
            cmap="coolwarm", s=8
        )
        plt.colorbar(label="reconstruction error")
        plt.title("PCA of latent embeddings")
        plt.savefig(f"{ARTIFACTS_DIR}/plots/latent_pca.png", dpi=150)
        plt.close()
    except Exception as e:
        print("PCA plot failed:", e)
