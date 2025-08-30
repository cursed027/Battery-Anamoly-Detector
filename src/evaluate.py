import numpy as np
import torch
from src.errors import reconstruction_errors_and_recons
from src.config import MODEL_PATH, ARTIFACTS_DIR,FEATURES
import os
import mlflow
import json
import joblib

def compute_thresholds(errors, method="quantile", quantile=99.0, k=3.0):
    thresholds = []
    for f in range(errors.shape[1]):
        feat_err = errors[:, f]

        if method == "quantile":
            thr = np.percentile(feat_err, quantile)
        elif method == "gaussian":
            mu, sigma = feat_err.mean(), feat_err.std()
            thr = mu + k * sigma
        elif method == "mad":
            med = np.median(feat_err)
            mad = np.median(np.abs(feat_err - med))
            thr = med + k * mad
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        thresholds.append(thr)

    thresholds = np.array(thresholds).reshape(1, -1)
    feature_thresholds = {FEATURES[i]: float(thr) for i, thr in enumerate(thresholds.flatten())}
    return thresholds, feature_thresholds



def apply_persistence(mask, min_consecutive=2):
    """
    Apply persistence rule: an anomaly must last >= min_consecutive steps.
    """
    if not mask.any():
        return mask

    new_mask = mask.copy()
    run_len = 0
    for i, val in enumerate(mask):
        if val:
            run_len += 1
        else:
            if run_len < min_consecutive:
                new_mask[i - run_len : i] = False
            run_len = 0

    # Edge case: end of sequence
    if run_len < min_consecutive:
        new_mask[len(mask) - run_len :] = False

    return new_mask


def evaluate_model(model, X_val_tensor, X_test_tensor, device, method="quantile", persistence=2):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)

    val_errors, val_recons, val_embs = reconstruction_errors_and_recons(model, X_val_tensor, device)
    test_errors, test_recons, test_embs = reconstruction_errors_and_recons(model, X_test_tensor, device)

    # Ensure shape (samples, features)
    if val_errors.ndim == 1:
        val_errors = val_errors[:, None]
        test_errors = test_errors[:, None]

    # Per-feature thresholds
    thresholds, feat_thr = compute_thresholds(val_errors, method=method)

    # Save thresholds
    np.save(os.path.join(ARTIFACTS_DIR, "thresholds.npy"), thresholds)
    mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, "thresholds.npy"))

    with open(os.path.join(ARTIFACTS_DIR, "thresholds.json"), "w") as f:
        json.dump(feat_thr, f, indent=2)
    mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, "thresholds.json"))

    # Save PKL for FastAPI
    joblib.dump(feat_thr, os.path.join(ARTIFACTS_DIR, "thresholds.pkl"))

    # --- Per-feature anomaly mask ---
    per_feature_mask = (test_errors > thresholds[0])

    # --- Cycle-level anomaly (any feature exceeds) ---
    raw_cycle_mask = per_feature_mask.any(axis=1)

    # --- Global error score thresholding ---
    global_scores = test_errors.mean(axis=1)
    global_thr = np.percentile(val_errors.mean(axis=1), 99)  # same quantile method but global
    global_cycle_mask = global_scores > global_thr

    # --- Persistence rule ---
    persisted_mask = apply_persistence(global_cycle_mask, min_consecutive=persistence)

    print(f"Thresholds per feature: {thresholds}")
    print(f"Raw anomalies detected (any feature): {raw_cycle_mask.sum()} / {len(test_errors)}")
    print(f"Global anomalies detected: {global_cycle_mask.sum()} / {len(test_errors)}")
    print(f"Persisted anomalies detected (≥{persistence} cycles): {persisted_mask.sum()} / {len(test_errors)}")

    return (
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
        feat_thr,
        global_thr,
    )
