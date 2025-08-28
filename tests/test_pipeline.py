import os
import pandas as pd
import numpy as np
import tempfile
import torch
import pytest

from main import run_pipeline
from src.config import FEATURES, SEQ_LEN

@pytest.mark.pipeline
def test_dummy_pipeline_runs(tmp_path):
    """
    Integration test: run the full pipeline with dummy CSVs instead of real dataset.
    Ensures training, evaluation, and MLflow logging work without crashing.
    """

    # ---- Step 1: Create dummy CSVs ----
    def make_dummy_csv(path, n_cycles=5, seq_len=SEQ_LEN):
        rows = []
        for cycle in range(1, n_cycles+1):
            for t in range(seq_len):
                rows.append({
                    "Voltage_measured": np.random.uniform(3.0, 4.2),
                    "Current_measured": np.random.uniform(0.1, 1.0),
                    "Temperature_measured": np.random.uniform(20, 35),
                    "Time": float(t),
                    "battery_id": "A",
                    "cycle_count": cycle,
                })
        pd.DataFrame(rows).to_csv(path, index=False)

    train_csv = tmp_path / "train.csv"
    val_csv   = tmp_path / "val.csv"
    test_csv  = tmp_path / "test.csv"

    make_dummy_csv(train_csv)
    make_dummy_csv(val_csv)
    make_dummy_csv(test_csv)

    # ---- Step 2: Run pipeline ----
    run_pipeline(str(train_csv), str(val_csv), str(test_csv), device_str="cpu")

    # ---- Step 3: Assert MLflow created a run ----
    mlruns_path = os.path.abspath("mlruns")
    assert os.path.exists(mlruns_path), "MLflow did not log any runs"
    # At least 1 run directory should exist
    found = False
    for root, dirs, files in os.walk(mlruns_path):
        if "meta.yaml" in files:
            found = True
            break
    assert found, "No MLflow run metadata found"
