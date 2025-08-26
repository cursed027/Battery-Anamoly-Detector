# Tries to import from your modular layout first, falls back to the monolith script.
try:
    from src.data.processing import clean_and_sort, resample_cycle_to_len, build_sequences, pick_normal_cycles  # type: ignore
except Exception:
    try:
        from data.processing import clean_and_sort, resample_cycle_to_len, build_sequences, pick_normal_cycles  # type: ignore
    except Exception:
        from unified_lstm_ae_pipeline import clean_and_sort, resample_cycle_to_len, build_sequences, pick_normal_cycles  # type: ignore

import numpy as np
import pandas as pd

def _toy_df():
    # 2 cycles, intentionally shuffled time to verify clean_and_sort
    return pd.DataFrame({
        "Voltage_measured": [3.9, 4.0, 3.8, 3.7],
        "Current_measured": [0.5, 0.6, 0.4, 0.3],
        "Temperature_measured": [25.1, 25.3, 24.9, 24.8],
        "Time": [2.0, 1.0, 2.0, 1.0],
        "battery_id": ["A","A","A","A"],
        "cycle_count": [1,1,2,2],
    })

def test_clean_and_sort_orders_by_keys():
    df = _toy_df()
    out = clean_and_sort(df)
    assert list(out["Time"]) == [1.0, 2.0, 1.0, 2.0]

def test_resample_cycle_to_len_shape():
    df = _toy_df()
    g1 = df[df["cycle_count"] == 1]
    arr = resample_cycle_to_len(g1, seq_len=30)
    assert arr.shape == (30, 3)  # (T, F)

def test_build_sequences_shapes_and_meta():
    df = clean_and_sort(_toy_df())
    X, meta = build_sequences(df)
    # default SEQ_LEN inside your code is 300 → expect (2, 300, 3)
    assert X.shape[0] == 2
    assert X.shape[2] == 3
    assert set(meta.columns) == {"battery_id", "cycle_count"}

def test_pick_normal_cycles_limit_logic():
    # A has 10 cycles → floor(0.4*10)=4 (cap 40) → first 4 True
    # B has 5 cycles → floor(0.4*5)=2 → first 2 True
    meta = pd.DataFrame(
        {"battery_id": ["A"]*10 + ["B"]*5,
         "cycle_count": list(range(1,11)) + list(range(1,6))}
    )
    mask = pick_normal_cycles(meta)
    assert mask.sum() == 4 + 2
    # Spot check first and last
    assert mask[0] is True
    assert mask[9] is False
