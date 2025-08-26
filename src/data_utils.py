import numpy as np
import pandas as pd
from config import FEATURES, TIME_COL, SEQ_LEN

def clean_and_sort(df):
    cols = FEATURES + [TIME_COL, "battery_id", "cycle_count"]
    return (df[cols]
            .dropna()
            .sort_values(["battery_id","cycle_count",TIME_COL])
            .reset_index(drop=True))

def resample_cycle_to_len(cycle_df, seq_len=SEQ_LEN):
    n = len(cycle_df)
    if n < 2:
        cycle_df = pd.concat([cycle_df, cycle_df], ignore_index=True)
        n = 2
    x_old = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, seq_len)
    cols = []
    for col in FEATURES:
        y = cycle_df[col].values.astype(np.float64)
        cols.append(np.interp(x_new, x_old, y))
    return np.stack(cols, axis=1)

def build_sequences(df):
    X, meta_rows = [], []
    for (bid, cyc), g in df.groupby(["battery_id","cycle_count"], sort=False):
        X.append(resample_cycle_to_len(g))
        meta_rows.append((bid, int(cyc)))
    X = np.stack(X) if len(X) else np.zeros((0, SEQ_LEN, len(FEATURES)))
    meta = pd.DataFrame(meta_rows, columns=["battery_id","cycle_count"])
    return X, meta

def pick_normal_cycles(meta):
    max_by_batt = meta.groupby("battery_id")["cycle_count"].max().to_dict()
    mask = []
    for _, r in meta.iterrows():
        mx = int(max_by_batt[r["battery_id"]])
        limit = min(40, max(1, int(0.4 * mx)))
        mask.append(int(r["cycle_count"]) <= limit)
    return np.array(mask, dtype=bool)
