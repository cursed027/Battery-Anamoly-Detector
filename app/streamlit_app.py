# app/streamlit_app.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from src.config import FEATURES, SEQ_LEN, TIME_COL

API_URL = "http://127.0.0.1:8000/predict"

# --- Helper functions ---
def clean_and_sort(df):
    cols = FEATURES + [TIME_COL, "battery_id", "cycle_count"]
    return (df[cols]
            .dropna()
            .sort_values(["battery_id","cycle_count", TIME_COL])
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

def predict_cycle(data):
    payload = {"data": data.tolist()}
    try:
        res = requests.post(API_URL, json=payload, timeout=10)
        return res.json()
    except Exception as e:
        return {"error": str(e)}

# --- Page layout ---
st.set_page_config(
    page_title="Battery Anomaly Detector",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Header ---
st.title("🔋 Battery Anomaly Detection")
st.markdown("Detect irregularities in battery cycles with smart reconstruction error analysis ⚡")

# --- Sidebar ---
st.sidebar.header("⚙️ Input Options")
input_method = st.sidebar.radio("Select input method", ["Upload CSV", "Manual input"])
st.sidebar.markdown(f"**Sequence Length:** {SEQ_LEN} | **Features:** {len(FEATURES)}")

# --- CSV Upload ---
uploaded_file = st.file_uploader("📂 Upload battery cycle CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_clean = clean_and_sort(df)
    X_seq, meta = build_sequences(df_clean)

    all_results = []
    for i, seq in enumerate(X_seq):
        res = predict_cycle(seq)
        all_results.append(res)

    # --- Compute summary metrics ---
    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        mse_vals = [r["mean_reconstruction_error"] for r in valid_results]
        mae_vals = [np.mean([abs(err) for err in r["feature_errors"].values()]) for r in valid_results]
        anomaly_flags = [int(r["global_anomaly"]) for r in valid_results]

        summary = pd.DataFrame({
            "Metric": ["MSE", "MAE", "Anomaly Rate (%)"],
            "Value": [
                np.mean(mse_vals),
                np.mean(mae_vals),
                100 * np.mean(anomaly_flags)
            ]
        })

        st.subheader("📈 Model Effectiveness (Uploaded File)")
        st.table(summary)

        # Error vs Cycle (global)
        error_curve = pd.DataFrame({
            "Cycle": meta["cycle_count"],
            "Mean Error": [r["mean_reconstruction_error"] for r in valid_results]
        })
        fig_curve = px.line(error_curve, x="Cycle", y="Mean Error", title="Error vs Cycle")
        fig_curve.add_hline(
            y=np.mean(list(valid_results[0]["feature_thresholds"].values())),
            line_color="red", line_dash="dash", annotation_text="Threshold"
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    # --- Per-cycle details ---
    for i, res in enumerate(all_results):
        bid, cyc = meta.iloc[i]
        if "error" in res:
            st.error(f"{bid} cycle {cyc}: {res['error']}")
        else:
            st.markdown(f"### 🔎 {bid} cycle {cyc}")
            st.success(f"Global Anomaly: {res['global_anomaly']}")
            st.info(f"Mean Error: {res['mean_reconstruction_error']:.4f} | Threshold: {res['global_threshold']:.4f}")

            # Split tables and plots into columns
            c1, c2 = st.columns(2)

            # Feature-wise anomaly table
            feat_table = []
            for f in FEATURES:
                feat_table.append({
                    "Feature": f,
                    "Error": res["feature_errors"][f],
                    "Threshold": res["feature_thresholds"][f],
                    "Anomaly": res["feature_anomalies"][f]
                })
            with c1:
                st.markdown("🧩 Feature-wise Anomaly Check")
                st.dataframe(pd.DataFrame(feat_table))

            # Feature-wise bar chart
            feat_bar = pd.DataFrame({
                "Feature": FEATURES,
                "Error": [res["feature_errors"][f] for f in FEATURES],
                "Threshold": [res["feature_thresholds"][f] for f in FEATURES]
            })
            fig_bar = px.bar(
                feat_bar,
                x="Feature", y="Error", color="Error",
                title="Feature-wise Errors",
                text="Threshold"
            )
            with c2:
                st.plotly_chart(fig_bar, use_container_width=True)

# --- Manual Input ---
elif input_method == "Manual input":
    st.info(f"Enter {SEQ_LEN} rows of {len(FEATURES)} features manually")
    manual_data = []
    for i in range(SEQ_LEN):
        row = []
        cols = st.columns(len(FEATURES))
        for j, f in enumerate(FEATURES):
            val = cols[j].number_input(f"{f} (row {i+1})", value=0.0, format="%.4f")
            row.append(val)
        manual_data.append(row)

    if st.button("🚀 Run Prediction"):
        data_np = np.array(manual_data, dtype=np.float32)
        result = predict_cycle(data_np)
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"Global Anomaly: {result['global_anomaly']}")
            st.info(f"Reconstruction error: {result['mean_reconstruction_error']:.4f} | Threshold: {result['global_threshold']:.4f}")
            fig = px.line(data_np, title="Battery Cycle Input")
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Made with ⚡ by Milan Kumar Singh")
