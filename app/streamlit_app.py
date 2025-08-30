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

# --- Helper functions from your training pipeline ---
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

st.title("🔋 Battery Anomaly Detection")
st.markdown(
    """
Upload your battery cycle CSV file (Sequence_Length x Features) or Manually input cycle values.
The app will predict if the cycle shows an anomaly and visualize the reconstruction error.
"""
)

# --- Sidebar ---
st.sidebar.header("Input Options")
input_method = st.sidebar.radio("Select input method", ["Upload CSV", "Manual input"])
st.sidebar.markdown(f"**Sequence Length:** {SEQ_LEN} | **Features:** {len(FEATURES)}")

def predict_cycle(data):
    payload = {"data": data.tolist()}
    try:
        res = requests.post(API_URL, json=payload, timeout=10)
        return res.json()
    except Exception as e:
        return {"error": str(e)}

# --- CSV Upload ---
uploaded_file = st.file_uploader("Upload battery cycle CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_clean = clean_and_sort(df)
    X_seq, meta = build_sequences(df_clean)

    all_results = []
    for i, seq in enumerate(X_seq):
        res = predict_cycle(seq)
        all_results.append(res)

    # Display results
    for i, res in enumerate(all_results):
        bid, cyc = meta.iloc[i]
        if "error" in res:
            st.error(f"{bid} cycle {cyc}: {res['error']}")
        else:
            # Global anomaly decision
            st.success(f"{bid} cycle {cyc} → Global Anomaly: {res['global_anomaly']}")
            st.info(f"Mean Error: {res['mean_reconstruction_error']:.4f} | "
                    f"Global Threshold: {res['global_threshold']:.4f}")

            # Per-feature breakdown
            st.subheader("🔎 Feature-wise anomaly check")
            feat_table = []
            for f in FEATURES:
                feat_table.append({
                    "Feature": f,
                    "Error": res["feature_errors"][f],
                    "Threshold": res["feature_thresholds"][f],
                    "Anomaly": res["feature_anomalies"][f]
                })
            st.dataframe(pd.DataFrame(feat_table))

            # --- 2. Original vs Reconstructed ---
            # --- Display results in columns ---
            if "reconstructed" in res:
                recon = np.array(res["reconstructed"])
                
                # Create 4 columns
                cols = st.columns(4)
                
                # 1. Input sequence plot
                fig_input = px.line(
                    X_seq[i],
                    title=f"{bid} cycle {cyc} input",
                    labels={"index": "Timestep", "value": "Feature Value"}
                )
                cols[0].plotly_chart(fig_input, use_container_width=True)
                
                # 2. Original vs Reconstructed
                fig_recon = px.line(title=f"{bid} cycle {cyc} → Original vs Reconstructed")
                for j, f in enumerate(FEATURES):
                    fig_recon.add_scatter(y=X_seq[i][:, j], mode="lines", name=f"{f} (original)")
                    fig_recon.add_scatter(y=recon[:, j], mode="lines", name=f"{f} (reconstructed)")
                cols[1].plotly_chart(fig_recon, use_container_width=True)
                
                # 3. Error heatmap
                error_matrix = (recon - X_seq[i]) ** 2
                fig_error = px.imshow(
                    error_matrix.T,
                    aspect="auto",
                    labels=dict(x="Timestep", y="Feature", color="Error"),
                    y=FEATURES,
                    title=f"{bid} cycle {cyc} → Error Heatmap"
                )
                cols[2].plotly_chart(fig_error, use_container_width=True)
                
                # 4. Error distribution
                errors = np.mean(error_matrix, axis=1)
                fig_dist = px.histogram(errors, nbins=30, title=f"{bid} cycle {cyc} → Error Distribution")
                fig_dist.add_vline(x=res["global_threshold"], line_color="red", line_dash="dash", annotation_text="Threshold")
                cols[3].plotly_chart(fig_dist, use_container_width=True)


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

    if st.button("Predict Anomaly"):
        data_np = np.array(manual_data, dtype=np.float32)
        result = predict_cycle(data_np)
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"✅ Anomaly: {result['anomaly']}")
            st.info(f"Reconstruction error: {result['reconstruction_error']:.4f} | Threshold: {result['anomaly_threshold']:.4f}")
            fig = px.line(data_np, title="Battery Cycle Input")
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Made with ❤️ by Milan Kumar Singh")
