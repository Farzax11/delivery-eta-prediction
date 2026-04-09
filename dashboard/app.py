import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

st.set_page_config(page_title="Delivery ETA Prediction System", layout="wide")

st.markdown("""
    <style>
        .main-title { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
        .sub-title  { font-size: 1rem; color: #555; margin-top: 0.2rem; margin-bottom: 1.5rem; }
        .block-container { padding-top: 2rem; }
        [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Delivery ETA Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-time monitoring for quick-commerce order delivery estimates</p>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
page = st.sidebar.radio("Navigation", ["Overview", "Model Comparison", "Predictions Log", "Drift Report"])

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_meta():
    path = os.path.join(ROOT, "models", "metadata.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_orders():
    path = os.path.join(ROOT, "data", "delivery.db")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(path)
        df = pd.read_sql("SELECT * FROM orders", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_predictions_log():
    path = os.path.join(ROOT, "data", "delivery.db")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(path)
        df = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 200", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_drift():
    path = os.path.join(ROOT, "data", "drift_reports", "drift_report.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

meta  = load_meta()
df    = load_orders()
preds = load_predictions_log()

# ── Overview ───────────────────────────────────────────────────────────────────
if page == "Overview":
    if meta:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Model",    meta.get("best_model", "N/A"))
        c2.metric("Training Rows", f"{meta.get('n_train', 0):,}")
        c3.metric("Features",      meta.get("n_features", 0))
        cv = meta.get("cv_results", {}).get(meta.get("best_model", ""), {})
        c4.metric("CV MAE",        f"{cv.get('mae', 0):.2f} min")
    else:
        st.info("No model metadata found. Train the model first.")

    if not df.empty:
        st.subheader("ETA Distribution")
        fig = px.histogram(df, x="eta_minutes", nbins=50, color_discrete_sequence=["#1a1a2e"])
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Avg ETA by Hour")
            hourly = df.groupby("hour")["eta_minutes"].mean().reset_index()
            fig2 = px.line(hourly, x="hour", y="eta_minutes", markers=True,
                           color_discrete_sequence=["#1a1a2e"])
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            st.subheader("ETA by Weather")
            fig3 = px.box(df, x="weather", y="eta_minutes", color="weather")
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No order data found.")

# ── Model Comparison ───────────────────────────────────────────────────────────
elif page == "Model Comparison":
    st.subheader("Model Comparison (Cross-Validation Results)")
    if meta and "cv_results" in meta:
        cv_df = pd.DataFrame(meta["cv_results"]).T.reset_index()
        cv_df.columns = ["Model", "MAE", "RMSE", "R²"]
        cv_df = cv_df.sort_values("MAE").reset_index(drop=True)
        st.dataframe(cv_df, use_container_width=True)
        fig = px.bar(cv_df, x="Model", y="MAE", color="Model",
                     title="MAE by Model (lower is better)",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No CV results found.")

# ── Predictions Log ────────────────────────────────────────────────────────────
elif page == "Predictions Log":
    st.subheader("Recent Predictions")
    if preds.empty:
        st.info("No predictions logged yet. Send requests to the API first.")
    else:
        st.dataframe(preds, use_container_width=True)

# ── Drift Report ───────────────────────────────────────────────────────────────
elif page == "Drift Report":
    st.subheader("Data Drift Detection (KS Test)")
    report = load_drift()
    if report is None:
        st.info("No drift report found. Run `python monitoring/drift_detector.py` first.")
    else:
        rows = [{"Feature": k, "KS Statistic": v["statistic"],
                 "P-Value": v["p_value"],
                 "Status": "Drifted" if v["drifted"] else "OK"}
                for k, v in report.items()]
        drift_df = pd.DataFrame(rows)
        st.dataframe(drift_df, use_container_width=True)
        drifted = drift_df[drift_df["Status"] == "Drifted"]
        if not drifted.empty:
            st.warning(f"{len(drifted)} feature(s) show drift: {', '.join(drifted['Feature'].tolist())}")
        else:
            st.success("No drift detected.")
