"""
Streamlit Dashboard
Shows model performance, prediction logs, and drift status.
Run: streamlit run dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Delivery ETA Prediction System", layout="wide")
st.markdown("""
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1a1a2e;
            margin-bottom: 0;
        }
        .sub-title {
            font-size: 1rem;
            color: #555;
            margin-top: 0.2rem;
            margin-bottom: 1.5rem;
        }
        .block-container { padding-top: 2rem; }
        [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Delivery ETA Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-time monitoring for quick-commerce order delivery estimates</p>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Predictions Log", "Drift Report", "Model Comparison"])

# ── Load metadata ──────────────────────────────────────────────────────────────
@st.cache_data
def load_meta():
    try:
        with open("models/metadata.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_orders():
    try:
        from data_processing import load_from_sql
        return load_from_sql()
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_predictions_log():
    try:
        from monitoring.tracker import load_predictions
        records = load_predictions()
        if not records:
            return pd.DataFrame()
        rows = []
        for r in records:
            feat = json.loads(r["features"])
            feat["predicted"] = r["predicted"]
            feat["actual"]    = r["actual"]
            feat["timestamp"] = r["timestamp"]
            rows.append(feat)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

meta  = load_meta()
df    = load_orders()
preds = load_predictions_log()

# ── Overview ───────────────────────────────────────────────────────────────────
if page == "Overview":
    if meta:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Model",   meta["best_model"])
        col2.metric("Training Rows", f"{meta['n_train']:,}")
        col3.metric("Features",      meta["n_features"])
        best = meta["best_model"]
        cv   = meta["cv_results"].get(best, {})
        col4.metric("CV MAE", f"{cv.get('mae', 'N/A'):.2f} min")

    if not df.empty:
        st.subheader("ETA Distribution")
        fig = px.histogram(df, x="eta_minutes", nbins=50, color_discrete_sequence=["#636EFA"])
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Avg ETA by Hour")
            hourly = df.groupby("hour")["eta_minutes"].mean().reset_index()
            fig2 = px.line(hourly, x="hour", y="eta_minutes", markers=True)
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            st.subheader("ETA by Weather")
            fig3 = px.box(df, x="weather", y="eta_minutes", color="weather")
            st.plotly_chart(fig3, use_container_width=True)

# ── Predictions Log ────────────────────────────────────────────────────────────
elif page == "Predictions Log":
    st.subheader("Recent Predictions")
    if preds.empty:
        st.info("No predictions logged yet. Start the API and send some requests.")
    else:
        st.dataframe(preds.tail(100), use_container_width=True)
        if "actual" in preds.columns:
            valid = preds.dropna(subset=["actual"])
            if not valid.empty:
                errors = valid["actual"] - valid["predicted"]
                st.metric("Mean Error (actual - predicted)", f"{errors.mean():.2f} min")

# ── Drift Report ───────────────────────────────────────────────────────────────
elif page == "Drift Report":
    st.subheader("Data Drift Detection (KS Test)")
    try:
        with open("data/drift_reports/drift_report.json") as f:
            report = json.load(f)
        rows = [{"feature": k, **v} for k, v in report.items()]
        drift_df = pd.DataFrame(rows)
        drift_df["status"] = drift_df["drifted"].map({True: "🔴 Drifted", False: "🟢 OK"})
        st.dataframe(drift_df[["feature", "statistic", "p_value", "status"]], use_container_width=True)
    except FileNotFoundError:
        st.info("Run `python monitoring/drift_detector.py` to generate a drift report.")

# ── Model Comparison ───────────────────────────────────────────────────────────
elif page == "Model Comparison":
    st.subheader("Model Comparison (Cross-Validation)")
    if meta:
        cv_df = pd.DataFrame(meta["cv_results"]).T.reset_index()
        cv_df.columns = ["Model", "MAE", "RMSE", "R²"]
        cv_df = cv_df.sort_values("MAE")
        st.dataframe(cv_df, use_container_width=True)

        fig = px.bar(cv_df, x="Model", y="MAE", color="Model", title="MAE by Model (lower is better)")
        st.plotly_chart(fig, use_container_width=True)

    try:
        comp = pd.read_csv("data/plots/model_comparison.csv", index_col=0)
        st.subheader("Test Set Metrics")
        st.dataframe(comp, use_container_width=True)
    except FileNotFoundError:
        st.info("Run `python src/evaluate.py` to generate test-set comparison.")
