import os
import json
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Delivery ETA Prediction System", layout="wide")

st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("Delivery ETA Prediction System")
st.caption("Real-time monitoring for quick-commerce order delivery estimates")
st.divider()

page = st.sidebar.radio("Navigation", ["Overview", "Model Comparison", "Predictions Log"])

@st.cache_data
def load_meta():
    path = os.path.join(ROOT, "models", "metadata.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_orders():
    csv = os.path.join(ROOT, "data", "raw_orders.csv")
    if os.path.exists(csv):
        return pd.read_csv(csv)
    db = os.path.join(ROOT, "data", "delivery.db")
    if os.path.exists(db):
        conn = sqlite3.connect(db)
        df = pd.read_sql("SELECT * FROM orders", conn)
        conn.close()
        return df
    return pd.DataFrame()

@st.cache_data
def load_predictions_log():
    db = os.path.join(ROOT, "data", "delivery.db")
    if not os.path.exists(db):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(db)
        df = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 200", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

meta  = load_meta()
df    = load_orders()
preds = load_predictions_log()

if page == "Overview":
    if meta:
        cv = meta.get("cv_results", {}).get(meta.get("best_model", ""), {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Model",    meta.get("best_model", "N/A").replace("_", " ").title())
        c2.metric("Training Rows", f"{meta.get('n_train', 0):,}")
        c3.metric("Features",      meta.get("n_features", 0))
        c4.metric("CV MAE",        f"{cv.get('mae', 0):.2f} min")

    if not df.empty:
        st.subheader("ETA Distribution")
        fig = px.histogram(df[df["eta_minutes"] <= 90], x="eta_minutes", nbins=50,
                           color_discrete_sequence=["#1a1a2e"],
                           labels={"eta_minutes": "ETA (minutes)"})
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

        st.subheader("Avg ETA by Category")
        cat = df.groupby("category")["eta_minutes"].mean().reset_index().sort_values("eta_minutes")
        fig4 = px.bar(cat, x="eta_minutes", y="category", orientation="h",
                      color_discrete_sequence=["#1a1a2e"])
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No order data found.")

elif page == "Model Comparison":
    st.subheader("Model Comparison — Cross-Validation Results")
    if meta and "cv_results" in meta:
        cv_df = pd.DataFrame(meta["cv_results"]).T.reset_index()
        cv_df.columns = ["Model", "MAE", "RMSE", "R²"]
        cv_df = cv_df.sort_values("MAE").reset_index(drop=True)
        st.dataframe(cv_df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(cv_df, x="Model", y="MAE", color="Model",
                         title="MAE (lower is better)",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(cv_df, x="Model", y="R²", color="Model",
                          title="R² Score (higher is better)",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No CV results found.")

elif page == "Predictions Log":
    st.subheader("Recent API Predictions")
    if preds.empty:
        st.info("No predictions logged yet. Send requests to the API first.")
        st.markdown("Try the live API: [delivery-eta-prediction.onrender.com/docs](https://delivery-eta-prediction.onrender.com/docs)")
    else:
        st.dataframe(preds, use_container_width=True)
