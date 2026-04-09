import os
import json
import sqlite3
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import httpx

ROOT = os.path.dirname(os.path.abspath(__file__))
API_URL = "https://delivery-eta-prediction.onrender.com/predict"

st.set_page_config(page_title="Delivery ETA Prediction System", layout="wide")
st.markdown("<style>.block-container{padding-top:2rem;}</style>", unsafe_allow_html=True)

st.title("Delivery ETA Prediction System")
st.caption("Quick-commerce order delivery time estimator — powered by LightGBM")
st.divider()

page = st.sidebar.radio("Navigation", [
    "Predict ETA", "Overview", "Model Comparison", "Predictions Log"
])

# ── Data loaders ───────────────────────────────────────────────────────────────
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

# ── Page: Predict ETA ──────────────────────────────────────────────────────────
if page == "Predict ETA":
    st.subheader("Enter Order Details")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Location**")
        order_lat = st.number_input("Order Latitude",     value=12.97, format="%.4f")
        order_lon = st.number_input("Order Longitude",    value=77.59, format="%.4f")
        wh_lat    = st.number_input("Warehouse Latitude", value=12.93, format="%.4f")
        wh_lon    = st.number_input("Warehouse Longitude",value=77.55, format="%.4f")

    with col2:
        st.markdown("**Order & Conditions**")
        category      = st.selectbox("Category", ["Groceries","Snacks","Dairy","Beverages","Personal Care","Frozen"])
        item_count    = st.slider("Item Count", 1, 50, 5)
        hour          = st.slider("Hour of Day", 0, 23, 19)
        weekday       = st.selectbox("Weekday", [0,1,2,3,4,5,6],
                                     format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        weather       = st.selectbox("Weather", ["Clear","Cloudy","Rainy","Heavy Rain"])
        traffic_index = st.slider("Traffic Index", 0.0, 1.0, 0.65, 0.01)

    st.divider()
    if st.button("Predict ETA", type="primary", use_container_width=True):
        payload = {
            "order_lat": order_lat, "order_lon": order_lon,
            "wh_lat": wh_lat,       "wh_lon": wh_lon,
            "hour": hour,           "weekday": weekday,
            "item_count": item_count, "category": category,
            "weather": weather,     "traffic_index": traffic_index,
        }
        with st.spinner("Predicting..."):
            try:
                resp = httpx.post(API_URL, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted ETA",  f"{data['eta_minutes']} min")
                c2.metric("Model Used",     data["model_used"].replace("_"," ").title())
                c3.metric("Confidence",     data["confidence"])
                if data["confidence"] == "High":
                    st.success("High confidence prediction — short distance order.")
                elif data["confidence"] == "Medium":
                    st.warning("Medium confidence — moderate distance.")
                else:
                    st.error("Low confidence — long distance order.")
            except Exception as e:
                st.error(f"API error: {e}. The API may be waking up — try again in 30 seconds.")

# ── Page: Overview ─────────────────────────────────────────────────────────────
elif page == "Overview":
    if meta:
        cv = meta.get("cv_results", {}).get(meta.get("best_model", ""), {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Model",    meta.get("best_model","").replace("_"," ").title())
        c2.metric("Training Rows", f"{meta.get('n_train',0):,}")
        c3.metric("Features",      meta.get("n_features", 0))
        c4.metric("CV MAE",        f"{cv.get('mae',0):.2f} min")

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
                      color_discrete_sequence=["#1a1a2e"],
                      labels={"eta_minutes": "Avg ETA (min)", "category": ""})
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No order data found.")

# ── Page: Model Comparison ─────────────────────────────────────────────────────
elif page == "Model Comparison":
    st.subheader("Model Comparison — Cross-Validation Results")
    if meta and "cv_results" in meta:
        cv_df = pd.DataFrame(meta["cv_results"]).T.reset_index()
        cv_df.columns = ["Model", "MAE", "RMSE", "R²"]
        cv_df["Model"] = cv_df["Model"].str.replace("_", " ").str.title()
        cv_df = cv_df.sort_values("MAE").reset_index(drop=True)
        st.dataframe(cv_df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(cv_df, x="Model", y="MAE", color="Model",
                         title="MAE — lower is better",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(cv_df, x="Model", y="R²", color="Model",
                          title="R² Score — higher is better",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No CV results found.")

# ── Page: Predictions Log ──────────────────────────────────────────────────────
elif page == "Predictions Log":
    st.subheader("Recent API Predictions")
    if preds.empty:
        st.info("No predictions logged yet. Use the Predict ETA page to generate some.")
    else:
        st.dataframe(preds, use_container_width=True)
