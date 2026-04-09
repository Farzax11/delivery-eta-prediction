import streamlit as st
import json
import os

st.title("Delivery ETA Prediction System")

# Load model metadata
meta_path = os.path.join(os.path.dirname(__file__), "models", "metadata.json")
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    st.success(f"Model loaded: {meta['best_model']}")
    st.json(meta["cv_results"])
else:
    st.error("Model metadata not found")
