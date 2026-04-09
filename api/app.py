"""
FastAPI Prediction Service
Endpoint: POST /predict  →  returns ETA in minutes
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
from monitoring.tracker import log_prediction

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

app = FastAPI(
    title="Delivery ETA Prediction API",
    description="Predicts delivery time (minutes) for quick-commerce orders.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model at startup ──────────────────────────────────────────────────────
model        = None
feature_cols = None
metadata     = None


@app.on_event("startup")
def load_model():
    global model, feature_cols, metadata
    try:
        model        = joblib.load(f"{MODELS_DIR}/best_model.pkl")
        feature_cols = joblib.load(f"{MODELS_DIR}/feature_cols.pkl")
        with open(f"{MODELS_DIR}/metadata.json") as f:
            metadata = json.load(f)
        logger.success(f"Model loaded: {metadata['best_model']}")
    except FileNotFoundError:
        logger.error("Model not found. Run `python src/train.py` first.")


# ── Request / Response schemas ─────────────────────────────────────────────────
class OrderRequest(BaseModel):
    order_lat:     float = Field(..., example=12.97)
    order_lon:     float = Field(..., example=77.59)
    wh_lat:        float = Field(..., example=12.93)
    wh_lon:        float = Field(..., example=77.55)
    hour:          int   = Field(..., ge=0, le=23, example=19)
    weekday:       int   = Field(..., ge=0, le=6,  example=4)
    item_count:    int   = Field(..., ge=1, le=50, example=5)
    category:      str   = Field(..., example="Groceries")
    weather:       str   = Field(..., example="Clear")
    traffic_index: float = Field(..., ge=0.0, le=1.0, example=0.65)


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    eta_minutes:   float
    model_used:    str
    confidence:    str   # Low / Medium / High based on distance


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    if metadata is None:
        raise HTTPException(503, "Model not loaded")
    return metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded. Run training first.")

    from feature_engineering import build_features

    # Build a single-row DataFrame matching training schema
    raw_df = pd.DataFrame([order.model_dump()])

    # Compute distance_km (needed before feature engineering)
    raw_df["distance_km"] = _haversine(
        raw_df["order_lat"], raw_df["order_lon"],
        raw_df["wh_lat"],    raw_df["wh_lon"]
    )

    eng_df = build_features(raw_df, is_train=False)

    # Align columns to training feature set
    for col in feature_cols:
        if col not in eng_df.columns:
            eng_df[col] = 0
    X = eng_df[feature_cols].values

    eta = float(model.predict(X)[0])
    eta = round(max(5.0, min(eta, 90.0)), 1)

    dist = raw_df["distance_km"].iloc[0]
    confidence = "High" if dist < 3 else ("Medium" if dist < 7 else "Low")

    log_prediction(order.model_dump(), eta)

    return PredictionResponse(
        eta_minutes=eta,
        model_used=metadata["best_model"] if metadata else "unknown",
        confidence=confidence,
    )


def _haversine(lat1, lon1, lat2, lon2):
    import numpy as np
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
