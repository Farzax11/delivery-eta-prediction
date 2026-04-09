"""
Training Pipeline
Trains multiple regression models, selects best, saves artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from loguru import logger

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

from data_processing import generate_dataset, clean, save_to_csv, save_to_sql
from feature_engineering import build_features, get_feature_columns, correlation_report
from evaluate import compute_metrics

MODELS_DIR = "models"
DATA_DIR   = "data"
CV_FOLDS   = 5
RANDOM_STATE = 42


def get_models() -> dict:
    return {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression())
        ]),
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0))
        ]),
        "lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Lasso(alpha=0.1, max_iter=5000))
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=12,
            min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "xgboost": xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0
        ),
        "lightgbm": lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbose=-1
        ),
    }


def train(regenerate: bool = True) -> str:
    """Full training run. Returns name of best model."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,   exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    if regenerate:
        logger.info("Generating dataset...")
        raw = clean(generate_dataset())
        save_to_csv(raw)
        save_to_sql(raw)
    else:
        raw = pd.read_csv(f"{DATA_DIR}/raw_orders.csv")
        raw = clean(raw)

    eng = build_features(raw)
    corr = correlation_report(eng)

    feature_cols = get_feature_columns(eng)
    X = eng[feature_cols].values
    y = eng["eta_minutes"].values

    logger.info(f"Training on {X.shape[0]} samples, {X.shape[1]} features")

    # ── Cross-validation ──────────────────────────────────────────────────────
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]

    results = {}
    models  = get_models()

    for name, model in models.items():
        logger.info(f"CV: {name}")
        cv = cross_validate(model, X, y, cv=kf, scoring=scoring, n_jobs=-1)
        results[name] = {
            "mae":  -cv["test_neg_mean_absolute_error"].mean(),
            "rmse": -cv["test_neg_root_mean_squared_error"].mean(),
            "r2":    cv["test_r2"].mean(),
        }
        logger.info(f"  MAE={results[name]['mae']:.3f}  RMSE={results[name]['rmse']:.3f}  R²={results[name]['r2']:.4f}")

    # ── Select best by MAE ────────────────────────────────────────────────────
    best_name = min(results, key=lambda k: results[k]["mae"])
    logger.info(f"Best model: {best_name}  (MAE={results[best_name]['mae']:.3f})")

    # ── Retrain best on full data ─────────────────────────────────────────────
    best_model = models[best_name]
    best_model.fit(X, y)

    # ── Save artifacts ────────────────────────────────────────────────────────
    joblib.dump(best_model,  f"{MODELS_DIR}/best_model.pkl")
    joblib.dump(feature_cols, f"{MODELS_DIR}/feature_cols.pkl")

    # Save all models for comparison
    for name, model in models.items():
        model.fit(X, y)
        joblib.dump(model, f"{MODELS_DIR}/{name}.pkl")

    # Save metadata
    meta = {
        "best_model":   best_name,
        "feature_cols": feature_cols,
        "cv_results":   results,
        "n_train":      int(X.shape[0]),
        "n_features":   int(X.shape[1]),
    }
    with open(f"{MODELS_DIR}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.success(f"Training complete. Best: {best_name}")
    return best_name


if __name__ == "__main__":
    best = train(regenerate=True)
    print(f"\nBest model saved: {best}")
