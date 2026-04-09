"""
Feature Engineering Pipeline
Transforms raw order data into model-ready features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from loguru import logger


CATEGORY_ORDER = ["Groceries", "Snacks", "Dairy", "Beverages", "Personal Care", "Frozen"]
WEATHER_ORDER  = ["Clear", "Cloudy", "Rainy", "Heavy Rain"]

TARGET = "eta_minutes"
DROP_COLS = ["order_lat", "order_lon", "wh_lat", "wh_lon"]  # raw coords dropped after distance computed


def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = df.copy()

    # ── 1. Time-based features ─────────────────────────────────────────────────
    df["is_peak_hour"] = df["hour"].apply(
        lambda h: 1 if (8 <= h <= 10) or (18 <= h <= 21) else 0
    )
    df["is_weekend"]   = (df["weekday"] >= 5).astype(int)
    df["hour_sin"]     = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]     = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"]      = np.sin(2 * np.pi * df["weekday"] / 7)
    df["day_cos"]      = np.cos(2 * np.pi * df["weekday"] / 7)

    # ── 2. Distance buckets ────────────────────────────────────────────────────
    df["distance_bucket"] = pd.cut(
        df["distance_km"],
        bins=[0, 1, 3, 6, 10, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # ── 3. Order complexity ────────────────────────────────────────────────────
    df["is_large_order"] = (df["item_count"] > 10).astype(int)
    df["log_item_count"] = np.log1p(df["item_count"])

    # ── 4. Interaction features ────────────────────────────────────────────────
    df["traffic_x_distance"] = df["traffic_index"] * df["distance_km"]
    df["peak_x_traffic"]     = df["is_peak_hour"] * df["traffic_index"]

    # ── 5. Encode categoricals ─────────────────────────────────────────────────
    cat_map     = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    weather_map = {w: i for i, w in enumerate(WEATHER_ORDER)}

    df["category_enc"] = df["category"].map(cat_map).fillna(0).astype(int)
    df["weather_enc"]  = df["weather"].map(weather_map).fillna(0).astype(int)

    # ── 6. Drop raw columns ────────────────────────────────────────────────────
    drop = [c for c in DROP_COLS + ["category", "weather"] if c in df.columns]
    df = df.drop(columns=drop)

    logger.info(f"Feature engineering done. Shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return feature columns (everything except target)."""
    return [c for c in df.columns if c != TARGET]


def correlation_report(df: pd.DataFrame) -> pd.Series:
    """Pearson correlation of all features with target."""
    feats = get_feature_columns(df)
    corr  = df[feats + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    logger.info("Top correlations with ETA:\n" + corr.head(10).to_string())
    return corr


if __name__ == "__main__":
    from data_processing import generate_dataset, clean
    raw = clean(generate_dataset(1000))
    eng = build_features(raw)
    print(eng.head())
    print("\nFeatures:", get_feature_columns(eng))
    print("\nCorrelations:\n", correlation_report(eng))
