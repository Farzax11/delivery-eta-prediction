"""
Data Processing Pipeline
Handles data generation, loading, cleaning, and SQL storage.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from loguru import logger
import os

# ── Constants ──────────────────────────────────────────────────────────────────
DB_PATH = "data/delivery.db"
RAW_CSV  = "data/raw_orders.csv"
SEED     = 42
N_ROWS   = 10_000

# Bangalore-area bounding box (realistic quick-commerce city)
LAT_MIN, LAT_MAX = 12.85, 13.10
LON_MIN, LON_MAX = 77.45, 77.75

CATEGORIES = ["Groceries", "Snacks", "Dairy", "Beverages", "Personal Care", "Frozen"]
WEATHER     = ["Clear", "Cloudy", "Rainy", "Heavy Rain"]


def generate_dataset(n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Simulate realistic quick-commerce order data."""
    rng = np.random.default_rng(seed)

    order_lat  = rng.uniform(LAT_MIN, LAT_MAX, n)
    order_lon  = rng.uniform(LON_MIN, LON_MAX, n)
    wh_lat     = rng.uniform(LAT_MIN, LAT_MAX, n)
    wh_lon     = rng.uniform(LON_MIN, LON_MAX, n)

    hour       = rng.integers(0, 24, n)
    weekday    = rng.integers(0, 7, n)
    item_count = rng.integers(1, 20, n)
    category   = rng.choice(CATEGORIES, n)
    weather    = rng.choice(WEATHER, n, p=[0.5, 0.25, 0.15, 0.10])

    # Traffic index: 0-1, higher during peak hours
    traffic = np.clip(
        0.3
        + 0.4 * ((hour >= 8) & (hour <= 10)).astype(float)
        + 0.4 * ((hour >= 18) & (hour <= 21)).astype(float)
        + rng.normal(0, 0.1, n),
        0, 1
    )

    # Haversine distance (km)
    distance_km = _haversine(order_lat, order_lon, wh_lat, wh_lon)

    # Weather delay factor
    weather_delay = np.where(weather == "Heavy Rain", 1.4,
                    np.where(weather == "Rainy",       1.2,
                    np.where(weather == "Cloudy",      1.05, 1.0)))

    # Ground-truth ETA (minutes) — realistic formula with noise
    eta = (
        5                                    # base prep time
        + distance_km * 3.5                  # travel component
        + traffic * 12                       # traffic penalty
        + item_count * 0.3                   # packing time
        + rng.normal(0, 2, n)               # noise
    ) * weather_delay

    eta = np.clip(eta, 5, 90).round(1)

    df = pd.DataFrame({
        "order_lat":   order_lat,
        "order_lon":   order_lon,
        "wh_lat":      wh_lat,
        "wh_lon":      wh_lon,
        "hour":        hour,
        "weekday":     weekday,
        "item_count":  item_count,
        "category":    category,
        "weather":     weather,
        "traffic_index": traffic.round(3),
        "distance_km": distance_km.round(3),
        "eta_minutes": eta,
    })

    logger.info(f"Generated {len(df)} rows. ETA range: {eta.min():.1f} – {eta.max():.1f} min")
    return df


def _haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorised haversine distance in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def save_to_csv(df: pd.DataFrame, path: str = RAW_CSV) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved CSV → {path}")


def save_to_sql(df: pd.DataFrame, db_path: str = DB_PATH) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql("orders", engine, if_exists="replace", index=False)
    logger.info(f"Saved {len(df)} rows → SQLite: {db_path}")


def load_from_sql(db_path: str = DB_PATH, query: str = "SELECT * FROM orders") -> pd.DataFrame:
    engine = create_engine(f"sqlite:///{db_path}")
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} rows from SQL")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop nulls, clip outliers."""
    before = len(df)
    df = df.dropna()
    df = df[df["eta_minutes"].between(5, 90)]
    df = df[df["distance_km"] > 0]
    logger.info(f"Cleaned: {before} → {len(df)} rows")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = generate_dataset()
    save_to_csv(df)
    save_to_sql(df)
    print(df.head())
    print(df.describe())
