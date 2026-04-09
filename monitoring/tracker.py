"""
Prediction Tracker
Logs every prediction (features + predicted ETA) to SQLite for monitoring.
"""

import os
import json
from datetime import datetime
from sqlalchemy import create_engine, text
from loguru import logger

DB_PATH = "data/delivery.db"


def _get_engine():
    os.makedirs("data", exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}")


def _ensure_table():
    engine = _get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT,
                features    TEXT,
                predicted   REAL,
                actual      REAL
            )
        """))
        conn.commit()


def log_prediction(features: dict, predicted: float, actual: float = None) -> None:
    """Insert a prediction record."""
    _ensure_table()
    engine = _get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO predictions (timestamp, features, predicted, actual)
            VALUES (:ts, :feat, :pred, :actual)
        """), {
            "ts":     datetime.utcnow().isoformat(),
            "feat":   json.dumps(features),
            "pred":   predicted,
            "actual": actual,
        })
        conn.commit()
    logger.debug(f"Logged prediction: {predicted:.1f} min")


def load_predictions() -> list:
    """Return all logged predictions as list of dicts."""
    _ensure_table()
    engine = _get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT * FROM predictions ORDER BY id DESC")).fetchall()
    return [dict(r._mapping) for r in rows]


def update_actual(prediction_id: int, actual: float) -> None:
    """Update actual ETA once delivery is complete (feedback loop)."""
    engine = _get_engine()
    with engine.connect() as conn:
        conn.execute(text(
            "UPDATE predictions SET actual=:actual WHERE id=:id"
        ), {"actual": actual, "id": prediction_id})
        conn.commit()
    logger.info(f"Updated prediction #{prediction_id} with actual={actual}")
