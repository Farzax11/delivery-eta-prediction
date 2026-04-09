"""
SQL Setup & Utility Queries
Creates tables and demonstrates useful analytical queries.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from sqlalchemy import create_engine, text
from loguru import logger

DB_PATH = "data/delivery.db"


def get_engine():
    os.makedirs("data", exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}")


def setup_views(engine) -> None:
    """Create analytical SQL views."""
    with engine.connect() as conn:

        conn.execute(text("DROP VIEW IF EXISTS v_peak_hour_stats"))
        conn.execute(text("""
            CREATE VIEW v_peak_hour_stats AS
            SELECT
                hour,
                COUNT(*)                        AS order_count,
                ROUND(AVG(eta_minutes), 2)      AS avg_eta,
                ROUND(AVG(traffic_index), 3)    AS avg_traffic,
                ROUND(MIN(eta_minutes), 2)      AS min_eta,
                ROUND(MAX(eta_minutes), 2)      AS max_eta
            FROM orders
            GROUP BY hour
            ORDER BY hour
        """))

        conn.execute(text("DROP VIEW IF EXISTS v_weather_impact"))
        conn.execute(text("""
            CREATE VIEW v_weather_impact AS
            SELECT
                weather,
                COUNT(*)                        AS order_count,
                ROUND(AVG(eta_minutes), 2)      AS avg_eta,
                ROUND(AVG(distance_km), 3)      AS avg_distance
            FROM orders
            GROUP BY weather
            ORDER BY avg_eta DESC
        """))

        conn.execute(text("DROP VIEW IF EXISTS v_category_stats"))
        conn.execute(text("""
            CREATE VIEW v_category_stats AS
            SELECT
                category,
                COUNT(*)                        AS order_count,
                ROUND(AVG(eta_minutes), 2)      AS avg_eta,
                ROUND(AVG(item_count), 1)       AS avg_items
            FROM orders
            GROUP BY category
            ORDER BY avg_eta DESC
        """))

        conn.commit()
    logger.info("SQL views created.")


def run_sample_queries(engine) -> None:
    """Print results of key analytical queries."""
    queries = {
        "Peak Hour ETA Stats": "SELECT * FROM v_peak_hour_stats LIMIT 10",
        "Weather Impact":      "SELECT * FROM v_weather_impact",
        "Category Stats":      "SELECT * FROM v_category_stats",
        "Long Deliveries (>45 min)": """
            SELECT hour, weather, category, distance_km, eta_minutes
            FROM orders WHERE eta_minutes > 45
            ORDER BY eta_minutes DESC LIMIT 10
        """,
        "Avg ETA by Weekday": """
            SELECT weekday,
                   ROUND(AVG(eta_minutes),2) AS avg_eta,
                   COUNT(*) AS orders
            FROM orders GROUP BY weekday ORDER BY weekday
        """,
    }

    for title, q in queries.items():
        print(f"\n{'='*50}")
        print(f"  {title}")
        print('='*50)
        df = pd.read_sql(q, engine)
        print(df.to_string(index=False))


if __name__ == "__main__":
    engine = get_engine()
    setup_views(engine)
    run_sample_queries(engine)
