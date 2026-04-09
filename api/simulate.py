"""
Real-time Request Simulator
Sends N random orders to the FastAPI /predict endpoint and reports latency.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time
import random
import httpx
import numpy as np
from loguru import logger

API_URL    = "http://localhost:8080/predict"
N_REQUESTS = 50

CATEGORIES = ["Groceries", "Snacks", "Dairy", "Beverages", "Personal Care", "Frozen"]
WEATHER    = ["Clear", "Cloudy", "Rainy", "Heavy Rain"]


def random_order() -> dict:
    return {
        "order_lat":     round(random.uniform(12.85, 13.10), 5),
        "order_lon":     round(random.uniform(77.45, 77.75), 5),
        "wh_lat":        round(random.uniform(12.85, 13.10), 5),
        "wh_lon":        round(random.uniform(77.45, 77.75), 5),
        "hour":          random.randint(0, 23),
        "weekday":       random.randint(0, 6),
        "item_count":    random.randint(1, 20),
        "category":      random.choice(CATEGORIES),
        "weather":       random.choice(WEATHER),
        "traffic_index": round(random.uniform(0.1, 1.0), 2),
    }


def simulate(n: int = N_REQUESTS) -> None:
    latencies = []
    errors    = 0

    logger.info(f"Sending {n} requests to {API_URL}")

    with httpx.Client(timeout=10.0) as client:
        for i in range(n):
            order = random_order()
            t0 = time.perf_counter()
            try:
                resp = client.post(API_URL, json=order)
                resp.raise_for_status()
                data = resp.json()
                latency_ms = (time.perf_counter() - t0) * 1000
                latencies.append(latency_ms)
                logger.info(
                    f"[{i+1:03d}] ETA={data['eta_minutes']}min  "
                    f"conf={data['confidence']}  latency={latency_ms:.1f}ms"
                )
            except Exception as e:
                errors += 1
                logger.error(f"[{i+1:03d}] Error: {e}")

            time.sleep(0.05)   # 50ms between requests

    if latencies:
        print(f"\n{'='*40}")
        print(f"  Simulation Results ({n} requests)")
        print(f"{'='*40}")
        print(f"  Success rate : {(n-errors)/n*100:.1f}%")
        print(f"  Avg latency  : {np.mean(latencies):.1f} ms")
        print(f"  P95 latency  : {np.percentile(latencies, 95):.1f} ms")
        print(f"  P99 latency  : {np.percentile(latencies, 99):.1f} ms")
        print(f"  Max latency  : {np.max(latencies):.1f} ms")
        print(f"{'='*40}")


if __name__ == "__main__":
    simulate()
