"""
Data Drift Detection
Compares training distribution vs recent inference requests.
Uses statistical tests (KS-test) and optional Evidently report.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger
from monitoring.tracker import load_predictions

DRIFT_THRESHOLD = 0.05   # p-value threshold for KS test
REPORT_DIR      = "data/drift_reports"


def extract_inference_df() -> pd.DataFrame:
    """Parse logged predictions into a DataFrame of features."""
    records = load_predictions()
    if not records:
        logger.warning("No predictions logged yet.")
        return pd.DataFrame()

    rows = []
    for r in records:
        feat = json.loads(r["features"])
        feat["predicted"] = r["predicted"]
        rows.append(feat)
    return pd.DataFrame(rows)


def ks_drift_report(train_df: pd.DataFrame, inference_df: pd.DataFrame) -> dict:
    """
    Run KS test on numeric columns.
    Returns dict: {column: {"statistic": float, "p_value": float, "drifted": bool}}
    """
    numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
    report = {}

    for col in numeric_cols:
        if col not in inference_df.columns:
            continue
        stat, p = stats.ks_2samp(
            train_df[col].dropna().values,
            inference_df[col].dropna().values
        )
        drifted = p < DRIFT_THRESHOLD
        report[col] = {"statistic": round(float(stat), 4), "p_value": round(float(p), 4), "drifted": bool(drifted)}
        if drifted:
            logger.warning(f"DRIFT detected in '{col}': KS={stat:.4f}, p={p:.4f}")

    drifted_cols = [k for k, v in report.items() if v["drifted"]]
    logger.info(f"Drift check: {len(drifted_cols)}/{len(report)} features drifted")
    return report


def run_drift_check() -> dict:
    """Full drift check: load training data + inference logs, run KS tests."""
    from data_processing import load_from_sql, clean
    from feature_engineering import build_features

    train_raw  = clean(load_from_sql())
    train_eng  = build_features(train_raw)

    inf_df = extract_inference_df()
    if inf_df.empty:
        return {"error": "No inference data available"}

    report = ks_drift_report(train_eng, inf_df)

    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = f"{REPORT_DIR}/drift_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Drift report saved → {out_path}")
    return report


def should_retrain(drift_report: dict, max_drifted_ratio: float = 0.3) -> bool:
    """Return True if more than max_drifted_ratio of features have drifted."""
    if not drift_report or "error" in drift_report:
        return False
    total   = len(drift_report)
    drifted = sum(1 for v in drift_report.values() if v.get("drifted"))
    ratio   = drifted / total if total > 0 else 0
    logger.info(f"Drift ratio: {ratio:.2%} (threshold: {max_drifted_ratio:.2%})")
    return ratio >= max_drifted_ratio


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    report = run_drift_check()
    print(json.dumps(report, indent=2))
    print(f"\nShould retrain: {should_retrain(report)}")
