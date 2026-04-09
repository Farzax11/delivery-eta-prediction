"""
Retraining Pipeline
Checks drift, decides if retraining is needed, triggers training if so.
Can be run as a cron job or scheduled task.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from loguru import logger
from monitoring.drift_detector import run_drift_check, should_retrain


def retrain_if_needed(force: bool = False) -> bool:
    """
    1. Run drift detection
    2. If drift exceeds threshold (or force=True), retrain
    3. Return True if retraining happened
    """
    logger.info("=== Retraining Pipeline Started ===")

    drift_report = run_drift_check()

    if force or should_retrain(drift_report):
        logger.warning("Retraining triggered.")
        from train import train
        best = train(regenerate=False)   # use existing data, just retrain
        logger.success(f"Retraining complete. Best model: {best}")
        return True
    else:
        logger.info("No retraining needed.")
        return False


if __name__ == "__main__":
    force = "--force" in sys.argv
    retrain_if_needed(force=force)
