"""
Evaluation & Error Analysis
Metrics, residual analysis, model comparison, feature importance.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PLOTS_DIR  = "data/plots"
MODELS_DIR = "models"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def residual_analysis(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "model") -> None:
    """Plot residuals and prediction vs actual."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"Residual Analysis — {model_name}", fontsize=13)

    # Predicted vs Actual
    axes[0].scatter(y_pred, y_true, alpha=0.3, s=10, color="steelblue")
    mn, mx = min(y_pred.min(), y_true.min()), max(y_pred.max(), y_true.max())
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=1.5)
    axes[0].set_xlabel("Predicted ETA (min)")
    axes[0].set_ylabel("Actual ETA (min)")
    axes[0].set_title("Predicted vs Actual")

    # Residual distribution
    axes[1].hist(residuals, bins=50, color="coral", edgecolor="white")
    axes[1].axvline(0, color="black", lw=1.5, linestyle="--")
    axes[1].set_xlabel("Residual (min)")
    axes[1].set_title("Residual Distribution")

    # Residuals vs Predicted
    axes[2].scatter(y_pred, residuals, alpha=0.3, s=10, color="seagreen")
    axes[2].axhline(0, color="red", lw=1.5, linestyle="--")
    axes[2].set_xlabel("Predicted ETA (min)")
    axes[2].set_ylabel("Residual")
    axes[2].set_title("Residuals vs Predicted")

    plt.tight_layout()
    path = f"{PLOTS_DIR}/residuals_{model_name}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info(f"Residual plot saved → {path}")


def feature_importance_plot(model, feature_cols: list, model_name: str = "model") -> None:
    """Plot feature importances for tree-based models."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Handle sklearn Pipeline
    m = model.named_steps["model"] if hasattr(model, "named_steps") else model

    if hasattr(m, "feature_importances_"):
        importances = m.feature_importances_
    elif hasattr(m, "coef_"):
        importances = np.abs(m.coef_)
    else:
        logger.warning(f"No importances available for {model_name}")
        return

    idx  = np.argsort(importances)[::-1]
    cols = [feature_cols[i] for i in idx]
    vals = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=vals[:15], y=cols[:15], palette="viridis", ax=ax)
    ax.set_title(f"Feature Importance — {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/importance_{model_name}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info(f"Importance plot saved → {path}")


def compare_models(results: dict) -> pd.DataFrame:
    """Pretty-print model comparison table."""
    df = pd.DataFrame(results).T.sort_values("mae")
    logger.info("\nModel Comparison:\n" + df.to_string())
    return df


def run_full_evaluation() -> None:
    """Load all trained models and evaluate on held-out test set."""
    from data_processing import load_from_sql, clean
    from feature_engineering import build_features, get_feature_columns
    from sklearn.model_selection import train_test_split

    raw  = clean(load_from_sql())
    eng  = build_features(raw)
    feat = get_feature_columns(eng)
    X, y = eng[feat].values, eng["eta_minutes"].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with open(f"{MODELS_DIR}/metadata.json") as f:
        meta = json.load(f)

    model_names = list(meta["cv_results"].keys())
    all_results = {}

    for name in model_names:
        path = f"{MODELS_DIR}/{name}.pkl"
        if not os.path.exists(path):
            continue
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        all_results[name] = metrics
        residual_analysis(y_test, y_pred, name)
        feature_importance_plot(model, feat, name)
        logger.info(f"{name}: {metrics}")

    compare_models(all_results)

    # Save comparison
    os.makedirs(PLOTS_DIR, exist_ok=True)
    pd.DataFrame(all_results).T.to_csv(f"{PLOTS_DIR}/model_comparison.csv")
    logger.success("Evaluation complete.")


if __name__ == "__main__":
    run_full_evaluation()
