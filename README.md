# Delivery Time Prediction System (ETA Prediction)

A production-style machine learning system that predicts delivery ETA for quick-commerce orders, modelled after real-world platforms like Zepto and Blinkit.

---

## Live URLs

| What | URL |
|------|-----|
| Dashboard + Prediction | https://delivery-eta-prediction.streamlit.app |
| API Docs (Swagger UI) | https://delivery-eta-prediction.onrender.com/docs |
| API Health Check | https://delivery-eta-prediction.onrender.com/health |
| GitHub Repo | https://github.com/Farzax11/delivery-eta-prediction |

---

## Before You Demo

The API runs on Render free tier and sleeps after 15 minutes of inactivity.

**Always do this first:**
1. Open https://delivery-eta-prediction.onrender.com/health in your browser
2. Wait until you see `{"status":"ok","model_loaded":true}`
3. Then open the dashboard and predict

This takes about 30 seconds on a cold start.

---

## What to Show

**For a general demo** — open the dashboard:
- Go to "Predict ETA" in the sidebar
- Fill in order details and click Predict
- Shows predicted delivery time with confidence level

**For a technical interview** — open the API docs:
- Go to POST /predict → "Try it out"
- Fill in the JSON and click Execute
- Shows live prediction with model name and confidence

---

## Problem Statement

Quick-commerce promises 10–30 minute deliveries. Accurate ETA prediction is critical for customer trust, rider dispatch, and SLA management. This system predicts delivery time in minutes given order location, time, weather, and traffic conditions.

---

## ML Approach

- 6 models trained: Linear Regression, Ridge, Lasso, Random Forest, XGBoost, LightGBM
- 5-fold cross-validation for model selection
- Best model: LightGBM (CV MAE ~1.39 min, R² ~0.97)
- Features: haversine distance, cyclical time encoding, traffic index, weather, order size

---

## Project Structure

```
├── src/
│   ├── data_processing.py      # Data generation, cleaning, SQL storage
│   ├── feature_engineering.py  # Feature transforms and encoding
│   ├── train.py                # Train all 6 models, save best
│   └── evaluate.py             # Metrics, residual plots, feature importance
├── api/
│   ├── app.py                  # FastAPI prediction service
│   └── simulate.py             # Load simulator
├── monitoring/
│   ├── tracker.py              # Log predictions to SQLite
│   ├── drift_detector.py       # KS-test drift detection
│   └── retrain_pipeline.py     # Auto-retrain trigger
├── sql/
│   └── setup.py                # SQL views and analytical queries
├── streamlit_app.py            # Combined dashboard + prediction UI
├── Dockerfile
└── requirements.txt
```

---

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python src/train.py

# Start API
uvicorn api.app:app --reload --port 8080

# Start dashboard
streamlit run streamlit_app.py

# Simulate requests
python api/simulate.py

# Check drift
python monitoring/drift_detector.py
```

---

## Results

| Model | MAE (min) | RMSE (min) | R² |
|-------|-----------|------------|-----|
| Linear Regression | ~2.1 | ~2.7 | ~0.91 |
| Ridge | ~2.1 | ~2.7 | ~0.91 |
| Lasso | ~2.2 | ~2.8 | ~0.90 |
| Random Forest | ~1.4 | ~1.9 | ~0.96 |
| XGBoost | ~1.3 | ~1.7 | ~0.97 |
| LightGBM | ~1.4 | ~1.8 | ~0.97 |

---

## Future Improvements

- Real GPS data via Google Maps Distance Matrix API
- Live traffic feed integration
- Rider availability as a feature
- Online learning for continuous model adaptation
- A/B testing framework for model rollouts
