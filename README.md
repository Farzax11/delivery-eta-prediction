# Delivery Time Prediction System (ETA Prediction)

A production-style ML system that predicts delivery ETA for quick-commerce orders,
modelled after real-world platforms like Zepto / Blinkit.

---

## Problem Statement

Quick-commerce promises 10–30 minute deliveries. Accurate ETA prediction is critical for:
- Customer trust and satisfaction
- Rider dispatch optimization
- SLA breach prevention

This system predicts ETA (in minutes) given order features, location, time, and environmental conditions.

---

## Project Structure

```
delivery-eta/
├── src/
│   ├── data_processing.py      # Data generation, cleaning, SQL I/O
│   ├── feature_engineering.py  # Feature transforms, encoding
│   ├── train.py                # Multi-model training + CV + artifact saving
│   └── evaluate.py             # Metrics, residual plots, feature importance
├── api/
│   ├── app.py                  # FastAPI prediction service
│   └── simulate.py             # Load simulator (N concurrent requests)
├── monitoring/
│   ├── tracker.py              # Log predictions to SQLite
│   ├── drift_detector.py       # KS-test based drift detection
│   └── retrain_pipeline.py     # Auto-retrain trigger
├── sql/
│   └── setup.py                # SQL views + analytical queries
├── dashboard/
│   └── app.py                  # Streamlit monitoring dashboard
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── data/                       # Generated at runtime
├── models/                     # Saved model artifacts
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## ML Approach

### Models Trained
| Model             | Type          |
|-------------------|---------------|
| Linear Regression | Baseline      |
| Ridge             | Regularized   |
| Lasso             | Regularized   |
| Random Forest     | Tree ensemble |
| XGBoost           | Gradient boost|
| LightGBM          | Gradient boost|

### Features
- `distance_km` — Haversine distance between order and warehouse
- `hour`, `weekday` — Cyclically encoded time features
- `is_peak_hour`, `is_weekend` — Derived time flags
- `traffic_index` — Simulated real-time traffic (0–1)
- `item_count`, `log_item_count` — Order size
- `weather_enc` — Encoded weather condition
- `traffic_x_distance`, `peak_x_traffic` — Interaction features

### Evaluation
- MAE, RMSE, R² via 5-fold cross-validation
- Best model selected by lowest MAE
- Residual analysis + feature importance plots saved to `data/plots/`

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models
```bash
python src/train.py
```
This generates data, saves to SQLite, trains all 6 models, and saves the best to `models/`.

### 3. Run SQL analytics
```bash
python sql/setup.py
```

### 4. Evaluate all models
```bash
python src/evaluate.py
```
Plots saved to `data/plots/`.

### 5. Start the API
```bash
uvicorn api.app:app --reload --port 8000
```

### 6. Test a prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "order_lat": 12.97, "order_lon": 77.59,
    "wh_lat": 12.93,    "wh_lon": 77.55,
    "hour": 19, "weekday": 4,
    "item_count": 5, "category": "Groceries",
    "weather": "Clear", "traffic_index": 0.65
  }'
```

### 7. Simulate load
```bash
python api/simulate.py
```

### 8. Check drift
```bash
python monitoring/drift_detector.py
```

### 9. Retrain if needed
```bash
python monitoring/retrain_pipeline.py
# Force retrain:
python monitoring/retrain_pipeline.py --force
```

### 10. Launch dashboard
```bash
streamlit run dashboard/app.py
```

---

## Docker

```bash
docker-compose up --build
```
- API:       http://localhost:8000
- Dashboard: http://localhost:8501
- Docs:      http://localhost:8000/docs

---

## Results (typical on 10k synthetic samples)

| Model          | MAE (min) | RMSE (min) | R²    |
|----------------|-----------|------------|-------|
| Linear Reg.    | ~2.1      | ~2.7       | ~0.91 |
| Ridge          | ~2.1      | ~2.7       | ~0.91 |
| Lasso          | ~2.2      | ~2.8       | ~0.90 |
| Random Forest  | ~1.4      | ~1.9       | ~0.96 |
| XGBoost        | ~1.3      | ~1.7       | ~0.97 |
| LightGBM       | ~1.3      | ~1.7       | ~0.97 |

---

## Monitoring

Every `/predict` call is logged to `data/delivery.db` (predictions table).
Drift detection uses the KS test — if >30% of features drift, retraining is triggered.

---

## Future Improvements

- Real GPS data integration (Google Maps Distance Matrix API)
- Live traffic feed (HERE / TomTom API)
- Rider availability as a feature
- Multi-city model with city embeddings
- Online learning for continuous adaptation
- A/B testing framework for model rollouts
- Kubernetes deployment with autoscaling
