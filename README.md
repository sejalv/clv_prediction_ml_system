## Problem
Marketing wants an estimate of customer life-time value to determine user segments that are worth targeting in a campaign. As a proxy for CLV, the task is to predict `monetary_value_30`, i.e. the total money a passenger spends on trips within 30 days of registration, using only signals observable in their first 7 days of activity.

### Input (features available at inference time)
| Feature | Description |
|---|---|
| `recency_7` | Days since last trip in the first week (1–7) |
| `frequency_7` | Number of trips in the first week |
| `monetary_value_7` | Total spend in the first week |

### Output
**Prediction target:** `monetary_value_30` (continuous, regression)

### Success metrics
- **Offline ML metric (regression)**: MAE/RMSE on a validation split, or a held-out validation set
- **Business proxy**: ability to rank passengers/segments by predicted spend; focus on error distribution for high-value passengers (tail).
- **Tail sensitivity**: MAE computed on the **top 25% of passengers by actual `monetary_value_30`** (the high-value tail). Reported as `tail_metrics.mae_top25pct` in `metadata.json` alongside the Q75 threshold used.


## Architecture overview

The solution is structured in three independently runnable components, all wired together via `make`:

```
SQLite DB
   │
   ▼
app/training.py ──►  Artifacts: 
                        models/current/model.pkl
                        models/current/metadata.json
   │
   ▼
FastAPI service (port 8080):
                   POST /api/predict ──► predict monetary_value_30 + track request to DB
                   GET  /api/requests/{id} ──► request count per passenger
   │
   ▼
pytest test/test_api.py
```


## System design

### Prerequisites

- Python >= 3.10
- Poetry >= 1.0
- Docker + docker-compose
- `database.sqlite` sample database must be present in the project root (bundled in submission zip) to run training.

### Commands

```bash
make setup   # install dependencies (requires Poetry)
make train   # train model and save artifact to models/current/
make run     # start app in docker; FastAPI service on http://0.0.0.0:8080 (docs at http://0.0.0.0:8080/docs)
make test    # run API contract tests inside Docker
```

Note: If you do not install Poetry locally, you can still use `make run`, `make test`, and `make train`, where Docker images will install dependencies from `poetry.lock` during build.

---

## 1. Model Training
`app/training.py`

Uses a RidgeCV regression model (with StandardScaler) to automatically tune regularization, compared against a mean baseline (DummyRegressor). Features' coefficients are saved to `metadata.json` for explainability.

The training pipeline:
1. Loads data from `database.sqlite` (table `passenger_activity_after_registration`)
2. Contract filters: Uses only `recency_7`, `frequency_7`, `monetary_value_7` as features to avoid any lookahead leakage (optionally applying a golden slice for testing)
3. Split: Performs a reproducible 80/20 train/validation split (fixed `random_state=7`)
4. Trains baseline and RidgeCV models: evaluates both and raises a `UserWarning` if RidgeCV fails to beat the mean-baseline (data quality signal).
5. Evaluate both (overall + tail): Validates predictions are non-negative (clamps if necessary)
6. Saves the model artifacts to `models/current/`:
  - `model.joblib` (sklearn `Pipeline`: `StandardScaler` + `RidgeCV`)
  - `metadata.json`: DB row count and training timestamp are logged in here. It includes a `random_seed` and a sqlite fingerprint (`database.sqlite|bytes=…|mtime_utc=…`) to identify the snapshot cheaply. 

Run:
- `make train`
- Golden subset example: `TRAIN_GOLDEN_MAX_ROWS=600 make train` (also respects `TRAIN_RANDOM_SEED`, `SQLITE_PATH`, `MODEL_DIR` when set in your shell)
- NOTE: Re-running `make train` overwrites `models/current/` deterministically

### Baseline
- **Mean-predictor (DummyRegressor)** trained on the same split as the primary model. Both MAE and RMSE are recorded in `metadata.json` under `baseline_metrics`. `make train` prints a side-by-side comparison table at the end of the run, making the RidgeCV choice evidence-based. For example:
  ```
  --- Model vs Baseline (alpha=100.0) ---
  Metric            Baseline (mean)      RidgeCV   Δ (↓ better)
  --------------------------------------------------------------
  MAE                       34.0851      17.8985         +47.5%
  RMSE                      66.3405      43.1215         +35.0%
  MAE (top 25%)             68.2236      46.7394         +31.5%

  (top-25% threshold: monetary_value_30 >= 49.62)
  ```

---

## 2. Model Serving 
`POST /api/predict`

### Request

```
POST /api/predict
Content-Type: application/json

{
  "id": 1234,
  "recency_7": 1,
  "frequency_7": 3,
  "monetary_7": 24.5
}
```

- Note: DB uses `monetary_value_7`; API uses `monetary_7` and maps it to the trained feature order.

### Response

```json
{
  "passenger_id": 1234,
  "predicted_monetary_value_30": 87.3,
  "model_version": "1"
}
```

### Input validation

Pydantic enforces:
- `recency_7` ∈ [1, 7]
- `frequency_7` >= 0
- `monetary_7` >= 0

Invalid requests return HTTP 422 with a structured error.

### Model management

- **Artifact format**:
  - Save a single on-disk artifact (e.g., `joblib`) plus a small metadata JSON (features list, training timestamp, metrics, version).
- **Model loading**:
  - The API loads the model from `MODEL_DIR` (default `./models/current`). If missing, it trains quickly on startup to keep the challenge self-contained.
- **To swap models:** 
  - Replace the files in `models/current/` and restart the service (or set `MODEL_PATH` to a new directory).

---

## 3. Request Tracking

### Schema

A `predict_requests` table is created automatically on startup in `database.sqlite`:

```sql
CREATE TABLE IF NOT EXISTS predict_requests (
  id                     INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at             DATETIME NOT NULL,
  passenger_id           INTEGER NOT NULL,
  recency_7              INTEGER NOT NULL,
  frequency_7            INTEGER NOT NULL,
  monetary_7             FLOAT NOT NULL,
  prediction_monetary_30 FLOAT NOT NULL,
  model_version          VARCHAR NOT NULL
);
```

Every call to `POST /api/predict` inserts a row synchronously before returning the response.

### Request count endpoint

```http
GET /api/requests/1234
```

**Response:**
```json
{
  "id": 1234,
  "count": 7
}
```

The app enables **SQLite WAL** on the SQLAlchemy engine to reduce write lock issues under light concurrency.

### Data Contracts

- Training uses table `passenger_activity_after_registration` with columns `recency_7`, `frequency_7`, `monetary_value_7`, `monetary_value_30` (and `id`).
- Invalid ranges (after dropna): rows are kept only if `recency_7 ∈ [1, 7]`, `frequency_7 ≥ 0`, `monetary_value_7 ≥ 0`, `monetary_value_30 ≥ 0`.
- Inference validates the same ranges via Pydantic (`recency_7`, `frequency_7`, `monetary_7`).
- Missing values: Any row with NA in the feature columns or target is dropped (no imputation in MVP).

---

## Security & rollout
- **PII / data**: Request payloads include a passenger `id` and trip aggregates; treat logs and SQLite as sensitive. Production would use access control, retention limits, and no secrets in the repo.
- **Exposure**: Endpoints are unauthenticated in this challenge; production would sit behind an API gateway (authn/z, rate limits).
- **Rollout**: Use offline validation before campaign use; ship behind a feature flag or “shadow” logging first, then promote after business sign-off on error/tail behavior.
- **Backtest**: evaluate on held-out rows in SQLite; document performance and limitations (small feature set, short horizon).


## Limitations

- SQLite concurrency: SQLite has write-locking constraints under concurrent load, as training data (`passenger_activity_after_registration`) and request logs (`predict_requests`) live in the same `database.sqlite`. Request tracking under high QPS would require a proper RDBMS.
- No authentication: Endpoints are unauthenticated, production would sit behind an API gateway with authn/z and rate limits.
- No temporal split: Validation uses a random 80/20 split. A time-ordered split (earlier IDs train, later IDs validate) would better simulate deployment conditions.
- Small feature set: Only 3 features derived from a single week of activity. Predictive power is inherently limited; the model should be evaluated against a business threshold before use in campaign decisions.
- Short horizon: 30-day spend is a useful CLV proxy but not a true LTV. Passengers who churn within 30 days and those who become long-term regulars may look identical at day 7.
- Limited monitoring beyond request tracking and basic guardrails: The model will silently degrade if passenger behaviour patterns shift (seasonality, product changes, market expansion). Add data drift checks and delayed-label performance monitoring.


## Production-grade notes

**Future enhancements:**
- Hot-reload model safely (atomic swaps + lock) or use a registry.
- Explicit dataset snapshot IDs and golden set in CI.
- Model selection: Extending to gradient boosting (e.g. LightGBM, XGBoost) using SHAP values for global and local explainability, mapping closely to business stakeholders.

**Production Evolution Path:**

1. Feature engineering: Add derived features (e.g. average trip value in week 1, trip frequency trend) and eventually hook into a real-time feature store.
2. Experiment tracking: MLflow for every training run, to track parameters, metrics, artifacts, and data lineage.
3. Model registry: Promote models through `staging → production` gates with automated evaluation checks.
4. Offline scoring: For marketing campaigns, pre-score all active passengers nightly and write results to a scoring table rather than scoring on demand.
5. Online serving at scale: Move from FastAPI on a single container to a model serving platform (e.g. BentoML, Seldon, or SageMaker endpoints) behind a load balancer.
6. Drift monitoring: Track input feature distributions and prediction distributions over time. Add SLOs (p95 latency) and structured logs. Alert and trigger retraining when drift exceeds a threshold.
7. Retraining pipeline: Airflow DAG (or any other orchestrator) on a weekly schedule (or drift-triggered) that runs `training.py`, evaluates against a golden dataset, and promotes the model if it passes evaluation gates.
8. A/B testing: Shadow mode first (log predictions without serving), then a proper traffic split between model versions.


## Runbook

- Model file missing on startup:
If `/api/predict` returns **503 ("Model not available")**: run `make train` to regenerate the artifact, then restart the app (`make run` or `docker-compose restart app`). The `model_version` in `metadata.json` and in the request log will update automatically.

- Model file corrupt:
`joblib` will raise a deserialization error on startup. Same fix as above. Consider keeping the previous version in `models/previous/` as a fallback.

- If `SQLite` is locked: WAL mode is enabled for the ORM DB; for heavy write load, move request logging to a separate database or scale the store. Do not run `make train` while `make run` is serving heavy traffic, both processes share the same SQLite file.
