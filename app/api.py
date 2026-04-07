"""
extend this file for serving the HTTP endpoint
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import PredictRequestLog, SessionLocal, get_db, init_db
from app.training import FEATURE_COLUMNS, train_and_save

logger = logging.getLogger(__name__)

app = FastAPI()


@app.on_event("startup")
def startup():
    init_db()
    model, model_version = _load_or_train_model()
    app.state.model = model
    app.state.model_version = model_version
    if model_version is not None:
        print(f"model_version={model_version}")


def _load_or_train_model() -> tuple[Any | None, str | None]:
    """
    Load the model artifact from MODEL_DIR.  If the artifact is missing, train
    a fresh model so the system stays self-contained for the coding challenge.
    Returns (model, model_version) — never raises; sets both to None on failure
    and raises RuntimeError so startup propagates the error clearly.
    """
    model_dir = Path(os.environ.get("MODEL_DIR", "./models/current"))
    model_path = model_dir / "model.joblib"
    meta_path = model_dir / "metadata.json"

    if not model_path.exists() or not meta_path.exists():
        # For the coding challenge, keep the system self-contained:
        # training is quick and avoids requiring a manual pre-step.
        train_and_save(
            sqlite_path=os.environ.get("SQLITE_PATH", "./database.sqlite"),
            model_dir=str(model_dir),
            random_seed=int(os.environ.get("TRAIN_RANDOM_SEED", "7")),
        )

    try:
        from joblib import load

        model = load(model_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_version = str(
            meta.get("model_version") or (model_dir / "VERSION").read_text().strip()
        )
        return model, model_version
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifact from {model_dir}") from e


class PredictRequest(BaseModel):
    id: int
    recency_7: int = Field(..., ge=1, le=7)
    frequency_7: int = Field(..., ge=0)
    monetary_7: float = Field(..., ge=0)


class PredictResponse(BaseModel):
    id: int
    monetary_30: float


class CountResponse(BaseModel):
    id: int
    count: int


@app.get("/health", response_model=str)
async def get_health():
    """
    check for API health
    """
    return "healthy"


@app.post("/api/predict", response_model=PredictResponse)
async def predict_monetary(
    http_request: Request,
    request_payload: PredictRequest,
    sessions: Session = Depends(get_db),
):
    """
    * predict monetary value for next 30 days
    * track passenger information in dedicated table
    """
    model = http_request.app.state.model
    model_version = http_request.app.state.model_version

    if model is None or model_version is None:
        raise HTTPException(status_code=503, detail="Model not available")

    # Pass a named DataFrame so sklearn doesn't warn about missing feature names
    # and the feature order is explicit rather than positional.
    X = pd.DataFrame(
        [[request_payload.recency_7, request_payload.frequency_7, request_payload.monetary_7]],
        columns=FEATURE_COLUMNS,
    )
    pred = float(model.predict(X)[0])
    pred = max(0.0, pred)

    logger.info(
        "predict ok passenger_id=%s model_version=%s monetary_30=%s",
        request_payload.id,
        model_version,
        pred,
    )

    sessions.add(
        PredictRequestLog(
            passenger_id=request_payload.id,
            recency_7=request_payload.recency_7,
            frequency_7=request_payload.frequency_7,
            monetary_7=request_payload.monetary_7,
            prediction_monetary_30=pred,
            model_version=model_version,
        )
    )
    sessions.commit()

    return PredictResponse(id=request_payload.id, monetary_30=pred)


@app.get("/api/requests/{passenger_id}", response_model=CountResponse)
async def count_number_of_requests(
    passenger_id: int, sessions: Session = Depends(get_db)
):
    """
    * get the number of times this passenger id requested a prediction
    """
    count = (
        sessions.query(PredictRequestLog)
        .filter(PredictRequestLog.passenger_id == passenger_id)
        .count()
    )
    return CountResponse(id=passenger_id, count=int(count))
