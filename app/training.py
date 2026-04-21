"""
use this file to implement a simple training pipeline
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["recency_7", "frequency_7", "monetary_value_7"]
TARGET_COLUMN = "monetary_value_30"
TABLE_NAME = "passenger_activity_after_registration"

# Saved artifact is a sklearn Pipeline(StandardScaler, RidgeCV); this label is for metadata only.
MODEL_TYPE = "RidgeCV"

# Regularization strengths searched by cross-validation on the training fold.
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


@dataclass(frozen=True)
class TrainingMetadata:
    model_version: str
    trained_at: str
    model_type: str
    feature_columns: list[str]
    target_column: str
    row_count: int
    train_size: int
    val_size: int
    random_seed: int
    best_alpha: float
    coefficients: dict[str, float]
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    tail_metrics: dict[str, float]
    baseline_tail_metrics: dict[str, float]
    sqlite_fingerprint: str
    training_subset: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sqlite_fingerprint(sqlite_path: str) -> str:
    """Cheap fingerprint of the DB file for metadata (not a cryptographic guarantee)."""
    path = Path(sqlite_path)
    st = path.stat()
    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    return f"{path.name}|bytes={st.st_size}|mtime_utc={mtime}"


def apply_training_data_contract(
    df: pd.DataFrame,
    *,
    golden_max_rows: Optional[int],
) -> tuple[pd.DataFrame, str]:
    """
    Drop invalid rows (see README.md data contract), then optionally keep the first N rows
    by `id` after sorting — deterministic ordering for golden / smoke training.
    """
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    df = df[df["recency_7"].between(1, 7)]
    df = df[df["frequency_7"] >= 0]
    df = df[df["monetary_value_7"] >= 0]
    df = df[df[TARGET_COLUMN] >= 0]

    if golden_max_rows is None:
        return df, "full"

    if "id" not in df.columns:
        raise ValueError("Golden subset requires an `id` column in the training table")

    df = df.sort_values("id", kind="mergesort").head(golden_max_rows)
    return df, f"golden_first_{golden_max_rows}_by_id"


def load_training_frame(sqlite_path: str = "./database.sqlite") -> pd.DataFrame:
    # Use sqlite3 + read_sql_query so we never depend on pandas/SQLAlchemy Engine integration
    # (pandas 2.2 + some SQLAlchemy builds mis-wrap Engine and call .cursor() on it).
    conn = sqlite3.connect(sqlite_path)
    try:
        return pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    finally:
        conn.close()


def train_and_save(
    *,
    sqlite_path: str = "./database.sqlite",
    model_dir: str = "./models/current",
    random_seed: int = 7,
    golden_max_rows: Optional[int] = None,
) -> TrainingMetadata:
    df = load_training_frame(sqlite_path=sqlite_path)

    missing = [c for c in (FEATURE_COLUMNS + [TARGET_COLUMN]) if c not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")

    df, training_subset = apply_training_data_contract(df, golden_max_rows=golden_max_rows)
    fp = sqlite_fingerprint(sqlite_path)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # --- Baseline: mean-predictor (DummyRegressor) ---
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_val).clip(min=0.0)
    baseline_mae = float(mean_absolute_error(y_val, baseline_preds))
    baseline_rmse = float(math.sqrt(mean_squared_error(y_val, baseline_preds)))

    # --- Primary model: RidgeCV with standard scaling ---
    # RidgeCV searches across RIDGE_ALPHAS using leave-one-out CV on the training fold,
    # selecting the regularization strength that minimises in-fold MSE automatically.
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=RIDGE_ALPHAS)),
        ]
    )
    pipeline.fit(X_train, y_train)

    ridge = pipeline.named_steps["model"]
    best_alpha = float(ridge.alpha_)
    # Surface coefficients for marketing explainability (pre-scaling, so units are comparable).
    coefficients = {feat: float(coef) for feat, coef in zip(FEATURE_COLUMNS, ridge.coef_)}

    preds = pipeline.predict(X_val).clip(min=0.0)

    mae = float(mean_absolute_error(y_val, preds))
    # Avoid `squared=` — not supported on some sklearn builds resolved in Docker.
    rmse = float(math.sqrt(mean_squared_error(y_val, preds)))

    # --- Tail-sensitivity: MAE on top-25% passengers by actual spend ---
    # Marketing cares most about high-value passengers; aggregate MAE can hide
    # large errors in the tail.  We report MAE on the top quartile separately.
    y_val_arr = np.asarray(y_val)
    q75 = float(np.percentile(y_val_arr, 75))
    tail_mask = y_val_arr >= q75
    if tail_mask.sum() > 0:
        tail_mae = float(mean_absolute_error(y_val_arr[tail_mask], preds[tail_mask]))
        baseline_tail_mae = float(
            mean_absolute_error(y_val_arr[tail_mask], baseline_preds[tail_mask])
        )
    else:
        tail_mae = float("nan")
        baseline_tail_mae = float("nan")

    # Sanity-check: trained model should beat the mean baseline.
    if mae >= baseline_mae:
        import warnings

        warnings.warn(
            f"RidgeCV MAE ({mae:.4f}) did not improve over mean-baseline MAE ({baseline_mae:.4f}). "
            "Check data quality or feature set.",
            UserWarning,
            stacklevel=2,
        )

    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    metadata_path = out_dir / "metadata.json"

    dump(pipeline, model_path)

    metadata = TrainingMetadata(
        model_version=model_version,
        trained_at=_utc_now_iso(),
        model_type=MODEL_TYPE,
        feature_columns=list(FEATURE_COLUMNS),
        target_column=TARGET_COLUMN,
        row_count=int(df.shape[0]),
        train_size=int(X_train.shape[0]),
        val_size=int(X_val.shape[0]),
        random_seed=int(random_seed),
        best_alpha=best_alpha,
        coefficients=coefficients,
        metrics={"mae": mae, "rmse": rmse},
        baseline_metrics={"mae": baseline_mae, "rmse": baseline_rmse},
        tail_metrics={"mae_top25pct": tail_mae, "q75_threshold": q75},
        baseline_tail_metrics={"mae_top25pct": baseline_tail_mae, "q75_threshold": q75},
        sqlite_fingerprint=fp,
        training_subset=training_subset,
    )

    metadata_path.write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")

    # Convenience: keep a stable "current version" marker for serving/logging.
    (out_dir / "VERSION").write_text(model_version + "\n", encoding="utf-8")

    return metadata


def main() -> None:
    sqlite_path = os.environ.get("SQLITE_PATH", "").strip() or "./database.sqlite"
    model_dir = os.environ.get("MODEL_DIR", "").strip() or "./models/current"
    rs = os.environ.get("TRAIN_RANDOM_SEED", "").strip()
    random_seed = int(rs) if rs else 7
    golden_raw = os.environ.get("TRAIN_GOLDEN_MAX_ROWS", "").strip()
    golden_max_rows = int(golden_raw) if golden_raw else None

    metadata = train_and_save(
        sqlite_path=sqlite_path,
        model_dir=model_dir,
        random_seed=random_seed,
        golden_max_rows=golden_max_rows,
    )

    print(json.dumps(asdict(metadata), indent=2))

    # Human-readable comparison table printed to stdout.
    b = metadata.baseline_metrics
    m = metadata.metrics
    bt = metadata.baseline_tail_metrics
    mt = metadata.tail_metrics
    mae_lift = (b["mae"] - m["mae"]) / b["mae"] * 100 if b["mae"] else 0.0
    rmse_lift = (b["rmse"] - m["rmse"]) / b["rmse"] * 100 if b["rmse"] else 0.0
    tail_lift = (
        (bt["mae_top25pct"] - mt["mae_top25pct"]) / bt["mae_top25pct"] * 100
        if bt["mae_top25pct"] and not math.isnan(bt["mae_top25pct"])
        else 0.0
    )
    print(f"\n--- Model vs Baseline (alpha={metadata.best_alpha}) ---")
    print(f"{'Metric':<14} {'Baseline (mean)':>18} {'RidgeCV':>12} {'Δ (↓ better)':>14}")
    print("-" * 62)
    print(f"{'MAE':<14} {b['mae']:>18.4f} {m['mae']:>12.4f} {mae_lift:>+13.1f}%")
    print(f"{'RMSE':<14} {b['rmse']:>18.4f} {m['rmse']:>12.4f} {rmse_lift:>+13.1f}%")
    tail_label = "MAE (top 25%)"
    print(
        f"{tail_label:<14} {bt['mae_top25pct']:>18.4f}"
        f" {mt['mae_top25pct']:>12.4f} {tail_lift:>+13.1f}%"
    )
    print(f"\n  (top-25% threshold: monetary_value_30 >= {mt['q75_threshold']:.2f})")
    print(f"\n  Coefficients: {metadata.coefficients}")


if __name__ == "__main__":
    main()
