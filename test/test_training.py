"""Training pipeline: quality gates, baseline comparison, and deterministic golden subset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.training import train_and_save


@pytest.fixture
def sqlite_path() -> str:
    return str(Path(__file__).resolve().parent.parent / "database.sqlite")


def test_golden_training_is_deterministic(sqlite_path: str, tmp_path: Path) -> None:
    """Same inputs → identical metrics and metadata (reproducibility gate)."""
    kwargs = dict(
        sqlite_path=sqlite_path,
        model_dir=str(tmp_path / "m1"),
        random_seed=7,
        golden_max_rows=600,
    )
    m1 = train_and_save(**kwargs)
    kwargs["model_dir"] = str(tmp_path / "m2")
    m2 = train_and_save(**kwargs)

    assert m1.row_count == m2.row_count <= 600
    assert m1.metrics == m2.metrics
    assert m1.baseline_metrics == m2.baseline_metrics
    
    # NaN != NaN in Python, so we check if it's NaN first before equating
    import math
    if math.isnan(m1.tail_metrics["mae_top25pct"]):
        assert math.isnan(m2.tail_metrics["mae_top25pct"])
    else:
        assert m1.tail_metrics == m2.tail_metrics
        
    if math.isnan(m1.baseline_tail_metrics["mae_top25pct"]):
        assert math.isnan(m2.baseline_tail_metrics["mae_top25pct"])
    else:
        assert m1.baseline_tail_metrics == m2.baseline_tail_metrics

    assert m1.training_subset == "golden_first_600_by_id"
    assert m1.model_type == "Ridge"
    assert "|mtime_utc=" in m1.sqlite_fingerprint
    assert m1.sqlite_fingerprint == m2.sqlite_fingerprint


def test_metadata_json_includes_fingerprint_and_subset(sqlite_path: str, tmp_path: Path) -> None:
    """metadata.json must contain all required fields for auditability."""
    out = tmp_path / "modeldir"
    train_and_save(
        sqlite_path=sqlite_path,
        model_dir=str(out),
        random_seed=7,
        golden_max_rows=400,
    )
    raw = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert raw["training_subset"] == "golden_first_400_by_id"
    assert raw["model_type"] == "Ridge"
    assert "sqlite_fingerprint" in raw and "mtime_utc" in raw["sqlite_fingerprint"]
    assert "baseline_metrics" in raw
    assert "mae" in raw["baseline_metrics"] and "rmse" in raw["baseline_metrics"]
    assert "tail_metrics" in raw
    assert "baseline_tail_metrics" in raw
    assert "mae_top25pct" in raw["tail_metrics"]

def test_model_beats_mean_baseline(sqlite_path: str, tmp_path: Path) -> None:
    """Quality gate: Ridge must outperform a mean-predictor on the held-out set.

    This is the minimum bar for a model to be worth deploying.  If this test
    fails it signals a data contract issue, feature leak, or training bug.
    """
    meta = train_and_save(
        sqlite_path=sqlite_path,
        model_dir=str(tmp_path / "model"),
        random_seed=7,
        golden_max_rows=600,
    )
    assert meta.metrics["mae"] < meta.baseline_metrics["mae"], (
        f"Ridge MAE ({meta.metrics['mae']:.4f}) must be lower than "
        f"mean-baseline MAE ({meta.baseline_metrics['mae']:.4f})"
    )


def test_model_mae_within_absolute_threshold(sqlite_path: str, tmp_path: Path) -> None:
    """Quality gate: absolute MAE on the golden subset must stay below a known threshold.

    The threshold (35.0) was set empirically on the golden-600 split (seed=7).
    Adjust if the data distribution changes significantly.  A regression here
    signals an unexpected change in data or model behaviour.
    """
    MAE_THRESHOLD = 35.0  # monetary units

    meta = train_and_save(
        sqlite_path=sqlite_path,
        model_dir=str(tmp_path / "model"),
        random_seed=7,
        golden_max_rows=600,
    )
    assert meta.metrics["mae"] < MAE_THRESHOLD, (
        f"Ridge MAE ({meta.metrics['mae']:.4f}) exceeded threshold of {MAE_THRESHOLD}. "
        "Check for data drift or model regression."
    )
