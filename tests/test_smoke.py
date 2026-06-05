"""End-to-end smoke test: build a trial frame, train, check metrics + outputs."""

from __future__ import annotations

import math

import pytest

from dsm.config import DEFAULT_CANDIDATE_DETAIL, DEFAULT_TRIAL_DETAIL, FeatureConfig, ModelingConfig
from dsm.train import train_one_run, write_run

pytestmark = pytest.mark.skipif(
    not (DEFAULT_CANDIDATE_DETAIL.exists() and DEFAULT_TRIAL_DETAIL.exists()),
    reason="inputs/ parquets not present; run ./setup_inputs.sh first",
)


def _config(**kw) -> ModelingConfig:
    base = dict(
        features=FeatureConfig(enabled=("molecule",)),
        training_granularity="trial",
        time_split_column="trial_start_date",
        time_split_year=2019,
    )
    base.update(kw)
    return ModelingConfig(**base)


def test_trial_run_produces_valid_metrics_and_predictions(tmp_path):
    result = train_one_run(_config())

    roc = result.metrics["roc_auc"]
    assert isinstance(roc, float) and math.isfinite(roc) and 0.0 < roc < 1.0

    # Per-phase metrics exist for trial granularity.
    assert result.per_phase_metrics
    assert "P1->P2" in result.per_phase_metrics

    # Predictions are HINT-ready: nct_id + trial_phase keyed.
    preds = result.test_predictions
    for col in ("nct_id", "trial_phase", "y_true", "y_proba"):
        assert col in preds.columns

    write_run(result, tmp_path)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "predictions.csv").exists()


def test_drug_indication_run(tmp_path):
    result = train_one_run(
        _config(training_granularity="drug_indication", time_split_column="earliest_start_date")
    )
    roc = result.metrics["roc_auc"]
    assert math.isfinite(roc) and 0.0 < roc < 1.0
    assert not result.per_phase_metrics  # no per-phase for drug_indication
