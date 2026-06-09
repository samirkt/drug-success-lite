"""Smoke + contract tests for the unified pipeline.

Covers: the canonical materializer, the in-process sklearn adapter, and the one
piece of the dsm<->HINT contract that must never drift — the flat-ICD -> nested
list-of-lists string conversion HINT's parser slices back apart.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

from dsm.config import DEFAULT_CANDIDATE_DETAIL, HINT_BENCHMARK_DIR
from dsm.datasets import CANONICAL_CORE, DatasetSpec, materialize
from dsm.evaluate import evaluate_predictions
from dsm.models import run_model

HINT_P1 = HINT_BENCHMARK_DIR / "phase_I_test.csv"
pytestmark = pytest.mark.skipif(
    not HINT_P1.exists(),
    reason="HINT benchmark not present under data/hint_benchmark/",
)


def test_materialize_canonical_schema(tmp_path):
    spec = DatasetSpec(name="hint_p1", kind="hint_benchmark", phase_stem="phase_I")
    df = pd.read_parquet(materialize(spec))
    for col in CANONICAL_CORE:
        assert col in df.columns
    assert df["split"].isin({"train", "valid", "test"}).all()
    assert df["phase"].isin({"P1", "P2", "P3", "P4", "P1P2", "P2P3", "EarlyP1", "other"}).all()
    assert df["label"].isin({0, 1}).all()
    assert (df["smiles"].map(len) > 0).all() and (df["icd_codes"].map(len) > 0).all()


def test_sklearn_adapter_runs(tmp_path):
    materialize(DatasetSpec(name="hint_p1", kind="hint_benchmark", phase_stem="phase_I"))
    out = tmp_path / "predictions.parquet"
    run_model("xgb", Path("data/datasets/hint_p1.parquet"), ["molecule", "disease"], out)
    preds = pd.read_parquet(out)
    assert list(preds.columns) == ["example_id", "label", "phase", "y_proba"]
    m = evaluate_predictions(preds)
    roc = m["overall"]["roc_auc"]
    assert math.isfinite(roc) and 0.0 < roc < 1.0
    assert m["per_phase"]  # grouped by normalized phase


def _hint_icd_parser(text):
    """Verbatim copy of HINT/dataloader.py:icdcode_text_2_lst_of_lst (the consumer)."""
    text = text[2:-2]
    lst_lst = []
    for i in text.split('", "'):
        i = i[1:-1]
        lst_lst.append([j.strip()[1:-1] for j in i.split(',')])
    return lst_lst


def test_icd_contract_roundtrips():
    """to_hint_cells must produce exactly what HINT's parser slices back to the codes."""
    path = Path(__file__).resolve().parents[1] / "hint" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("hint_run_experiment", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # only stdlib imports at module top -> safe in dsm venv

    codes = ["C71.7", "C71.9", "C79.31"]
    _, icd_cell = mod.to_hint_cells(["CC", "CCO"], codes)
    recovered = _hint_icd_parser(icd_cell)
    assert recovered == [codes], (recovered, icd_cell)


def test_shared_population_between_models():
    """xgb and HINT consume the SAME materialized file -> identical test population."""
    if not DEFAULT_CANDIDATE_DETAIL.exists():
        pytest.skip("our candidate parquet not present")
    p = materialize(DatasetSpec(name="hint_p1", kind="hint_benchmark", phase_stem="phase_I"))
    df = pd.read_parquet(p)
    test_ids = set(df[df.split == "test"].example_id)
    assert len(test_ids) == 627  # the documented Phase I test size
