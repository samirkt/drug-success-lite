"""Serving layer for the molecule+disease model: build a persistent artifact and score one
(SMILES, ICD-10) pair.

The artifact reproduces the `xgb_di_md` experiment exactly — same train split, encoders, and
classifier (via `fit_encoders_clf`) — so a served prediction matches that experiment's saved
`predictions.parquet`. Pure-python (no web deps); the FastAPI layer lives in `app.py`.

    python -m dsm.serve        # (re)build runs/xgb_di_md/serve_artifact.joblib
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .datasets import materialize
from .experiments import DATASETS
from .models.sklearn_adapter import _matrix, fit_encoders_clf

logger = logging.getLogger(__name__)

FEATURES = ("molecule", "disease")            # the xgb_di_md feature set
DATASET = "ours_di"
RUNS_DIR = PROJECT_ROOT / "runs"
ARTIFACT_PATH = RUNS_DIR / "xgb_di_md" / "serve_artifact.joblib"

_PREDICTOR: dict | None = None


def _load_metrics() -> dict:
    """Overall ROC-AUC + test size from the xgb_di_md run, if it exists (for UI context)."""
    mj = RUNS_DIR / "xgb_di_md" / "metrics.json"
    if not mj.exists():
        return {}
    try:
        d = json.loads(mj.read_text())
        return {"roc_auc": d.get("overall", {}).get("roc_auc"), "n_test": d.get("n")}
    except (json.JSONDecodeError, OSError):
        return {}


def build_artifact(path: Path = ARTIFACT_PATH) -> Path:
    """Fit (encoders, clf) on the ours_di train split and dump a servable artifact."""
    ds_path = materialize(DATASETS[DATASET])
    df = pd.read_parquet(ds_path)
    train_df = df[df["split"].isin(["train", "valid"])].reset_index(drop=True)
    encoders, clf = fit_encoders_clf(train_df, FEATURES)
    artifact = {
        "encoders": encoders,
        "clf": clf,
        "features": list(FEATURES),
        "base_rate": float(train_df["label"].mean()),
        "metrics": _load_metrics(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    logger.info("wrote serving artifact -> %s (base_rate=%.4f)", path, artifact["base_rate"])
    return path


def load_predictor(path: Path = ARTIFACT_PATH) -> dict:
    """Load (and cache) the artifact, building it on first use if absent."""
    global _PREDICTOR
    if _PREDICTOR is None:
        if not path.exists():
            logger.info("no artifact at %s — building it now", path)
            build_artifact(path)
        _PREDICTOR = joblib.load(path)
    return _PREDICTOR


def _disease_vocab(encoders) -> set[str]:
    for e in encoders:
        if getattr(e, "column", None) == "icd_codes":
            return set(e.vocab)
    return set()


def predict_one(smiles: str, icd_codes: list[str]) -> dict:
    """Score one (SMILES, ICD-10 list) pair with the molecule+disease model.

    Returns the approval probability plus honest diagnostics: whether the SMILES parsed and how
    many of the supplied ICD codes are in the model's learned vocabulary."""
    art = load_predictor()
    encoders = art["encoders"]

    smiles = (smiles or "").strip()
    codes = [c.strip() for c in (icd_codes or []) if c and c.strip()]
    row = pd.DataFrame({
        "smiles": [[smiles] if smiles else []],
        "icd_codes": [codes],
    })
    X = np.hstack([e.transform(row) for e in encoders])
    proba = float(art["clf"].predict_proba(X)[0, 1])

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    smiles_valid = bool(smiles) and Chem.MolFromSmiles(smiles) is not None
    vocab = _disease_vocab(encoders)

    return {
        "approval_probability": proba,
        "base_rate": art.get("base_rate"),
        "smiles_valid": smiles_valid,
        "n_icd_total": len(codes),
        "n_icd_in_vocab": sum(1 for c in codes if c in vocab),
        "model": art.get("metrics", {}),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build_artifact()
