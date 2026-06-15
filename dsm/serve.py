"""Serving layer for the molecule+disease model: load the saved experiment model and score one
(SMILES, ICD-10) pair.

The web tool serves the *exact* model trained by the `xgb_di_md` experiment — `dsm run xgb_di_md`
persists `runs/xgb_di_md/model.joblib` (encoders + classifier + isotonic calibrator), and this
module just loads it. No training happens here. Pure-python (no web deps); the FastAPI layer lives
in `app.py`.

    uv run python -m dsm run xgb_di_md     # train + save the model the tool serves
    uv run python -m dsm.serve             # sanity-check that the saved model loads
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT

logger = logging.getLogger(__name__)

RUNS_DIR = PROJECT_ROOT / "runs"
EXPERIMENT = "xgb_di_md"
ARTIFACT_PATH = RUNS_DIR / EXPERIMENT / "model.joblib"

_PREDICTOR: dict | None = None


def load_predictor(path: Path = ARTIFACT_PATH) -> dict:
    """Load (and cache) the saved experiment model. Raises if it hasn't been trained yet."""
    global _PREDICTOR
    if _PREDICTOR is None:
        if not path.exists():
            raise RuntimeError(
                f"no saved model at {path} — train it first with "
                f"`uv run python -m dsm run {EXPERIMENT}`")
        _PREDICTOR = joblib.load(path)
    return _PREDICTOR


def _disease_vocab(encoders) -> set[str]:
    for e in encoders:
        if getattr(e, "column", None) == "icd_codes":
            return set(e.vocab)
    return set()


def predict_one(smiles: str, icd_codes: list[str]) -> dict:
    """Score one (SMILES, ICD-10 list) pair with the saved molecule+disease model.

    Returns the calibrated approval probability (plus the raw model score) and honest diagnostics:
    whether the SMILES parsed and how many ICD codes are in the model's learned vocabulary."""
    art = load_predictor()
    encoders = art["encoders"]

    smiles = (smiles or "").strip()
    codes = [c.strip() for c in (icd_codes or []) if c and c.strip()]
    row = pd.DataFrame({
        "smiles": [[smiles] if smiles else []],
        "icd_codes": [codes],
    })
    X = np.hstack([e.transform(row) for e in encoders])
    raw = float(art["clf"].predict_proba(X)[0, 1])
    calibrator = art.get("calibrator")
    proba = float(calibrator.predict([raw])[0]) if calibrator is not None else raw

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    smiles_valid = bool(smiles) and Chem.MolFromSmiles(smiles) is not None
    vocab = _disease_vocab(encoders)

    return {
        "approval_probability": proba,
        "raw_score": raw,
        "base_rate": art.get("base_rate"),
        "smiles_valid": smiles_valid,
        "n_icd_total": len(codes),
        "n_icd_in_vocab": sum(1 for c in codes if c in vocab),
        "model": art.get("metrics", {}),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    art = load_predictor()
    logger.info("loaded %s — features=%s metrics=%s", ARTIFACT_PATH, art.get("features"),
                art.get("metrics"))
