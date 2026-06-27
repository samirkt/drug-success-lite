"""Serving layer for the molecule+disease model: load the saved experiment model and score one
(SMILES, ICD-10) pair.

The web tool serves the *exact* model trained by the `abl_md` experiment — `dsm run abl_md`
persists `runs/abl_md/model.joblib` (encoders + classifier + calibrator), and this module just
loads it. No training happens here. Pure-python (no web deps); the FastAPI layer lives in
`app.py`.

    uv run python -m dsm run abl_md     # train + save the model the tool serves
    uv run python -m dsm.serve         # sanity-check that the saved model loads
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .cohort import training_support
from .config import PROJECT_ROOT

logger = logging.getLogger(__name__)

RUNS_DIR = PROJECT_ROOT / "runs"
EXPERIMENT = "abl_md"   # PCA-50 molecule+disease — the model the web tool serves
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


def _disease_encoder(encoders):
    for e in encoders:
        if getattr(e, "column", None) == "icd_codes":
            return e
    return None


def _n_in_vocab(encoders, codes: list[str]) -> int:
    """How many entered ICD codes the model actually has signal for — applying the encoder's own
    token_fn (e.g. 3-char category truncation) before checking the learned vocabulary."""
    enc = _disease_encoder(encoders)
    if enc is None:
        return 0
    toks = enc._token_fn(codes) if getattr(enc, "_token_fn", None) else codes
    vocab = set(enc.vocab)
    return sum(1 for t in toks if t in vocab)


def predict_one(smiles: str, icd_codes: list[str]) -> dict:
    """Score one (SMILES, ICD-10 list) pair with the saved molecule+disease model.

    Returns the calibrated approval probability (plus the raw model score) and honest diagnostics:
    whether the SMILES parsed and how many ICD codes are in the model's learned vocabulary."""
    art = load_predictor()
    pipeline = art["pipeline"]

    smiles = (smiles or "").strip()
    codes = [c.strip() for c in (icd_codes or []) if c and c.strip()]
    row = pd.DataFrame({
        "smiles": [[smiles] if smiles else []],
        "icd_codes": [codes],
    })
    X = pipeline.transform(row)
    raw = float(art["clf"].predict_proba(X)[0, 1])
    calibrator = art.get("calibrator")
    proba = float(calibrator.predict([raw])[0]) if calibrator is not None else raw

    support = training_support(smiles, codes)
    match = support["exact_match"]
    if match is not None:
        proba = 1.0

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    smiles_valid = bool(smiles) and Chem.MolFromSmiles(smiles) is not None

    return {
        "approval_probability": proba,
        "raw_score": raw,
        "already_approved": match is not None,
        "matched_drug_name": match["drug_name"] if match else None,
        "matched_indication": match["indication"] if match else None,
        # training-data support (applicability domain): how far this query is from what the model
        # learned from, combining molecular and disease proximity. See dsm/cohort.training_support.
        "support_band": support["band"],
        "support_score": support["support_score"],
        "mol_similarity": support["molecular"],
        "disease_support": support["disease"],
        "base_rate": art.get("base_rate"),
        "smiles_valid": smiles_valid,
        "n_icd_total": len(codes),
        "n_icd_in_vocab": _n_in_vocab(pipeline.encoders, codes),
        "model": art.get("metrics", {}),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    art = load_predictor()
    logger.info("loaded %s — features=%s metrics=%s", ARTIFACT_PATH, art.get("features"),
                art.get("metrics"))
