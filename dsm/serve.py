"""Serving layer for the molecule+disease model: build a persistent artifact and score one
(SMILES, ICD-10) pair.

The base model reproduces the `xgb_di_md` experiment exactly — same full train split, encoders, and
classifier (via `fit_encoders_clf`) — so the raw score matches that experiment's saved
`predictions.parquet`. An isotonic calibrator, fit on cross-validated out-of-fold train predictions
(no data withheld from the final model, no leakage), is applied on top to produce a calibrated
probability. Pure-python (no web deps); the FastAPI layer lives in `app.py`.

    python -m dsm.serve        # (re)build runs/xgb_di_md/serve_artifact.joblib
"""

from __future__ import annotations

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


N_CALIB_FOLDS = 5   # CV folds for the out-of-fold predictions the calibrator is fit on


def build_artifact(path: Path = ARTIFACT_PATH) -> Path:
    """Dump a servable artifact: the full-train xgb_di_md model + an isotonic calibrator on top.

    The final (encoders, clf) are fit on the FULL train split — identical to the experiment, so the
    raw score matches its predictions.parquet. The calibrator is fit on cross-validated out-of-fold
    predictions of train (each fold scored by a model trained on the other folds), which gives an
    unbiased calibration map without withholding any data from the final model. Brier/ROC are
    reported on the untouched test split."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    ds_path = materialize(DATASETS[DATASET])
    df = pd.read_parquet(ds_path)
    train_df = df[df["split"].isin(["train", "valid"])].reset_index(drop=True)
    y_train = train_df["label"].to_numpy(dtype=int)

    # Out-of-fold train predictions -> isotonic calibrator (no leakage into the final model).
    oof = np.zeros(len(train_df), dtype=float)
    skf = StratifiedKFold(n_splits=N_CALIB_FOLDS, shuffle=True, random_state=0)
    for tr_idx, oof_idx in skf.split(np.arange(len(train_df)), y_train):
        enc_f, clf_f = fit_encoders_clf(train_df.iloc[tr_idx].reset_index(drop=True), FEATURES)
        oof_df = train_df.iloc[oof_idx].reset_index(drop=True)
        oof[oof_idx] = clf_f.predict_proba(_matrix(enc_f, oof_df))[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof, y_train)

    # Final model: the original xgb_di_md, fit on the full train split (untouched).
    encoders, clf = fit_encoders_clf(train_df, FEATURES)

    # Honest evaluation on the untouched test split: raw vs calibrated.
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    y_test = test_df["label"].to_numpy(dtype=int)
    raw_test = clf.predict_proba(_matrix(encoders, test_df))[:, 1]
    cal_test = calibrator.predict(raw_test)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, raw_test)),  # unchanged by monotonic calibration
        "brier_raw": float(brier_score_loss(y_test, raw_test)),
        "brier": float(brier_score_loss(y_test, cal_test)),
        "n_test": int(len(y_test)),
    }

    artifact = {
        "encoders": encoders,
        "clf": clf,
        "calibrator": calibrator,
        "features": list(FEATURES),
        "base_rate": float(train_df["label"].mean()),
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    logger.info("wrote serving artifact -> %s (base_rate=%.4f, Brier %.4f -> %.4f after calibration)",
                path, artifact["base_rate"], metrics["brier_raw"], metrics["brier"])
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
    build_artifact()
