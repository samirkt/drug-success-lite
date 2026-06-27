"""In-process xgb / logreg adapter over the canonical example parquet.

Builds the requested feature groups from whatever the canonical frame carries:
  - molecule / mol : ECFP4(2048)+MACCS(167) from the canonical `smiles` column
                     (rdkit) — identical construction on every dataset, so it
                     matches what HINT's MPNN consumes.
  - disease / icd  : multi-hot over the canonical `icd_codes` — the same ICD
                     input HINT's GRAM consumes — on every dataset. (disease_area
                     is uninformative and the MeSH tree is incomplete, so the old
                     DiseaseGroup is intentionally no longer used.)
  - admet / target / pathway : the rich dsm composite groups (our data only).

Trains on split in {train, valid} (carving its own stratified inner-val for
xgb early stopping), predicts on split == "test", writes the canonical
predictions parquet. Replaces the old standalone `xgb_on_benchmark.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..encoders import MultiHot
from ..evaluate import metrics
from ..features import build_group
from ..model import build_model

logger = logging.getLogger(__name__)

ECFP_BITS = 2048
MACCS_BITS = 167


class MoleculeFP:
    """ECFP4(2048)+MACCS(167) over a row's canonical `smiles` list (bit-union for
    multi-drug rows). Deterministic — `fit` is a no-op."""

    name = "molecule"

    def fit(self, df: pd.DataFrame, y=None) -> None:  # noqa: D401 - deterministic
        return None

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, MACCSkeys

        RDLogger.DisableLog("rdApp.*")
        out = np.zeros((len(df), ECFP_BITS + MACCS_BITS), dtype=np.float32)
        for i, smiles in enumerate(df["smiles"].values):
            ecfp = np.zeros(ECFP_BITS, dtype=np.float32)
            maccs = np.zeros(MACCS_BITS, dtype=np.float32)
            for smi in (smiles if smiles is not None else []):
                m = Chem.MolFromSmiles(str(smi))
                if m is None:
                    continue
                ecfp = np.maximum(ecfp, np.array(
                    AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=ECFP_BITS), dtype=np.float32))
                maccs = np.maximum(maccs, np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.float32))
            out[i, :ECFP_BITS] = ecfp
            out[i, ECFP_BITS:] = maccs
        return out

    def feature_names(self) -> list[str]:
        return [f"ecfp4_{i}" for i in range(ECFP_BITS)] + [f"maccs_{i}" for i in range(MACCS_BITS)]


def _icd_category(codes: list[str]) -> list[str]:
    """ICD-10 codes -> their 3-char category, normalized. Strips case/punctuation/whitespace and
    drops the sub-class so messy user input matches the model's vocabulary exactly as the clean
    training data did: 'k51.90', ' K51.90 ', and 'K5190' all -> 'K51'. The model is trained only on
    this high-level code, so sub-class differences ('C50.9' vs 'C50.91') collapse to one category.
    Module-level (not a lambda) so the fitted encoder stays picklable for serving."""
    out = []
    for c in codes:
        cat = str(c).strip().upper().replace(".", "")[:3]
        if cat:
            out.append(cat)
    return out


def _make_encoders(features: list[str], df: pd.DataFrame) -> list:
    """Map requested feature names to encoders, dispatching on available columns."""
    encs = []
    for name in features:
        n = name.lower()
        if n in ("molecule", "mol"):
            encs.append(MoleculeFP())
        elif n in ("disease", "icd"):
            # ICD-code multi-hot everywhere, truncated to the 3-char category (e.g. K51.90 -> K51)
            # so coverage is robust to subcode choice — full-resolution codes are too sparse (6k+
            # distinct, ~97% out of a top-200 vocab), which made distinct diseases collapse to the
            # same "other" bucket. disease_area / MeSH are intentionally not used.
            encs.append(MultiHot("icd_codes", prefix="icd", top_k=200, token_fn=_icd_category))
        elif n in ("admet", "target", "pathway", "target_genes"):
            if not _group_available(n, df):
                raise ValueError(
                    f"feature group {n!r} needs rich columns absent from this dataset"
                )
            encs.append(build_group(n))
        elif n == "criteria":
            continue  # sklearn models have no criteria feature; silently skip
        else:
            raise ValueError(f"unknown feature {name!r} for sklearn adapter")
    return encs


def _group_available(name: str, df: pd.DataFrame) -> bool:
    return build_group(name).is_available(df)


def _matrix(encoders: list, df: pd.DataFrame) -> np.ndarray:
    blocks = [e.transform(df) for e in encoders]
    blocks = [b for b in blocks if b.shape[1] > 0]
    if not blocks:
        raise ValueError("no feature group produced any columns")
    return np.hstack(blocks)


class FeaturePipeline:
    """Encoders (+ optional per-group PCA) -> feature matrix. Picklable, so it travels in the saved
    model and rebuilds the exact serving input for a single example."""

    def __init__(self, encoders, pcas=None):
        self.encoders = encoders
        self.pcas = pcas  # list aligned to encoders, or None for raw (full-dimensional) features

    def transform(self, df) -> np.ndarray:
        blocks = []
        for i, e in enumerate(self.encoders):
            b = e.transform(df)
            if self.pcas is not None:
                b = self.pcas[i].transform(b)
            blocks.append(b)
        blocks = [b for b in blocks if b.shape[1] > 0]
        if not blocks:
            raise ValueError("no feature group produced any columns")
        return np.hstack(blocks).astype(np.float32)


def fit_encoders_clf(train_df, features, *, model: str = "xgb", seed: int = 0,
                     inner_val_size: float = 0.1, pca: int | None = None):
    """Fit a feature pipeline + classifier on a train frame.

    Returns (pipeline, clf, val_scores, y_val): the fitted FeaturePipeline/classifier plus the
    held-out inner-val raw scores and labels, which a caller can use to fit a probability calibrator
    without extra training. Carves a stratified inner-val for early stopping; fits encoders (and,
    when `pca` is set, a per-group PCA-<pca> bottleneck) on the inner-train slice."""
    from sklearn.model_selection import train_test_split

    y_train = train_df["label"].to_numpy(dtype=int)
    inner_idx, val_idx = train_test_split(
        np.arange(len(train_df)), test_size=inner_val_size,
        stratify=y_train, random_state=seed,
    )
    inner_df = train_df.iloc[inner_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    y_inner, y_val = y_train[inner_idx], y_train[val_idx]

    encoders = _make_encoders(features, train_df)
    for e in encoders:
        e.fit(inner_df, y_inner) if _takes_y(e) else e.fit(inner_df)

    pcas = None
    if pca:
        from sklearn.decomposition import PCA
        pcas = []
        for e in encoders:
            mat = e.transform(inner_df)
            p = PCA(n_components=min(pca, mat.shape[1]), random_state=seed)
            p.fit(mat)
            pcas.append(p)
    pipeline = FeaturePipeline(encoders, pcas)

    X_inner = pipeline.transform(inner_df)
    X_val = pipeline.transform(val_df)
    logger.info("features %s -> X_inner=%s (pca=%s)", features, X_inner.shape, pca)

    n_pos = int(y_inner.sum())
    spw = (len(y_inner) - n_pos) / n_pos if n_pos else 1.0
    clf = build_model(model, scale_pos_weight=spw, random_state=seed)
    clf.fit(X_inner, y_inner, X_val=X_val, y_val=y_val)
    val_scores = clf.predict_proba(X_val)[:, 1]
    return pipeline, clf, val_scores, y_val


class PlattCalibrator:
    """Platt (logistic) calibration: maps a raw model score to a probability via a fitted sigmoid.
    Smooth and monotone — unlike isotonic it neither collapses score ranges into flat steps nor
    overfits a sparse high tail to certainty. Picklable; exposes `.predict()` like IsotonicRegression."""

    def __init__(self, lr):
        self._lr = lr

    def predict(self, x):
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        return self._lr.predict_proba(x)[:, 1]

    @classmethod
    def fit(cls, scores, labels):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(np.asarray(scores, dtype=float).reshape(-1, 1), np.asarray(labels, dtype=int))
        return cls(lr)


def _fit_calibrator(train_df, features, val_scores, y_val, *, model, seed, inner_val_size, pca,
                    cv_folds):
    """Fit a Platt calibrator. With `cv_folds` > 1, fit it on cross-validated out-of-fold
    predictions over the FULL train split (each fold scored by a model trained on the others) — far
    more, cleaner calibration data than the single inner-val slice. Otherwise fall back to the
    inner-val scores (no extra training)."""
    if cv_folds and cv_folds > 1:
        from sklearn.model_selection import StratifiedKFold

        y = train_df["label"].to_numpy(dtype=int)
        oof = np.zeros(len(train_df), dtype=float)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        for k, (tr_idx, oof_idx) in enumerate(skf.split(np.arange(len(train_df)), y), 1):
            logger.info("calibration fold %d/%d", k, cv_folds)
            pipe_f, clf_f, _, _ = fit_encoders_clf(
                train_df.iloc[tr_idx].reset_index(drop=True), features,
                model=model, seed=seed, inner_val_size=inner_val_size, pca=pca)
            oof[oof_idx] = clf_f.predict_proba(
                pipe_f.transform(train_df.iloc[oof_idx].reset_index(drop=True)))[:, 1]
        return PlattCalibrator.fit(oof, y), f"platt-cv{cv_folds}"
    return PlattCalibrator.fit(val_scores, y_val), "platt-innerval"


def run(*, dataset_path: Path, features: list[str], out_path: Path,
        model: str = "xgb", seed: int = 0, inner_val_size: float = 0.1,
        pca: int | None = None, calibration_folds: int = 0, **_ignored) -> Path:
    df = pd.read_parquet(dataset_path)
    train_df = df[df["split"].isin(["train", "valid"])].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    y_test = test_df["label"].to_numpy(dtype=int)

    pipeline, clf, val_scores, y_val = fit_encoders_clf(
        train_df, features, model=model, seed=seed, inner_val_size=inner_val_size, pca=pca)
    X_test = pipeline.transform(test_df)
    y_proba = clf.predict_proba(X_test)[:, 1]

    preds = pd.DataFrame({
        "example_id": test_df["example_id"].astype(str).values,
        "label": y_test.astype(np.int8),
        "phase": test_df["phase"].astype(str).values,
        "y_proba": y_proba.astype(float),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_path, index=False)
    m = metrics(y_test, y_proba)
    logger.info("%s on %s: ROC-AUC=%.4f PR-AUC=%.4f F1=%.4f -> %s",
                model, dataset_path.stem, m["roc_auc"], m["pr_auc"], m["f1"], out_path)

    calibrator, calib_kind = _fit_calibrator(
        train_df, features, val_scores, y_val, model=model, seed=seed,
        inner_val_size=inner_val_size, pca=pca, cv_folds=calibration_folds)
    _save_model(out_path.parent / "model.joblib", pipeline, clf, calibrator, calib_kind,
                features, train_df, y_test, y_proba, m["roc_auc"])
    return out_path


def _save_model(path, pipeline, clf, calibrator, calib_kind, features, train_df, y_test, y_proba,
                roc_auc) -> None:
    """Persist the fitted model + calibrator for reloading (serving, re-scoring), reporting the
    calibrator's effect via the raw vs calibrated Brier on the test split."""
    import joblib
    from sklearn.metrics import brier_score_loss

    payload = {
        "pipeline": pipeline,
        "clf": clf,
        "calibrator": calibrator,
        "calibration": calib_kind,
        "features": list(features),
        "base_rate": float(train_df["label"].mean()),
        "metrics": {
            "roc_auc": float(roc_auc),
            "brier_raw": float(brier_score_loss(y_test, y_proba)),
            "brier": float(brier_score_loss(y_test, calibrator.predict(y_proba))),
            "n_test": int(len(y_test)),
        },
    }
    joblib.dump(payload, path)
    logger.info("saved model -> %s (%s; Brier %.4f -> %.4f after calibration)",
                path, calib_kind, payload["metrics"]["brier_raw"], payload["metrics"]["brier"])


def _takes_y(encoder) -> bool:
    """CompositeGroup.fit accepts (df, y); leaf encoders accept (df)."""
    from ..features import CompositeGroup
    return isinstance(encoder, CompositeGroup)
