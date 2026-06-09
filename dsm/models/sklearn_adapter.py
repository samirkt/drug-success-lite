"""In-process xgb / logreg adapter over the canonical example parquet.

Builds the requested feature groups from whatever the canonical frame carries:
  - molecule / mol : ECFP4(2048)+MACCS(167) from the canonical `smiles` column
                     (rdkit) — identical construction on every dataset, so it
                     matches what HINT's MPNN consumes.
  - disease        : the rich dsm DiseaseGroup (disease_area + MeSH) when those
                     columns are present (our data); otherwise multi-hot over the
                     canonical `icd_codes` (the HINT benchmark) — the same ICD
                     input HINT's GRAM consumes.
  - icd            : force the icd_codes multi-hot regardless of source.
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
from ..features import DiseaseGroup, build_group
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


def _make_encoders(features: list[str], df: pd.DataFrame) -> list:
    """Map requested feature names to encoders, dispatching on available columns."""
    encs = []
    for name in features:
        n = name.lower()
        if n in ("molecule", "mol"):
            encs.append(MoleculeFP())
        elif n == "disease":
            if "disease_area" in df.columns or "mesh_condition_tree_numbers" in df.columns:
                encs.append(DiseaseGroup())          # rich (our data)
            else:
                encs.append(MultiHot("icd_codes", prefix="icd", top_k=200))  # benchmark
        elif n == "icd":
            encs.append(MultiHot("icd_codes", prefix="icd", top_k=200))
        elif n in ("admet", "target", "pathway"):
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


def run(*, dataset_path: Path, features: list[str], out_path: Path,
        model: str = "xgb", seed: int = 0, inner_val_size: float = 0.1,
        **_ignored) -> Path:
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(dataset_path)
    train_df = df[df["split"].isin(["train", "valid"])].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    y_train = train_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)

    inner_idx, val_idx = train_test_split(
        np.arange(len(train_df)), test_size=inner_val_size,
        stratify=y_train, random_state=seed,
    )
    inner_df = train_df.iloc[inner_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    y_inner, y_val = y_train[inner_idx], y_train[val_idx]

    encoders = _make_encoders(features, df)
    for e in encoders:
        e.fit(inner_df, y_inner) if _takes_y(e) else e.fit(inner_df)
    X_inner = _matrix(encoders, inner_df)
    X_val = _matrix(encoders, val_df)
    X_test = _matrix(encoders, test_df)
    logger.info("features %s -> X_inner=%s X_test=%s", features, X_inner.shape, X_test.shape)

    n_pos = int(y_inner.sum())
    spw = (len(y_inner) - n_pos) / n_pos if n_pos else 1.0
    clf = build_model(model, scale_pos_weight=spw, random_state=seed)
    clf.fit(X_inner, y_inner, X_val=X_val, y_val=y_val)
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
    return out_path


def _takes_y(encoder) -> bool:
    """CompositeGroup.fit accepts (df, y); leaf encoders accept (df)."""
    from ..features import CompositeGroup
    return isinstance(encoder, CompositeGroup)
