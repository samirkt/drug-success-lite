"""Dataset cohort lookup for the web tool: given a query (SMILES, ICD-10) pair, summarize how
*comparable* drug-indication programs in our dataset actually fared.

Three intuitive questions, all answered against `data/datasets/ours_di.parquet` — the exact
dataset the served `abl_md` model trains on:

  1. molecular  — of the top-10 drugs most similar to this molecule (Tanimoto ECFP4 >= 0.4),
                  how many reached final approval?
  2. disease    — of all programs targeting this disease area (matched on the 3-char ICD-10
                  category), how many reached final approval?
  3. both       — how many similar drugs were tested on this disease, and how many were approved?

Similarity is vectorized over precomputed ECFP4 fingerprints (`data/features/fingerprints.parquet`,
Morgan radius 2 / 2048 bits — the same fingerprint the model uses), so a query is a single
matrix-vector product, no per-row RDKit loop. Pure-python (no web deps); the FastAPI layer is in
`app.py`.

    uv run python -m dsm.cohort     # sanity-check that the cohort loads + a sample query
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import DATASETS_DIR, DEFAULT_FINGERPRINTS
from .models.sklearn_adapter import _icd_category

logger = logging.getLogger(__name__)

DATASET_PATH = DATASETS_DIR / "ours_di.parquet"
FINGERPRINTS_PATH = DEFAULT_FINGERPRINTS

TANIMOTO_THRESHOLD = 0.4   # ECFP4 Tanimoto floor for "similar"
TOP_K_NEIGHBORS = 10       # molecular cohort target size
_SELF_MATCH = 0.999        # treat sim >= this with identical SMILES as the query itself

_COHORT: dict | None = None


def load_cohort(force: bool = False) -> dict:
    """Build (and cache) the in-memory cohort: ECFP4 bit-matrix + per-program metadata.

    Raises if the materialized dataset is missing."""
    global _COHORT
    if _COHORT is not None and not force:
        return _COHORT

    if not DATASET_PATH.exists():
        raise RuntimeError(
            f"no cohort dataset at {DATASET_PATH} — materialize it first with "
            f"`uv run python -m dsm materialize ours_di` (or run the `abl_md` experiment)")

    df = pd.read_parquet(DATASET_PATH)
    fps = pd.read_parquet(FINGERPRINTS_PATH)[["candidate_id", "ecfp4"]]
    df = df.merge(fps, left_on="example_id", right_on="candidate_id", how="inner")

    fp_matrix = np.stack(df["ecfp4"].values).astype(np.uint8)   # (N, 2048)
    canon_smiles = [(s[0] if isinstance(s, (list, np.ndarray)) and len(s) else "") for s in df["smiles"]]

    # Distinct-drug key: the minimized serving dataset ships an opaque `drug_uid` (no DrugBank
    # references); the full local dataset still has `drugbank_id`. Either works as a grouping key.
    key_col = "drug_uid" if "drug_uid" in df.columns else "drugbank_id"

    _COHORT = {
        "fp_matrix": fp_matrix,
        "fp_popcount": fp_matrix.sum(1).astype(np.int32),
        "drug_uid": df[key_col].astype(str).to_numpy(),
        "drug_name": df["drug_name"].astype(str).to_numpy(),
        "indication": df["indication"].astype(str).to_numpy(),
        "label": df["label"].astype(np.int8).to_numpy(),
        "icd_cats": [set(_icd_category(list(c))) for c in df["icd_codes"]],
        "canon_smiles": np.array(canon_smiles, dtype=object),
        "base_rate": float(df["label"].mean()),
        "n": len(df),
    }
    logger.info("loaded cohort: %d programs, base approval rate %.3f",
                _COHORT["n"], _COHORT["base_rate"])
    return _COHORT


def _query_fp(smiles: str):
    """Morgan radius-2 / 2048-bit fingerprint of the query SMILES as a (2048,) uint8 array.
    Returns None if the SMILES doesn't parse."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles((smiles or "").strip())
    if m is None:
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    return np.array(bv, dtype=np.uint8)


def _tanimoto(matrix: np.ndarray, popcount: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Vectorized Tanimoto of every row of `matrix` against query bit-vector `q`."""
    inter = matrix @ q.astype(np.int32)
    union = popcount + int(q.sum()) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    return sim


def _dedupe_by_drug(idx: np.ndarray, sim: np.ndarray, c: dict) -> list[dict]:
    """Collapse program rows to distinct drugs (by drug_uid), keeping each drug's max
    similarity and drug-level approval (approved if ANY of its programs was approved)."""
    by_drug: dict[str, dict] = {}
    for i in idx:
        dbid = c["drug_uid"][i]
        cur = by_drug.get(dbid)
        approved = bool(c["label"][i] == 1)
        if cur is None:
            by_drug[dbid] = {
                "drug_uid": dbid,
                "drug_name": c["drug_name"][i],
                "similarity": float(sim[i]),
                "approved": approved,
            }
        else:
            cur["similarity"] = max(cur["similarity"], float(sim[i]))
            cur["approved"] = cur["approved"] or approved
    return sorted(by_drug.values(), key=lambda d: d["similarity"], reverse=True)


def lookup_exact_match(smiles: str, icd_codes: list[str]) -> dict | None:
    """If this exact molecule has already been approved for this disease area in our
    dataset, return the matching program's identity. Otherwise None."""
    c = load_cohort()
    codes = [s.strip() for s in (icd_codes or []) if s and s.strip()]
    query_cats = set(_icd_category(codes))
    q = _query_fp(smiles) if smiles else None
    if q is None or not query_cats:
        return None

    sim = _tanimoto(c["fp_matrix"], c["fp_popcount"], q)
    same_molecule = sim >= _SELF_MATCH
    same_disease = np.array([bool(cats & query_cats) for cats in c["icd_cats"]])
    approved = c["label"] == 1
    idx = np.where(same_molecule & same_disease & approved)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    return {"drug_name": c["drug_name"][i], "indication": c["indication"][i]}


def cohort_stats(smiles: str, icd_codes: list[str]) -> dict:
    """Summarize comparable programs for one (SMILES, ICD-10 list) query."""
    c = load_cohort()
    smiles = (smiles or "").strip()
    codes = [s.strip() for s in (icd_codes or []) if s and s.strip()]
    query_cats = set(_icd_category(codes))

    q = _query_fp(smiles) if smiles else None
    sim = _tanimoto(c["fp_matrix"], c["fp_popcount"], q) if q is not None else None
    disease_match = (
        np.array([bool(cats & query_cats) for cats in c["icd_cats"]])
        if query_cats else None
    )

    out: dict = {
        "threshold": TANIMOTO_THRESHOLD,
        "smiles_valid": q is not None,
        "n_icd_matched_cats": len(query_cats),
        "base_rate": c["base_rate"],
        "dataset_n": c["n"],
        "molecular": {"available": q is not None},
        "disease": {"available": bool(query_cats)},
        "both": {"available": q is not None and bool(query_cats)},
    }

    # --- Card 1: molecular neighbors (top-K distinct drugs above threshold) ---
    similar_drugs: list[dict] = []
    if sim is not None:
        is_self = (sim >= _SELF_MATCH) & (c["canon_smiles"] == smiles)
        hit = np.where((sim >= TANIMOTO_THRESHOLD) & ~is_self)[0]
        similar_drugs = _dedupe_by_drug(hit, sim, c)
        top = similar_drugs[:TOP_K_NEIGHBORS]
        out["molecular"] = {
            "available": True,
            "n": len(top),
            "n_approved": sum(1 for d in top if d["approved"]),
            "neighbors": [
                {"drug_name": d["drug_name"],
                 "similarity": round(d["similarity"], 3),
                 "approved": d["approved"]}
                for d in top
            ],
        }

    # --- Card 2: disease area (all matching programs) ---
    if disease_match is not None:
        idx = np.where(disease_match)[0]
        n = len(idx)
        n_app = int(c["label"][idx].sum()) if n else 0
        out["disease"] = {
            "available": True,
            "n": n,
            "n_approved": n_app,
            "rate": (n_app / n) if n else None,
            "base_rate": c["base_rate"],
        }

    # --- Card 3: similar drugs tested on this disease ---
    if sim is not None and disease_match is not None:
        idx = np.where((sim >= TANIMOTO_THRESHOLD) & disease_match)[0]
        drugs = _dedupe_by_drug(idx, sim, c)  # approved == approved for a matching-disease program
        out["both"] = {
            "available": True,
            "n": len(drugs),
            "n_approved": sum(1 for d in drugs if d["approved"]),
            "drugs": [
                {"drug_name": d["drug_name"],
                 "similarity": round(d["similarity"], 3),
                 "approved": d["approved"]}
                for d in drugs[:TOP_K_NEIGHBORS]
            ],
        }

    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    c = load_cohort()
    # aspirin + angina (I20)
    stats = cohort_stats("CC(=O)Oc1ccccc1C(=O)O", ["I20.0"])
    logger.info("molecular: %s", stats["molecular"])
    logger.info("disease:   %s", stats["disease"])
    logger.info("both:      %s", stats["both"])
