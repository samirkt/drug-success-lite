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
from collections import Counter

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
    fp_popcount = fp_matrix.sum(1).astype(np.int32)
    canon_smiles = [(s[0] if isinstance(s, (list, np.ndarray)) and len(s) else "") for s in df["smiles"]]
    icd_cats = [set(_icd_category(list(c))) for c in df["icd_codes"]]

    # Distinct-drug key: the minimized serving dataset ships an opaque `drug_uid` (no DrugBank
    # references); the full local dataset still has `drugbank_id`. Either works as a grouping key.
    key_col = "drug_uid" if "drug_uid" in df.columns else "drugbank_id"

    # Applicability-domain references, computed against the model's TRAINING pool only (ours_di is
    # train/test; "valid" folded in if ever present) — that's the data the model learned from.
    train_mask = df["split"].isin(["train", "valid"]).to_numpy()
    mol_ref = _molecular_reference(fp_matrix[train_mask], fp_popcount[train_mask])
    cat_counts = Counter(cat for cats, t in zip(icd_cats, train_mask) if t for cat in cats)
    cat_sizes_sorted = np.array(sorted(cat_counts.values()) or [0])

    _COHORT = {
        "fp_matrix": fp_matrix,
        "fp_popcount": fp_popcount,
        "drug_uid": df[key_col].astype(str).to_numpy(),
        "drug_name": df["drug_name"].astype(str).to_numpy(),
        "indication": df["indication"].astype(str).to_numpy(),
        "label": df["label"].astype(np.int8).to_numpy(),
        "icd_cats": icd_cats,
        "canon_smiles": np.array(canon_smiles, dtype=object),
        "base_rate": float(df["label"].mean()),
        "n": len(df),
        "train_mask": train_mask,
        "mol_ref": mol_ref,                    # sorted train nearest-neighbor Tanimoto distribution
        "cat_counts": cat_counts,              # train programs per 3-char ICD category
        "cat_sizes_sorted": cat_sizes_sorted,  # sorted category sizes (disease AD reference)
    }
    logger.info("loaded cohort: %d programs (%d train), base approval rate %.3f",
                _COHORT["n"], int(train_mask.sum()), _COHORT["base_rate"])
    return _COHORT


def _molecular_reference(fp: np.ndarray, pop: np.ndarray, *, sample: int = 1500,
                         seed: int = 0) -> np.ndarray:
    """Sorted distribution of each (sampled) train molecule's nearest-neighbor Tanimoto to the rest
    of train. Sets data-driven applicability-domain cutoffs (no magic constants); sampling keeps
    startup cheap versus the full O(N^2)."""
    n = len(fp)
    if n <= 1:
        return np.array([0.0])
    rng = np.random.default_rng(seed)
    s = min(sample, n)
    sel = rng.choice(n, size=s, replace=False)
    inter = fp[sel].astype(np.int32) @ fp.T                  # (s, n)
    union = pop[sel][:, None] + pop[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    sim[np.arange(s), sel] = -1.0                            # exclude self-match
    return np.sort(sim.max(axis=1))


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


def _rank(sorted_ref: np.ndarray, value: float) -> float:
    """Percentile rank of `value` in `sorted_ref` (fraction at or below it), in [0, 1]."""
    if len(sorted_ref) == 0:
        return 0.0
    return float(np.searchsorted(sorted_ref, value, side="right") / len(sorted_ref))


def _band(score: float) -> str:
    """Applicability-domain band from a 0–1 percentile-rank score. Cutoffs are percentiles of the
    model's own training distribution: below the 5th -> extrapolating; below the 25th -> borderline."""
    if score < 0.05:
        return "Out-of-domain"
    if score < 0.25:
        return "Borderline"
    return "In-domain"


_BAND_RANK = {"Out-of-domain": 0, "Borderline": 1, "In-domain": 2}


def training_support(smiles: str, icd_codes: list[str]) -> dict:
    """How close a (SMILES, ICD-10) query is to the examples the model was trained on — the model's
    applicability domain. Combines molecular proximity (ECFP4 Tanimoto k-NN to train molecules) and
    disease proximity (training programs sharing the query's high-level ICD category) via the
    weakest link. Also reports `exact_match`: the same molecule already *approved* for this disease
    area anywhere in our dataset (the certain endpoint of this signal). One Tanimoto pass."""
    c = load_cohort()
    smiles = (smiles or "").strip()
    codes = [s.strip() for s in (icd_codes or []) if s and s.strip()]
    query_cats = set(_icd_category(codes))
    q = _query_fp(smiles) if smiles else None

    out: dict = {
        "exact_match": None,
        "available": False,
        "band": None,
        "support_score": None,
        "molecular": {"available": q is not None},
        "disease": {"available": bool(query_cats)},
    }
    if q is None and not query_cats:
        return out

    sim = _tanimoto(c["fp_matrix"], c["fp_popcount"], q) if q is not None else None

    # Exact match: same molecule (Tanimoto >= 0.999) already tested for this disease area, anywhere
    # in the dataset (train or test) — a factual "we've already seen this" claim, not an AD measure.
    # Approval takes precedence over failure (a terminal positive outcome dominates a prior failure).
    if sim is not None and query_cats:
        same_disease = np.array([bool(cats & query_cats) for cats in c["icd_cats"]])
        same_md = (sim >= _SELF_MATCH) & same_disease
        appr = np.where(same_md & (c["label"] == 1))[0]
        fail = np.where(same_md & (c["label"] == 0))[0]
        hit = int(appr[0]) if len(appr) else (int(fail[0]) if len(fail) else None)
        if hit is not None:
            out["exact_match"] = {"drug_name": c["drug_name"][hit],
                                  "indication": c["indication"][hit],
                                  "approved": bool(c["label"][hit] == 1)}

    # Molecular AD: nearest-neighbor (and top-5) Tanimoto to the training molecules.
    if sim is not None:
        sim_train = sim[c["train_mask"]]
        k = min(5, len(sim_train))
        nn = float(sim_train.max()) if len(sim_train) else 0.0
        knn = float(np.sort(sim_train)[-k:].mean()) if k else 0.0
        score = _rank(c["mol_ref"], nn)
        out["molecular"] = {"available": True, "nn": round(nn, 3), "knn": round(knn, 3),
                            "score": round(score, 3), "band": _band(score)}

    # Disease AD: how many training programs share the query's high-level ICD category.
    if query_cats:
        n_train = max((c["cat_counts"].get(cat, 0) for cat in query_cats), default=0)
        category = max(query_cats, key=lambda cat: c["cat_counts"].get(cat, 0))
        score = _rank(c["cat_sizes_sorted"], n_train)
        out["disease"] = {"available": True, "category": category, "n_train": int(n_train),
                          "score": round(score, 3), "band": _band(score)}

    # Combined = weakest link across the two modalities the model uses.
    if out["molecular"]["available"] and out["disease"]["available"]:
        out["available"] = True
        out["support_score"] = round(min(out["molecular"]["score"], out["disease"]["score"]), 3)
        out["band"] = min((out["molecular"]["band"], out["disease"]["band"]), key=_BAND_RANK.get)
    return out


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
    # applicability domain: a marketed drug in a common area vs a messy-cased ICD code
    logger.info("support (aspirin, i20.0): %s", training_support("CC(=O)Oc1ccccc1C(=O)O", ["i20.0"]))
    logger.info("support (novel scaffold): %s",
                training_support("O=C(N)c1ccc(-c2nnc3n2CCCCC3)cc1Br", ["Z99"]))
