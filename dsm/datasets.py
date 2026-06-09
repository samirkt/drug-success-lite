"""The single dataset materializer — the one place every modeling decision is made.

`materialize(spec)` writes `data/datasets/<name>.parquet` in the **canonical
example schema** that both model families consume:

    example_id : str        candidate_id (drug_indication) or nct_id (trial / benchmark)
    label      : int8        0/1 — decided HERE, once, never re-derived downstream
    phase      : str         normalized vocabulary (P1/P2/P3/P4/P1P2/P2P3/EarlyP1/other)
    smiles     : list[str]   molecule input (xgb fingerprints it; HINT MPNNs it)
    icd_codes  : list[str]   FLAT ICD-10 list (HINT's nested form is built only at its edge)
    criteria   : str         eligibility text or "" (HINT-only)
    split      : str         train | valid | test  (valid only when the source defines it)
    + passthrough            rich admet/target/pathway columns for `dsm` sources; absent for benchmark

Two ingest paths, one output:
  - `_from_dsm_parquets`   our candidate/trial parquets (label + temporal split + row filter).
  - `_from_hint_benchmark` HINT's TOP phase_*_{train,valid,test}.csv (3-way split preserved).

Row filtering (drop empty SMILES / empty ICD) happens HERE so xgb and HINT train
and test on the identical population. Nothing downstream re-labels, re-splits,
re-filters, or re-normalizes phases.
"""

from __future__ import annotations

import ast
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import splits
from .config import (
    DATASETS_DIR,
    HINT_BENCHMARK_DIR,
    LabelConfig,
    ModelingConfig,
)
from .dataset import apply_label, load_candidate_detail, load_trial_detail

logger = logging.getLogger(__name__)
csv.field_size_limit(10 ** 9)

CANONICAL_CORE = ["example_id", "label", "phase", "smiles", "icd_codes", "criteria", "split"]


# --------------------------------------------------------------------------- #
# Dataset specs
# --------------------------------------------------------------------------- #
@dataclass
class DatasetSpec:
    """Declarative description of one materializable dataset."""

    name: str
    kind: str                       # "dsm" | "hint_benchmark"
    # dsm sources
    granularity: str = "drug_indication"   # "drug_indication" | "trial"
    time_split_year: Optional[int] = 2019
    label: LabelConfig = field(default_factory=LabelConfig)
    # hint_benchmark sources
    phase_stem: Optional[str] = None       # e.g. "phase_I" -> phase_I_{train,valid,test}.csv

    def path(self) -> Path:
        return DATASETS_DIR / f"{self.name}.parquet"


# --------------------------------------------------------------------------- #
# Shared normalizers
# --------------------------------------------------------------------------- #
_PHASE_MAP = {
    "phase 1": "P1", "phase 2": "P2", "phase 3": "P3", "phase 4": "P4",
    "phase 1/phase 2": "P1P2", "phase 2/phase 3": "P2P3",
    "early phase 1": "EarlyP1",
}


def normalize_phase(raw) -> str:
    """One phase vocabulary for every source. Unknown / N/A -> 'other'."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "other"
    s = str(raw).strip().lower()
    if not s:
        return "other"
    return _PHASE_MAP.get(s, "other")


def _flat_icd(cell) -> list[str]:
    """Coerce any ICD cell (array / flat list-repr / nested list-of-lists string)
    into a clean FLAT list[str]. This is the only ICD parser dsm needs; the
    nested-string form HINT wants is rebuilt at HINT's edge from this flat list."""
    if cell is None:
        return []
    if isinstance(cell, (list, tuple, np.ndarray)):
        items = list(cell)
    else:
        s = str(cell).strip()
        if not s:
            return []
        try:
            items = [ast.literal_eval(s)]
        except (ValueError, SyntaxError):
            return [s]

    out: list[str] = []

    def rec(x):
        if isinstance(x, str):
            xs = x.strip()
            if xs.startswith("[") and xs.endswith("]"):
                try:
                    rec(ast.literal_eval(xs))
                    return
                except (ValueError, SyntaxError):
                    pass
            if xs:
                out.append(xs)
        elif isinstance(x, (list, tuple, np.ndarray)):
            for e in x:
                rec(e)

    rec(items)
    # dedup preserving order
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _smiles_list(cell) -> list[str]:
    """Coerce a SMILES cell (single string or list-repr / array) into list[str]."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, (list, tuple, np.ndarray)):
        return [str(x) for x in cell if x is not None and str(x).strip()]
    s = str(cell).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v if x is not None and str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    return [s]


# --------------------------------------------------------------------------- #
# Materialize
# --------------------------------------------------------------------------- #
def materialize(spec: DatasetSpec, *, force: bool = False) -> Path:
    """Build (or reuse) the canonical example parquet for `spec`."""
    out = spec.path()
    if out.exists() and not force:
        logger.info("dataset %s already materialized -> %s", spec.name, out)
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    if spec.kind == "dsm":
        df = _from_dsm_parquets(spec)
    elif spec.kind == "hint_benchmark":
        df = _from_hint_benchmark(spec)
    else:
        raise ValueError(f"unknown dataset kind {spec.kind!r}")

    _validate(df, spec)
    df.to_parquet(out, index=False)
    logger.info(
        "materialized %s: %d rows (train=%d valid=%d test=%d) -> %s",
        spec.name, len(df),
        int((df.split == "train").sum()), int((df.split == "valid").sum()),
        int((df.split == "test").sum()), out,
    )
    return out


def _row_filter(smiles: pd.Series, icd: pd.Series) -> tuple[pd.Series, int, int]:
    """Keep rows with a non-empty SMILES list AND a non-empty ICD list."""
    has_smiles = smiles.map(len) > 0
    has_icd = icd.map(len) > 0
    keep = has_smiles & has_icd
    return keep, int((~has_smiles).sum()), int((has_smiles & ~has_icd).sum())


def _from_dsm_parquets(spec: DatasetSpec) -> pd.DataFrame:
    """Our candidate/trial parquets -> canonical schema (+ rich passthrough)."""
    if spec.granularity == "trial":
        src = load_trial_detail(ModelingConfig().trial_detail_path)
        src = src[src["trial_inferred_label"].notna()].copy()
        src["y"] = src["trial_inferred_label"].astype(np.int8)
        # join candidate-level rich columns for the feature groups
        cand = load_candidate_detail(ModelingConfig().candidate_detail_path)
        overlap = (set(src.columns) & set(cand.columns)) - {"candidate_id"}
        cand = cand.drop(columns=list(overlap), errors="ignore")
        src = src.merge(cand, on="candidate_id", how="left").reset_index(drop=True)
        id_col, phase_col, time_col = "nct_id", "trial_phase", "trial_start_date"
    else:
        src = load_candidate_detail(ModelingConfig().candidate_detail_path)
        src = apply_label(src, spec.label).reset_index(drop=True)
        id_col, phase_col, time_col = "candidate_id", "highest_phase", "earliest_start_date"

    smiles_col = "smiles_canonical" if "smiles_canonical" in src.columns else "smiles"
    smiles = src[smiles_col].map(_smiles_list)
    icd = src.get("icd10_codes", pd.Series([None] * len(src))).map(_flat_icd)

    keep, n_no_smi, n_no_icd = _row_filter(smiles, icd)
    src = src.loc[keep].reset_index(drop=True)
    smiles = smiles.loc[keep].reset_index(drop=True)
    icd = icd.loc[keep].reset_index(drop=True)

    # Temporal split (train <= year, test > year); rows without a year are dropped.
    train_idx, test_idx = splits.split(
        src, test_size=0.2, seed=0,
        time_split_column=time_col, time_split_year=spec.time_split_year,
    )
    split = np.array(["__drop__"] * len(src), dtype=object)
    split[train_idx] = "train"
    split[test_idx] = "test"

    out = pd.DataFrame({
        "example_id": src[id_col].astype(str),
        "label": src["y"].astype(np.int8),
        "phase": src[phase_col].map(normalize_phase) if phase_col in src.columns else "other",
        "smiles": smiles,
        "icd_codes": icd,
        "criteria": "",
        "split": split,
    })
    # rich passthrough (everything the dsm feature groups read) — keep source cols
    # that aren't part of the canonical core
    passthrough = [c for c in src.columns if c not in out.columns]
    out = pd.concat([out, src[passthrough].reset_index(drop=True)], axis=1)

    out = out[out["split"] != "__drop__"].reset_index(drop=True)
    logger.info(
        "dsm/%s: dropped no_smiles=%d no_icd=%d; kept %d rows",
        spec.name, n_no_smi, n_no_icd, len(out),
    )
    return out


def _read_hint_csv(path: Path, split_name: str) -> pd.DataFrame:
    rows = list(csv.reader(open(path)))[1:]
    recs = []
    for r in rows:
        recs.append({
            "example_id": r[0],
            "label": int(r[3]),
            "phase": normalize_phase(r[4]),
            "smiles": _smiles_list(r[8]),
            "icd_codes": _flat_icd(r[6]),
            "criteria": r[9],
            "split": split_name,
        })
    return pd.DataFrame(recs)


def _from_hint_benchmark(spec: DatasetSpec) -> pd.DataFrame:
    """HINT TOP phase_*_{train,valid,test}.csv -> canonical schema (3-way split kept)."""
    if not spec.phase_stem:
        raise ValueError(f"hint_benchmark spec {spec.name!r} needs phase_stem")
    parts = []
    for split_name in ("train", "valid", "test"):
        path = HINT_BENCHMARK_DIR / f"{spec.phase_stem}_{split_name}.csv"
        parts.append(_read_hint_csv(path, split_name))
    df = pd.concat(parts, ignore_index=True)

    keep, n_no_smi, n_no_icd = _row_filter(df["smiles"], df["icd_codes"])
    df = df.loc[keep].reset_index(drop=True)
    logger.info(
        "hint_benchmark/%s: dropped no_smiles=%d no_icd=%d; kept %d rows",
        spec.name, n_no_smi, n_no_icd, len(df),
    )
    return df[CANONICAL_CORE]


def _validate(df: pd.DataFrame, spec: DatasetSpec) -> None:
    missing = [c for c in CANONICAL_CORE if c not in df.columns]
    if missing:
        raise ValueError(f"{spec.name}: canonical columns missing {missing}")
    if not df["split"].isin({"train", "valid", "test"}).all():
        bad = set(df["split"].unique()) - {"train", "valid", "test"}
        raise ValueError(f"{spec.name}: bad split values {bad}")
    if df["example_id"].duplicated().any():
        n = int(df["example_id"].duplicated().sum())
        logger.warning("%s: %d duplicate example_id values", spec.name, n)
    if df.empty:
        raise ValueError(f"{spec.name}: materialized 0 rows")
