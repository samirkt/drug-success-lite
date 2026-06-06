"""Export a HINT-format CSV (10 cols + a train/test `split` tag) from the inputs.

This is the single contract between drug-success-lite and the HINT repo: HINT
trains and tests on exactly the rows + partition this file emits, so the two
models are comparable on identical data.

Two tasks, selected by `config.training_granularity`:
  - "trial"           : one row per NCT; label = trial_inferred_label (phase-transition).
  - "drug_indication" : one row per resolved candidate; label = eventual approval
                        from `outcome` (Ongoing dropped). The end-to-end LOA task;
                        `nctid` carries the candidate_id as the join key.

Only rows HINT can ingest are emitted (a parseable canonical SMILES + a non-empty
ICD-10 list). Drops are reported. Eligibility criteria are always blanked
(feature parity with the model, which has no criteria feature).
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import splits
from .config import ModelingConfig
from .dataset import apply_label, load_candidate_detail, load_trial_detail

logger = logging.getLogger(__name__)

HINT_COLUMNS = [
    "nctid", "status", "why_stop", "label", "phase",
    "diseases", "icdcodes", "drugs", "smiless", "criteria",
]


def _as_str_list(v) -> list[str]:
    """Coerce a list/array/JSON-ish cell into a clean list[str] ([] for missing)."""
    if v is None or (np.isscalar(v) and isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        return [str(x) for x in v if x is not None and str(x) != ""]
    s = str(v).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed if x is not None and str(x) != ""]
    except (ValueError, SyntaxError):
        pass
    return [s]


def _pylist_repr(items: list[str]) -> str:
    """HINT's list columns are Python-list reprs, e.g. "['a', 'b']"."""
    return str([str(x) for x in items])


def _split_tags(df: pd.DataFrame, config: ModelingConfig) -> np.ndarray:
    """train/test tag per row from the same temporal cutoff the trainer uses."""
    train_idx, test_idx = splits.split(
        df,
        test_size=config.test_size,
        seed=config.seed,
        time_split_column=config.time_split_column,
        time_split_year=config.time_split_year,
    )
    tag = np.array([""] * len(df), dtype=object)
    tag[train_idx] = "train"
    tag[test_idx] = "test"
    return tag


def build_hint_frame(config: ModelingConfig) -> pd.DataFrame:
    """Assemble the HINT-format DataFrame (10 cols + `split`), dispatched by granularity."""
    gran = (config.training_granularity or "trial").lower()
    if gran == "drug_indication":
        return _build_candidate_frame(config)
    return _build_trial_frame(config)


def _emit(df, *, nctid, label, phase, diseases, drugs, smiless, icd_lists, split_tag) -> pd.DataFrame:
    out = pd.DataFrame({
        "nctid": nctid.astype(str),
        "status": "",
        "why_stop": "",
        "label": label.astype(int),
        "phase": phase,
        "diseases": diseases.map(lambda s: _pylist_repr([s] if s else [])),
        "icdcodes": icd_lists.map(_pylist_repr),
        "drugs": drugs.map(lambda s: _pylist_repr([s] if s else [])),
        "smiless": smiless.map(lambda s: _pylist_repr([str(s)])),
        "criteria": "",  # eligibility-less / feature parity with the model
        "split": split_tag,
    })
    return out[out["split"] != ""].reset_index(drop=True)


def _build_candidate_frame(config: ModelingConfig) -> pd.DataFrame:
    """One row per resolved candidate; label = eventual approval (end-to-end LOA)."""
    cand = load_candidate_detail(config.candidate_detail_path)
    cand = apply_label(cand, config.label)  # outcome -> y (1=approved); drops Ongoing/excluded
    n0 = len(cand)

    smiles_col = "smiles_canonical" if "smiles_canonical" in cand.columns else "smiles"
    smiles = cand[smiles_col]
    has_smiles = smiles.notna() & (smiles.astype(str).str.strip() != "")
    icd_lists = cand.get("icd10_codes", pd.Series([None] * len(cand))).map(_as_str_list)
    has_icd = icd_lists.map(len) > 0

    keep = has_smiles & has_icd
    n_no_smiles = int((~has_smiles).sum())
    n_no_icd = int((has_smiles & ~has_icd).sum())
    cand = cand.loc[keep].reset_index(drop=True)
    icd_lists = icd_lists.loc[keep].reset_index(drop=True)
    smiles = smiles.loc[keep].reset_index(drop=True)

    split_tag = _split_tags(cand, config)

    def col(name):
        return cand[name].fillna("").astype(str) if name in cand.columns else pd.Series([""] * len(cand))

    out = _emit(
        cand,
        nctid=cand["candidate_id"],               # candidate_id is the join key
        label=cand["y"],
        phase=col("highest_phase"),               # unused by HINT's dataloader; informational
        diseases=col("indication"),
        drugs=col("drug_name"),
        smiless=smiles,
        icd_lists=icd_lists,
        split_tag=split_tag,
    )
    n_train = int((out["split"] == "train").sum())
    n_test = int((out["split"] == "test").sum())
    logger.info(
        "hint export (drug_indication): %d resolved candidates -> %d emitted (train=%d, test=%d); "
        "dropped no_smiles=%d no_icd=%d no_year=%d; label=approval; criteria=blanked",
        n0, len(out), n_train, n_test, n_no_smiles, n_no_icd, len(cand) - len(out),
    )
    return out


def _build_trial_frame(config: ModelingConfig) -> pd.DataFrame:
    """One row per NCT; label = trial_inferred_label (phase-transition task)."""
    trials = load_trial_detail(config.trial_detail_path)
    trials = trials[trials["trial_inferred_label"].notna()].copy()
    trials["y"] = trials["trial_inferred_label"].astype(np.int8)

    cand = load_candidate_detail(config.candidate_detail_path)
    cand_cols = ["candidate_id"]
    for c in ("smiles_canonical", "smiles", "icd10_codes"):
        if c in cand.columns:
            cand_cols.append(c)
    df = trials.merge(cand[cand_cols], on="candidate_id", how="left")
    n0 = len(df)

    smiles_col = "smiles_canonical" if "smiles_canonical" in df.columns else "smiles"
    smiles = df[smiles_col]
    has_smiles = smiles.notna() & (smiles.astype(str).str.strip() != "")
    icd_lists = df.get("icd10_codes", pd.Series([None] * len(df))).map(_as_str_list)
    has_icd = icd_lists.map(len) > 0

    keep = has_smiles & has_icd
    n_no_smiles = int((~has_smiles).sum())
    n_no_icd = int((has_smiles & ~has_icd).sum())
    df = df.loc[keep].reset_index(drop=True)
    icd_lists = icd_lists.loc[keep].reset_index(drop=True)
    smiles = smiles.loc[keep].reset_index(drop=True)

    split_tag = _split_tags(df, config)

    def col(name):
        return df[name].fillna("").astype(str) if name in df.columns else pd.Series([""] * len(df))

    diseases = col("candidate_indication") if "candidate_indication" in df.columns else col("indication")
    drugs = col("candidate_drug") if "candidate_drug" in df.columns else col("drug_name")

    out = _emit(
        df,
        nctid=df["nct_id"],
        label=df["y"],
        phase=col("trial_phase"),
        diseases=diseases,
        drugs=drugs,
        smiless=smiles,
        icd_lists=icd_lists,
        split_tag=split_tag,
    )
    n_no_year = n0 - n_no_smiles - n_no_icd - len(out)
    n_train = int((out["split"] == "train").sum())
    n_test = int((out["split"] == "test").sum())
    logger.info(
        "hint export (trial): %d labeled trials -> %d emitted (train=%d, test=%d); "
        "dropped no_smiles=%d no_icd=%d no_year=%d; criteria=blanked",
        n0, len(out), n_train, n_test, n_no_smiles, n_no_icd, n_no_year,
    )
    return out


def write_hint_csv(config: ModelingConfig, out_path: Path) -> pd.DataFrame:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = build_hint_frame(config)
    frame.to_csv(out_path, index=False)
    logger.info("wrote HINT export -> %s (%d rows)", out_path, len(frame))
    return frame
