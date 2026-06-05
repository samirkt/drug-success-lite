"""Load + join + label the modeling frame (read-only over the immutable inputs).

Joins `candidate_detail.parquet` (left) with `fingerprints.parquet` and
`molformer_embeddings.parquet` on `candidate_id`. Rows missing fingerprints or
embeddings are kept (NaN); per-group encoders handle the missing case via a
`_missing` indicator column. Applies the binary label policy from `LabelConfig`.

Nothing in this module writes to `inputs/`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LabelConfig, ModelingConfig

logger = logging.getLogger(__name__)


def load_candidate_detail(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "candidate_id" not in df.columns:
        raise ValueError(f"{path} has no candidate_id column")
    return df


def load_trial_detail(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for col in ("candidate_id", "nct_id", "trial_inferred_label"):
        if col not in df.columns:
            raise ValueError(f"{path} has no {col!r} column")
    return df


def load_fingerprints(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("fingerprints parquet not found at %s — fingerprint coverage will be 0", path)
        return pd.DataFrame(columns=["candidate_id", "ecfp4", "maccs"])
    df = pd.read_parquet(path)
    return df[["candidate_id", "ecfp4", "maccs"]]


def load_embeddings(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("embeddings parquet not found at %s — embedding coverage will be 0", path)
        return pd.DataFrame(columns=["candidate_id", "embedding"])
    df = pd.read_parquet(path)
    return df[["candidate_id", "embedding"]]


def binary_label_with_ongoing(df: pd.DataFrame, label: LabelConfig) -> pd.Series:
    """Float label Series (1.0 / 0.0 / nan) over ``df`` rows; nan = drop.

    Maps positive/negative outcomes, then applies ``label.ongoing_policy`` to the
    Ongoing rows ("drop" leaves them nan; "fail" labels them 0.0).
    """
    outcome = df["outcome"].astype("string")
    y = pd.Series(np.nan, index=df.index)
    y[outcome.isin(set(label.positive))] = 1.0
    y[outcome.isin(set(label.negative))] = 0.0

    ongoing = outcome.eq(label.ongoing_label)
    policy = (label.ongoing_policy or "drop").lower()
    if policy == "drop":
        pass  # ongoing stays nan -> dropped
    elif policy == "fail":
        y[ongoing] = 0.0
    else:
        raise ValueError(f"unknown ongoing_policy={policy!r} (expected drop|fail)")
    return y


def apply_label(df: pd.DataFrame, label: LabelConfig) -> pd.DataFrame:
    """Filter rows by `outcome` and attach a binary `y` column.

    Hard exclusions (`exclude_outcomes` other than the Ongoing marker) are always
    dropped. Ongoing programs are governed by `label.ongoing_policy`. Rows that end
    up unlabeled (nan) are dropped.
    """
    if "outcome" not in df.columns:
        raise ValueError("candidate_detail has no `outcome` column")

    n_before = len(df)
    hard_exclude = set(label.exclude_outcomes) - {label.ongoing_label}
    df = df[~df["outcome"].isin(hard_exclude)].copy()

    y = binary_label_with_ongoing(df, label)
    keep = y.notna()
    df = df.loc[keep].copy()
    df["y"] = y.loc[keep].astype(np.int8)

    n_pos = int(df["y"].sum())
    n_neg = len(df) - n_pos
    logger.info(
        "label: ongoing_policy=%s; filtered %d -> %d rows; positive=%d (%.1f%%) negative=%d",
        label.ongoing_policy,
        n_before,
        len(df),
        n_pos,
        100.0 * n_pos / max(len(df), 1),
        n_neg,
    )
    return df


def build_modeling_frame(config: ModelingConfig) -> pd.DataFrame:
    """Dispatch to the candidate- or trial-level builder by granularity."""
    granularity = (config.training_granularity or "drug_indication").lower()
    if granularity == "trial":
        return build_trial_modeling_frame(config)
    if granularity == "drug_indication":
        return build_candidate_modeling_frame(config)
    raise ValueError(
        f"unknown training_granularity={config.training_granularity!r}; "
        "expected 'drug_indication' or 'trial'"
    )


def build_candidate_modeling_frame(config: ModelingConfig) -> pd.DataFrame:
    """Join candidate_detail with fingerprints & embeddings; apply label."""
    cand = load_candidate_detail(config.candidate_detail_path)
    fps = load_fingerprints(config.fingerprints_path)
    embs = load_embeddings(config.embeddings_path)

    df = cand.merge(fps, on="candidate_id", how="left")
    df = df.merge(embs, on="candidate_id", how="left")

    n_with_fp = df["ecfp4"].notna().sum() if "ecfp4" in df.columns else 0
    n_with_emb = df["embedding"].notna().sum() if "embedding" in df.columns else 0
    logger.info(
        "joined: %d candidates; fingerprints=%d (%.1f%%) embeddings=%d (%.1f%%)",
        len(df),
        n_with_fp,
        100.0 * n_with_fp / max(len(df), 1),
        n_with_emb,
        100.0 * n_with_emb / max(len(df), 1),
    )

    df = apply_label(df, config.label)
    return df.reset_index(drop=True)


def build_trial_modeling_frame(config: ModelingConfig) -> pd.DataFrame:
    """One row per NCT × primary-candidate; y = `trial_inferred_label`.

    Drops trials with a null inferred label (ongoing / phase-not-reached), then
    joins the candidate feature columns from `candidate_detail` plus fingerprints
    + embeddings on `candidate_id` so the feature groups operate unmodified.
    """
    trials = load_trial_detail(config.trial_detail_path)
    n_before = len(trials)
    trials = trials[trials["trial_inferred_label"].notna()].copy()
    trials["y"] = trials["trial_inferred_label"].astype(np.int8)
    n_pos = int(trials["y"].sum())
    n_neg = len(trials) - n_pos
    logger.info(
        "trial label: filtered %d -> %d trials; positive=%d (%.1f%%) negative=%d",
        n_before,
        len(trials),
        n_pos,
        100.0 * n_pos / max(len(trials), 1),
        n_neg,
    )

    cand = load_candidate_detail(config.candidate_detail_path)
    # Resolve column overlap: the trial frame already carries candidate_drug,
    # candidate_indication, etc. — prefer those and drop the matching
    # candidate-side columns so `merge` doesn't suffix. `outcome` (candidate-level)
    # is retained — it differs from the trial-level `y`.
    overlap = (set(trials.columns) & set(cand.columns)) - {"candidate_id"}
    cand = cand.drop(columns=list(overlap), errors="ignore")

    df = trials.merge(cand, on="candidate_id", how="left")
    df = df.merge(load_fingerprints(config.fingerprints_path), on="candidate_id", how="left")
    df = df.merge(load_embeddings(config.embeddings_path), on="candidate_id", how="left")

    n_with_fp = df["ecfp4"].notna().sum() if "ecfp4" in df.columns else 0
    n_with_emb = df["embedding"].notna().sum() if "embedding" in df.columns else 0
    logger.info(
        "joined trial frame: %d trials across %d candidates; fingerprints=%d (%.1f%%) embeddings=%d (%.1f%%)",
        len(df),
        df["candidate_id"].nunique(),
        n_with_fp,
        100.0 * n_with_fp / max(len(df), 1),
        n_with_emb,
        100.0 * n_with_emb / max(len(df), 1),
    )
    return df.reset_index(drop=True)
