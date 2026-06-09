"""Dataclass configs for the modeling pipeline.

Single source of truth for paths, label policy, feature toggles, model choice,
and split knobs. The CLI builds a `ModelingConfig` and threads it through
`train_one_run`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_CANDIDATE_DETAIL = DATA_ROOT / "candidate_detail.parquet"
DEFAULT_TRIAL_DETAIL = DATA_ROOT / "trial_detail.parquet"
DEFAULT_FINGERPRINTS = DATA_ROOT / "features" / "fingerprints.parquet"
DEFAULT_EMBEDDINGS = DATA_ROOT / "features" / "molformer_embeddings.parquet"

# Canonical example datasets (materialized by dsm/datasets.py) + the HINT
# TOP benchmark (a symlink to hint/data, where HINT's native scripts read it).
DATASETS_DIR = DATA_ROOT / "datasets"
HINT_BENCHMARK_DIR = DATA_ROOT / "hint_benchmark"

# The five top-level composite feature groups (see `features.py`).
ALL_FEATURE_GROUPS: tuple[str, ...] = (
    "molecule",
    "disease",
    "admet",
    "target",
    "pathway",
)


@dataclass
class LabelConfig:
    """Binary label derived from `candidate_detail.outcome`.

    positive = {Approved, Commercialized}, negative = the three failure phases.
    Rows with outcome in `exclude_outcomes` are dropped. Only `drug_indication`
    granularity uses this; `trial` granularity labels come straight from
    `trial_inferred_label`.

    `ongoing_policy` governs right-censored `Ongoing` programs:
      "drop" - exclude them (default; reproduces the historical cohort)
      "fail" - label negative (Manski lower bound on the approval rate)
    """

    positive: tuple[str, ...] = ("Approved", "Commercialized")
    negative: tuple[str, ...] = ("Failed Phase 1", "Failed Phase 2", "Failed Phase 3")
    exclude_outcomes: tuple[str, ...] = ("Unknown", "Ongoing")
    ongoing_policy: str = "drop"  # "drop" | "fail"
    ongoing_label: str = "Ongoing"


@dataclass
class FeatureConfig:
    """Which top-level feature groups to assemble.

    Top-K cardinalities and ADMET thresholds live in the composite group
    definitions (`features.py`), not here.
    """

    enabled: tuple[str, ...] = ALL_FEATURE_GROUPS
    # The molecule group uses exactly one structural representation: the ECFP4/
    # MACCS "fingerprint" (default) or the MoLFormer "embedding".
    molecule_repr: str = "fingerprint"  # "fingerprint" | "embedding"


@dataclass
class ModelingConfig:
    """Top-level config for a single training run."""

    candidate_detail_path: Path = DEFAULT_CANDIDATE_DETAIL
    trial_detail_path: Path = DEFAULT_TRIAL_DETAIL
    fingerprints_path: Path = DEFAULT_FINGERPRINTS
    embeddings_path: Path = DEFAULT_EMBEDDINGS

    label: LabelConfig = field(default_factory=LabelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    # "drug_indication" → one row per candidate, y from candidate.outcome.
    # "trial" → one row per NCT, y from trial_detail.trial_inferred_label.
    training_granularity: str = "drug_indication"

    model_name: str = "xgb"
    model_kwargs: dict = field(default_factory=dict)

    test_size: float = 0.2
    inner_val_size: float = 0.1
    seed: int = 0

    # Temporal split: train on rows whose `time_split_column` year is
    # <= time_split_year, test on rows whose year is > it. When
    # `time_split_year` is None, falls back to a stratified random split.
    time_split_column: str = "earliest_start_date"
    time_split_year: Optional[int] = None

    # Bootstrap-percentile CIs on ROC-AUC / PR-AUC / F1 (0 = off).
    bootstrap_ci: int = 0

    output_dir: Optional[Path] = None
