"""Declarative registry: datasets and experiments.

An experiment is one (dataset × model × features × split) cell. To add one, add
an entry here — no new code. Run with `python -m dsm run <name>`.

Feature vocabulary:
  - sklearn (xgb/logreg): molecule, disease, admet, target, pathway
    (molecule = ECFP4+MACCS from SMILES; disease = ICD-code multi-hot on every
     dataset; admet/target/pathway = our data only).
  - hint: mol, disease[, criteria]  (mol = MPNN, disease = GRAM, criteria = BioBERT
    protocol encoder; without `criteria` it loads the empty stub embedding).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .datasets import DatasetSpec

ALL_GROUPS = ("molecule", "disease", "admet", "target", "pathway")

# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
DATASETS: dict[str, DatasetSpec] = {
    "ours_di": DatasetSpec(name="ours_di", kind="dsm",
                           granularity="drug_indication", time_split_year=2019),
    "ours_trial": DatasetSpec(name="ours_trial", kind="dsm",
                              granularity="trial", time_split_year=2019),
    "hint_p1": DatasetSpec(name="hint_p1", kind="hint_benchmark", phase_stem="phase_I"),
    "hint_p2": DatasetSpec(name="hint_p2", kind="hint_benchmark", phase_stem="phase_II"),
    "hint_p3": DatasetSpec(name="hint_p3", kind="hint_benchmark", phase_stem="phase_III"),
}


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    model: str                                   # "xgb" | "logreg" | "hint"
    features: tuple[str, ...]
    dataset: Optional[str] = None                # key into DATASETS (canonical path)
    native_benchmark: Optional[str] = None       # phase stem (HINT native repro path)
    epochs: int = 5                              # hint only
    class_weight: bool = False                   # hint only: pos_weight=n_neg/n_pos in the BCE loss
    pca: Optional[int] = None                    # sklearn only: PCA-<n> per feature group (bottleneck)

    def __post_init__(self):
        if not (self.dataset or self.native_benchmark):
            raise ValueError(f"{self.name}: needs `dataset` or `native_benchmark`")


def _e(*args, **kwargs) -> ExperimentSpec:
    s = ExperimentSpec(*args, **kwargs)
    return s


EXPERIMENTS: dict[str, ExperimentSpec] = {s.name: s for s in [
    # --- our data: best model + feature-matched HINT ---
    _e("xgb_di_2019", "xgb", ALL_GROUPS, dataset="ours_di"),
    _e("xgb_di_md", "xgb", ("molecule", "disease"), dataset="ours_di"),
    _e("hint_di_2019", "hint", ("mol", "disease"), dataset="ours_di", class_weight=True),
    # ChemAP (pretrained black box, SMILES-only) on the aligned approval target.
    _e("chemap_di_2019", "chemap", ("mol",), dataset="ours_di"),

    # --- benchmark, canonical comparison (identical population & inputs) ---
    _e("xgb_bench_p1", "xgb", ("molecule", "disease"), dataset="hint_p1", pca=50),
    _e("xgb_bench_p2", "xgb", ("molecule", "disease"), dataset="hint_p2", pca=50),
    _e("xgb_bench_p3", "xgb", ("molecule", "disease"), dataset="hint_p3", pca=50),
    _e("hint_bench_p1", "hint", ("mol", "disease"), dataset="hint_p1"),
    _e("hint_bench_p2", "hint", ("mol", "disease"), dataset="hint_p2"),
    _e("hint_bench_p3", "hint", ("mol", "disease"), dataset="hint_p3"),

    # --- benchmark reproduction (HINT native path, all features) ---
    _e("hint_bench_p1_repro", "hint", ("mol", "disease", "criteria"), native_benchmark="phase_I"),
    _e("hint_bench_p2_repro", "hint", ("mol", "disease", "criteria"), native_benchmark="phase_II"),
    _e("hint_bench_p3_repro", "hint", ("mol", "disease", "criteria"), native_benchmark="phase_III"),

    # --- PCA-50-per-group ablation family on ours_di ---
    # every feature group reduced to <=50-d (matching the embed-swap xgb_pca50 control), so the whole
    # family shares one representation; disease uses the category-level ICD encoding. m=molecule
    # d=disease t=target p=pathway a=admet (mdtpa = all 5 = FULL reference). abl_md is the served model.
    # single-group:
    *[_e(f"abl_{g}", "xgb", (g,), dataset="ours_di", pca=50) for g in ALL_GROUPS],
    # alternate target representation: raw drug->target UniProt multi-hot vs the engineered `target`.
    _e("abl_target_genes", "xgb", ("target_genes",), dataset="ours_di", pca=50),
    # cumulative (build up from md):
    _e("abl_md", "xgb", ("molecule", "disease"), dataset="ours_di", pca=50),
    _e("abl_mdt", "xgb", ("molecule", "disease", "target"), dataset="ours_di", pca=50),
    _e("abl_mdp", "xgb", ("molecule", "disease", "pathway"), dataset="ours_di", pca=50),
    _e("abl_mda", "xgb", ("molecule", "disease", "admet"), dataset="ours_di", pca=50),
    _e("abl_mdtp", "xgb", ("molecule", "disease", "target", "pathway"), dataset="ours_di", pca=50),
    _e("abl_mdtpa", "xgb", ("molecule", "disease", "target", "pathway", "admet"),
       dataset="ours_di", pca=50),
]}
