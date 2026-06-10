"""Declarative registry: datasets and experiments.

An experiment is one (dataset × model × features × split) cell. To add one, add
an entry here — no new code. Run with `python -m dsm run <name>`.

Feature vocabulary:
  - sklearn (xgb/logreg): molecule, disease, admet, target, pathway
    (molecule = ECFP4+MACCS from SMILES; disease = MeSH groups on our data /
     ICD multi-hot on the benchmark; admet/target/pathway = our data only).
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
    _e("hint_di_2019", "hint", ("mol", "disease"), dataset="ours_di"),
    # ChemAP (pretrained black box, SMILES-only) on the aligned approval target.
    _e("chemap_di_2019", "chemap", ("mol",), dataset="ours_di"),

    # --- benchmark, canonical comparison (identical population & inputs) ---
    _e("xgb_bench_p1", "xgb", ("molecule", "disease"), dataset="hint_p1"),
    _e("xgb_bench_p2", "xgb", ("molecule", "disease"), dataset="hint_p2"),
    _e("xgb_bench_p3", "xgb", ("molecule", "disease"), dataset="hint_p3"),
    _e("hint_bench_p1", "hint", ("mol", "disease"), dataset="hint_p1"),
    _e("hint_bench_p2", "hint", ("mol", "disease"), dataset="hint_p2"),
    _e("hint_bench_p3", "hint", ("mol", "disease"), dataset="hint_p3"),

    # --- benchmark reproduction (HINT native path, all features) ---
    _e("hint_bench_p1_repro", "hint", ("mol", "disease", "criteria"), native_benchmark="phase_I"),
    _e("hint_bench_p2_repro", "hint", ("mol", "disease", "criteria"), native_benchmark="phase_II"),
    _e("hint_bench_p3_repro", "hint", ("mol", "disease", "criteria"), native_benchmark="phase_III"),

    # --- single-group ablation of the headline model (xgb on ours_di) ---
    # baseline = xgb_di_2019 (all groups); see dsm/ablation.py.
    *[_e(f"abl_{g}", "xgb", (g,), dataset="ours_di") for g in ALL_GROUPS],
]}
