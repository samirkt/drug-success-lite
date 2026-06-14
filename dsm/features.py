"""The five top-level composite feature groups + a tiny registry.

Each group is a `CompositeGroup` — an ordered list of generic sub-encoders (see
`encoders.py`). The unit of selection is one of the five composites: MOLECULE /
DISEASE / ADMET / TARGET / PATHWAY. Top-K cardinalities live here as plain
constants.

Unlike the original repo, there is no per-group dimensionality reduction: a
group's `transform` simply hstacks every encoder's block in order.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .encoders import (
    AdmetPercentiles,
    DenseArray,
    MultiHot,
    OneHot,
    Scalar,
    WeightedMap,
)

logger = logging.getLogger(__name__)

EMB_DIM = 768
ECFP4_DIM = 2048
MACCS_DIM = 167

FEATURE_GROUPS: dict[str, type["FeatureGroup"]] = {}


@runtime_checkable
class FeatureGroup(Protocol):
    name: ClassVar[str]

    def is_available(self, df: pd.DataFrame) -> bool: ...
    def fit(self, df: pd.DataFrame, y=None) -> None: ...
    def transform(self, df: pd.DataFrame) -> np.ndarray: ...
    def feature_names(self) -> list[str]: ...


def register(cls: type[FeatureGroup]) -> type[FeatureGroup]:
    """Class decorator — register a FeatureGroup under its `name`."""
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} missing class-level `name`")
    if cls.name in FEATURE_GROUPS:
        raise ValueError(f"feature group {cls.name!r} already registered")
    FEATURE_GROUPS[cls.name] = cls
    return cls


def build_group(name: str, **kwargs) -> FeatureGroup:
    if name not in FEATURE_GROUPS:
        raise KeyError(f"unknown feature group {name!r}; known: {sorted(FEATURE_GROUPS)}")
    return FEATURE_GROUPS[name](**kwargs)


def _mesh_top_level(tokens: list[str]) -> list[str]:
    """MeSH tree numbers (e.g. `C04.557.470`) → set of top-level prefixes."""
    return sorted({t.split(".")[0] for t in tokens if t})


class CompositeGroup:
    """A feature group assembled from an ordered list of sub-encoders."""

    name: str = ""

    def __init__(self, name: str, encoders: list) -> None:
        self.name = name
        self._encoders = encoders

    def is_available(self, df: pd.DataFrame) -> bool:
        return any(e.is_available(df) for e in self._encoders)

    def fit(self, df: pd.DataFrame, y=None) -> None:
        for e in self._encoders:
            e.fit(df)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks = [e.transform(df) for e in self._encoders]
        blocks = [b for b in blocks if b.shape[1] > 0]
        if not blocks:
            return np.empty((len(df), 0), dtype=np.float32)
        return np.hstack(blocks)

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for e in self._encoders:
            names.extend(e.feature_names())
        return names


@register
class MoleculeGroup(CompositeGroup):
    """Molecular structure features.

    The MoLFormer `embedding` and the ECFP4/MACCS `fingerprint` are two competing
    representations of the same molecule and are mutually exclusive —
    `molecule_repr` selects exactly one. Default is ``"fingerprint"``.
    """

    name = "molecule"

    def __init__(self, molecule_repr: str = "fingerprint") -> None:
        if molecule_repr == "fingerprint":
            encoders = [DenseArray(
                [("ecfp4", ECFP4_DIM), ("maccs", MACCS_DIM)],
                prefix="fingerprints",
                dtype=np.uint8,
                impute="zero",
            )]
        elif molecule_repr == "embedding":
            encoders = [DenseArray([("embedding", EMB_DIM)], prefix="embedding", impute="mean")]
        else:
            raise ValueError(
                f"molecule_repr must be 'fingerprint' or 'embedding', got {molecule_repr!r}"
            )
        self.molecule_repr = molecule_repr
        super().__init__(self.name, encoders)


@register
class DiseaseGroup(CompositeGroup):
    name = "disease"

    def __init__(self) -> None:
        super().__init__(self.name, [
            OneHot("disease_area", prefix="disease_area"),
            MultiHot(
                "mesh_condition_tree_numbers",
                prefix="mesh",
                top_k=200,
                token_fn=_mesh_top_level,
            ),
        ])


@register
class AdmetGroup(CompositeGroup):
    name = "admet"

    def __init__(self) -> None:
        super().__init__(self.name, [AdmetPercentiles()])


@register
class TargetGroup(CompositeGroup):
    name = "target"

    def __init__(self) -> None:
        super().__init__(self.name, [
            # Target availability
            Scalar(["target_count", "has_target_data"]),
            # Mechanism (multi-hot)
            MultiHot("opentargets_moa", prefix="moa", top_k=200),
            MultiHot("opentargets_action_type", prefix="action", top_k=50),
            # Disease-specific genetic target support
            Scalar([
                "max_genetic_score_for_indication",
                "mean_genetic_score_for_indication",
                "num_targets_with_genetic_support",
                "has_genetic_support",
                "has_indication_target_match",
            ]),
            # Target constraint (LOEUF)
            Scalar([
                "min_loeuf_across_targets",
                "mean_loeuf_across_targets",
                "has_loeuf_data",
            ]),
            # Tractability
            MultiHot("opentargets_tractability_modalities", prefix="tract_mod", top_k=10),
            MultiHot("opentargets_tractability_labels", prefix="tract_lbl", top_k=25),
        ])


@register
class TargetGenesGroup(CompositeGroup):
    """Alternate target representation: a top-K multi-hot of the raw drug->target UniProt
    IDs (the `drug_targets` column), and nothing else. Homogeneous binary indicators, in
    contrast to TargetGroup's mixed engineered features (MoA / genetic / LOEUF / tractability)."""

    name = "target_genes"

    def __init__(self) -> None:
        super().__init__(self.name, [
            MultiHot("drug_targets", prefix="drug_target", top_k=500),
        ])


@register
class PathwayGroup(CompositeGroup):
    name = "pathway"

    def __init__(self) -> None:
        super().__init__(self.name, [
            # Reactome pathway context
            Scalar([
                "reactome_has_data",
                "reactome_n_pathways",
                "reactome_n_top_level_pathways",
                "reactome_n_leaf_pathways",
                "reactome_mean_depth",
                "reactome_max_depth",
            ]),
            MultiHot("reactome_top_level_pathway_ids", prefix="pathway_top", top_k=500),
            MultiHot("reactome_ancestor_pathway_ids", prefix="pathway_anc", top_k=500),
            # Disease-aligned pathway context
            WeightedMap("weighted_ancestor_pathway_scores", prefix="pathway_wanc", top_k=500),
            MultiHot("best_supported_target_pathway_ids", prefix="pathway_bst", top_k=500),
            Scalar(["num_pathways_from_supported_targets"]),
        ])


# Sanity: every registered composite satisfies the FeatureGroup protocol.
assert isinstance(MoleculeGroup(), FeatureGroup)
