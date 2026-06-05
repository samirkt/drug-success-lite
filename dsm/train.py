"""Single training run — assemble feature matrix, fit model, evaluate, write.

`train_one_run(config)` loads the frame, splits, fits the enabled feature groups
on the inner-train slice, trains the model, and returns a `RunResult`.
`write_run(result, out_dir)` persists metrics + predictions + importances.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import dataset as data_mod
from . import evaluate, splits
from .config import ModelingConfig
from .features import FEATURE_GROUPS, FeatureGroup, build_group
from .model import build_model

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    config: ModelingConfig
    groups: tuple[str, ...]
    feature_names: list[str]
    n_features: int
    n_train: int
    n_test: int
    train_pos: int
    test_pos: int
    metrics: dict
    per_phase_metrics: dict = field(default_factory=dict)
    feature_importances: Optional[np.ndarray] = None
    test_predictions: Optional[pd.DataFrame] = None
    fitted_groups: list[FeatureGroup] = field(default_factory=list)
    fitted_model: object = None  # ModelProtocol


def _instantiate_groups(config: ModelingConfig) -> list[FeatureGroup]:
    """Build the enabled composite groups (top-K / thresholds baked into them)."""
    feats = config.features
    groups: list[FeatureGroup] = []
    for name in feats.enabled:
        kwargs: dict = {}
        if name == "molecule":
            kwargs["molecule_repr"] = feats.molecule_repr
        groups.append(build_group(name, **kwargs))
    return groups


def _stack(matrices: list[np.ndarray]) -> np.ndarray:
    nonempty = [m for m in matrices if m is not None and m.shape[1] > 0]
    if not nonempty:
        raise ValueError("no enabled feature group produced any columns")
    return np.hstack(nonempty)


def train_one_run(
    config: ModelingConfig,
    *,
    df: Optional[pd.DataFrame] = None,
) -> RunResult:
    """Run one full train/eval cycle."""
    if not config.features.enabled:
        raise ValueError("config.features.enabled is empty — nothing to train on")
    for name in config.features.enabled:
        if name not in FEATURE_GROUPS:
            raise KeyError(f"unknown feature group {name!r}")

    if df is None:
        df = data_mod.build_modeling_frame(config)
    train_idx, test_idx = splits.split(
        df,
        test_size=config.test_size,
        seed=config.seed,
        time_split_column=config.time_split_column,
        time_split_year=config.time_split_year,
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    y_train = train_df["y"].values.astype(int)
    y_test = test_df["y"].values.astype(int)

    # Inner-val slice (off training set) for early stopping.
    from sklearn.model_selection import train_test_split

    inner_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=config.inner_val_size,
        stratify=y_train,
        random_state=config.seed,
    )
    inner_df = train_df.iloc[inner_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    y_inner = y_train[inner_idx]
    y_val = y_train[val_idx]

    # Fit groups on inner-train (so val + test stay held out).
    groups = _instantiate_groups(config)
    for g in groups:
        if not g.is_available(inner_df):
            logger.warning("feature group %r unavailable on the frame — empty output", g.name)
    feat_train, feat_val, feat_test = [], [], []
    feature_names: list[str] = []
    group_widths: list[tuple[str, int]] = []
    for g in groups:
        g.fit(inner_df, y_inner)
        m_train = g.transform(inner_df)
        feat_train.append(m_train)
        feat_val.append(g.transform(val_df))
        feat_test.append(g.transform(test_df))
        feature_names.extend(g.feature_names())
        group_widths.append((g.name, m_train.shape[1]))
        logger.info("group %s: n_features=%d", g.name, m_train.shape[1])

    X_train = _stack(feat_train)
    X_val = _stack(feat_val)
    X_test = _stack(feat_test)
    n_features = X_train.shape[1]
    logger.info("assembled feature matrix: train=%s val=%s test=%s", X_train.shape, X_val.shape, X_test.shape)

    # Build + fit model.
    n_pos = int(y_inner.sum())
    n_neg = len(y_inner) - n_pos
    spw = (n_neg / n_pos) if n_pos > 0 else 1.0
    model = build_model(
        config.model_name,
        scale_pos_weight=spw,
        random_state=config.seed,
        **config.model_kwargs,
    )
    model.fit(X_train, y_inner, X_val=X_val, y_val=y_val)

    y_proba = model.predict_proba(X_test)[:, 1]
    m = evaluate.metrics(y_test, y_proba, bootstrap_ci=config.bootstrap_ci, ci_seed=config.seed)

    # Per-phase test metrics for trial-granularity runs (no-op otherwise).
    per_phase_metrics: dict = {}
    if "trial_phase" in test_df.columns:
        per_phase_metrics = evaluate.metrics_by_phase(
            y_test,
            y_proba,
            test_df["trial_phase"].values,
            bootstrap_ci=config.bootstrap_ci,
            ci_seed=config.seed,
        )
        for label, mp in per_phase_metrics.items():
            logger.info(
                "phase %s: n=%d pos=%d roc_auc=%.4f pr_auc=%.4f",
                label, mp.get("n", 0), mp.get("n_pos", 0),
                mp.get("roc_auc", float("nan")), mp.get("pr_auc", float("nan")),
            )

    # Predictions table — keyed on nct_id + trial_phase for trial runs (HINT-ready).
    pred_cols: dict = {}
    if "nct_id" in test_df.columns:
        pred_cols["nct_id"] = test_df["nct_id"].values
    if "trial_phase" in test_df.columns:
        pred_cols["trial_phase"] = test_df["trial_phase"].values
    pred_cols["candidate_id"] = test_df["candidate_id"].values
    pred_cols["y_true"] = y_test
    pred_cols["y_proba"] = y_proba
    pred_df = pd.DataFrame(pred_cols)

    return RunResult(
        config=config,
        groups=tuple(g.name for g in groups),
        feature_names=feature_names,
        n_features=n_features,
        n_train=len(train_df),
        n_test=len(test_df),
        train_pos=int(y_train.sum()),
        test_pos=int(y_test.sum()),
        metrics={**m, "group_widths": dict(group_widths)},
        per_phase_metrics=per_phase_metrics,
        feature_importances=model.feature_importances(),
        test_predictions=pred_df,
        fitted_groups=groups,
        fitted_model=model,
    )


def _jsonable(config: ModelingConfig) -> dict:
    """asdict() with Paths stringified so json.dump accepts it."""
    d = asdict(config)
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
    return d


def write_run(result: RunResult, out_dir: Path) -> None:
    """Persist metrics.json, predictions.csv, feature_importances.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "granularity": result.config.training_granularity,
        "groups": list(result.groups),
        "model": result.config.model_name,
        "n_features": result.n_features,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "train_pos": result.train_pos,
        "test_pos": result.test_pos,
        "metrics": result.metrics,
        "per_phase_metrics": result.per_phase_metrics,
        "config": _jsonable(result.config),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))

    if result.test_predictions is not None:
        result.test_predictions.to_csv(out_dir / "predictions.csv", index=False)

    fi = result.feature_importances
    if fi is not None and len(fi) == len(result.feature_names):
        pd.DataFrame({"feature": result.feature_names, "importance": fi}).sort_values(
            "importance", ascending=False
        ).to_csv(out_dir / "feature_importances.csv", index=False)

    logger.info("wrote run artifacts to %s", out_dir)
