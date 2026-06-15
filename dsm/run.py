"""The one entry point: resolve an experiment, run it, write standardized metrics.

`run_experiment(name)`:
  1. materialize the dataset (canonical example parquet) if needed,
  2. dispatch to the model adapter -> runs/<name>/predictions.parquet,
  3. evaluate -> runs/<name>/metrics.json (overall + per-phase, one schema).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from . import evaluate
from .config import PROJECT_ROOT
from .datasets import materialize
from .experiments import DATASETS, EXPERIMENTS, ExperimentSpec
from .models import run_model

logger = logging.getLogger(__name__)

RUNS_DIR = PROJECT_ROOT / "runs"


def run_experiment(
    name: str,
    *,
    output_root: Optional[Path] = None,
    epochs: Optional[int] = None,
    bootstrap_ci: int = 1000,
    force_materialize: bool = False,
) -> dict:
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}; known: {sorted(EXPERIMENTS)}")
    spec: ExperimentSpec = EXPERIMENTS[name]
    out_dir = Path(output_root or RUNS_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.parquet"

    dataset_path = None
    if spec.dataset:
        dataset_path = materialize(DATASETS[spec.dataset], force=force_materialize)

    logger.info("=== experiment %s: model=%s features=%s ===", name, spec.model, spec.features)
    run_model(
        spec.model,
        dataset_path=dataset_path,
        features=spec.features,
        out_path=preds_path,
        native_benchmark=spec.native_benchmark,
        epochs=epochs or spec.epochs,
        class_weight=spec.class_weight,
        pca=spec.pca,
        calibration_folds=spec.calibration_folds,
    )

    payload = {
        "experiment": name,
        "model": spec.model,
        "features": list(spec.features),
        "dataset": spec.dataset,
        "native_benchmark": spec.native_benchmark,
        **evaluate.evaluate_predictions(preds_path, bootstrap_ci=bootstrap_ci),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    o = payload["overall"]
    logger.info("%s: ROC-AUC=%.4f PR-AUC=%.4f F1=%.4f (n=%d) -> %s",
                name, o["roc_auc"], o["pr_auc"], o["f1"], payload["n"], out_dir / "metrics.json")
    return payload


def collect_results(output_root: Optional[Path] = None) -> list[dict]:
    """Scan runs/*/metrics.json into flat rows (overall metrics per experiment), then fold in the
    embed_swap models, which aren't registered experiments and so have no metrics.json."""
    root = Path(output_root or RUNS_DIR)
    rows: list[dict] = []
    for mj in sorted(root.glob("*/metrics.json")):
        try:
            d = json.loads(mj.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        o = d.get("overall", {})
        rows.append({
            "experiment": d.get("experiment", mj.parent.name),
            "model": d.get("model", ""),
            "dataset": d.get("dataset") or (f"native:{d.get('native_benchmark')}"
                                            if d.get("native_benchmark") else ""),
            "n": d.get("n"),
            "n_pos": d.get("n_pos"),
            "roc_auc": o.get("roc_auc"),
            "pr_auc": o.get("pr_auc"),
            "f1": o.get("f1"),
            **{f"{m}_{b}": o.get(f"{m}_{b}")
               for m in ("roc_auc", "pr_auc", "f1") for b in ("lo", "hi")},
        })
    rows.extend(_embed_swap_rows(root))
    return rows


# embed_swap target -> the dataset it ran on.
_EMBED_SWAP_DATASET = {"p1": "hint_p1", "p2": "hint_p2", "p3": "hint_p3", "di": "ours_di"}


def _embed_swap_rows(root: Path) -> list[dict]:
    """Rows for the embed_swap-specific models (xgb_hint_emb, xgb_pca50) from
    runs/embed_swap_summary.csv. The hint/xgb_full baselines there are already covered by their
    registered experiments, so only these two are added. Uses the 'all' stratum (= overall)."""
    import csv

    path = root / "embed_swap_summary.csv"
    if not path.exists():
        return []

    def num(v, cast):
        try:
            return cast(float(v))
        except (TypeError, ValueError):
            return None

    out: list[dict] = []
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("stratum") != "all" or r.get("model") not in (
                        "xgb_hint_emb", "xgb_pca50",
                        "xgb_hint_emb_mdt", "xgb_pca50_mdt",
                        "xgb_hint_emb_mdg", "xgb_pca50_mdg",
                        "xgb_hint_emb_mdtp", "xgb_pca50_mdtp"):
                    continue
                target = r.get("target", "")
                out.append({
                    "experiment": f"{r['model']}_{target}",
                    "model": r["model"],
                    "dataset": _EMBED_SWAP_DATASET.get(target, target),
                    "n": num(r.get("n"), int),
                    "n_pos": num(r.get("n_pos"), int),
                    "roc_auc": num(r.get("roc_auc"), float),
                    "pr_auc": num(r.get("pr_auc"), float),
                    "f1": num(r.get("f1"), float),
                    **{f"{m}_{b}": num(r.get(f"{m}_{b}"), float)
                       for m in ("roc_auc", "pr_auc", "f1") for b in ("lo", "hi")},
                })
    except (OSError, KeyError, csv.Error):
        return []
    return out


def reeval_all(*, output_root: Optional[Path] = None, bootstrap_ci: int = 1000) -> tuple[int, int]:
    """Recompute metrics (with bootstrap CIs) from already-saved predictions — NO retraining.

    Rewrites runs/<name>/metrics.json for every registered experiment that has a saved
    predictions.parquet, then refreshes runs/embed_swap_summary.csv from the saved embed_swap
    predictions. Returns (n_registered, n_embed_swap_rows)."""
    root = Path(output_root or RUNS_DIR)
    n_reg = 0
    for name, spec in EXPERIMENTS.items():
        preds_path = root / name / "predictions.parquet"
        if not preds_path.exists():
            continue
        payload = {
            "experiment": name,
            "model": spec.model,
            "features": list(spec.features),
            "dataset": spec.dataset,
            "native_benchmark": spec.native_benchmark,
            **evaluate.evaluate_predictions(preds_path, bootstrap_ci=bootstrap_ci),
        }
        (root / name / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
        n_reg += 1
    from .embed_swap import rescore_from_saved  # lazy: embed_swap imports this module
    n_embed = rescore_from_saved(bootstrap_ci=bootstrap_ci)
    return n_reg, n_embed


def materialize_dataset(name: str, *, force: bool = False) -> Path:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    return materialize(DATASETS[name], force=force)
