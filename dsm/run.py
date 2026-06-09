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
    bootstrap_ci: int = 0,
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
    """Scan runs/*/metrics.json into flat rows (overall metrics per experiment)."""
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
        })
    return rows


def materialize_dataset(name: str, *, force: bool = False) -> Path:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    return materialize(DATASETS[name], force=force)
