"""Compare this repo's model against a retrained HINT model on the SAME test set.

Inner-joins the model's `predictions.csv` (nct_id + trial_phase + y_proba) with
HINT's per-NCT test predictions (nctid + y_proba) and recomputes per-phase +
overall metrics for both models on the identical set of joined nct_ids — the
honest "evaluate on the overlap" surface.

HINT predictions can be supplied either as a CSV (columns nctid, y_proba
[, y_true]) or as HINT's native `results/nctid2predict.pkl` ({nctid: proba}).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate
from .evaluate import PHASE_TRANSITIONS

logger = logging.getLogger(__name__)


def _load_hint_predictions(path: Path, key: str) -> pd.DataFrame:
    """Load HINT per-key predictions; the key column is renamed to `key` to match ours."""
    path = Path(path)
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return pd.DataFrame({key: list(d.keys()), "hint_proba": [float(v) for v in d.values()]})
    df = pd.read_csv(path)
    src = "nctid" if "nctid" in df.columns else ("nct_id" if "nct_id" in df.columns else "candidate_id")
    proba = "y_proba" if "y_proba" in df.columns else "hint_proba"
    return df.rename(columns={src: key, proba: "hint_proba"})[[key, "hint_proba"]]


def _metrics_pair(sub: pd.DataFrame, bootstrap_ci: int, seed: int) -> dict:
    """Metrics for both models on the same rows of `sub`."""
    y = sub["y_true"].values
    mine = evaluate.metrics(y, sub["y_proba"].values, bootstrap_ci=bootstrap_ci, ci_seed=seed)
    hint = evaluate.metrics(y, sub["hint_proba"].values, bootstrap_ci=bootstrap_ci, ci_seed=seed)
    return {"n": len(sub), "n_pos": int(y.sum()), "mine": mine, "hint": hint}


def compare(
    predictions_path: Path,
    hint_predictions_path: Path,
    *,
    bootstrap_ci: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a tidy per-row (level × model) comparison DataFrame."""
    mine = pd.read_csv(predictions_path)
    # Join key: nct_id for the trial task, candidate_id for the end-to-end task.
    key = "nct_id" if "nct_id" in mine.columns else "candidate_id"
    if key not in mine.columns:
        raise ValueError(f"{predictions_path} has neither nct_id nor candidate_id")
    hint = _load_hint_predictions(hint_predictions_path, key)

    merged = mine.merge(hint, on=key, how="inner")
    logger.info(
        "joined %d model preds × %d HINT preds -> %d shared %ss",
        len(mine), len(hint), len(merged), key,
    )
    if merged.empty:
        raise ValueError(f"no shared {key}s between model and HINT predictions")

    rows: list[dict] = []

    def emit(level: str, sub: pd.DataFrame) -> None:
        if sub.empty:
            return
        pair = _metrics_pair(sub, bootstrap_ci, seed)
        for model_name in ("mine", "hint"):
            m = pair[model_name]
            rows.append({
                "level": level,
                "model": "drug-success-lite" if model_name == "mine" else "hint",
                "n": pair["n"],
                "n_pos": pair["n_pos"],
                "roc_auc": m.get("roc_auc"),
                "pr_auc": m.get("pr_auc"),
                "f1": m.get("f1"),
                "brier": m.get("brier"),
            })

    emit("overall", merged)
    if "trial_phase" in merged.columns:
        for phase_value, label in PHASE_TRANSITIONS.items():
            emit(label, merged[merged["trial_phase"] == phase_value])

    return pd.DataFrame(rows)


def print_comparison(df: pd.DataFrame) -> None:
    cols = ["level", "model", "n", "n_pos", "roc_auc", "pr_auc", "f1", "brier"]
    widths = {c: len(c) for c in cols}
    fmt = []
    for _, r in df.iterrows():
        fr = {}
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                fr[c] = "nan" if v != v else f"{v:.4f}"
            else:
                fr[c] = "" if v is None else str(v)
            widths[c] = max(widths[c], len(fr[c]))
        fmt.append(fr)
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for fr in fmt:
        print("  ".join(fr[c].ljust(widths[c]) for c in cols))
