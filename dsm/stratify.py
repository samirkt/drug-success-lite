"""Stratify each model's TEST set by seen vs unseen drugs, and re-score it.

Pure evaluation over already-saved predictions (no retraining). For every
`runs/<exp>/predictions.parquet`, mark each test example "seen" if any of its
drugs appeared in that model's training rows, then report ROC-AUC / PR-AUC / F1
(max-F1 threshold per set) for ALL / SEEN / UNSEEN. Driven by `dsm stratify`.

Drug identity:
  - dsm datasets (ours_*)    : candidate_id prefix before "__" (DrugBank id).
  - hint_benchmark (hint_p*) : the row's SMILES (molecule identity; no drug col).
Multi-drug rows are "seen" if ANY constituent drug was in training.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .config import PROJECT_ROOT
from .datasets import materialize
from .evaluate import _bootstrap_metric_ci
from .experiments import DATASETS, EXPERIMENTS

RUNS_DIR = PROJECT_ROOT / "runs"
MIN_STRATUM_N = 20

# native-repro experiments share the canonical benchmark dataset for membership.
_NATIVE_TO_DATASET = {"phase_I": "hint_p1", "phase_II": "hint_p2", "phase_III": "hint_p3"}

# embed_swap target -> canonical dataset key (its predictions live in runs/embed_swap/<target>/).
_EMBED_SWAP_DATASET = {"p1": "hint_p1", "p2": "hint_p2", "p3": "hint_p3", "di": "ours_di"}


# --------------------------------------------------------------------------- #
# Drug identity + seen membership
# --------------------------------------------------------------------------- #
def drug_keys(dataset_df: pd.DataFrame, kind: str) -> pd.Series:
    """Per-row list of drug keys used for seen/unseen membership."""
    if kind == "dsm":
        col = "candidate_id" if "candidate_id" in dataset_df.columns else "example_id"
        return dataset_df[col].astype(str).map(lambda s: [s.split("__")[0]])
    if kind == "hint_benchmark":
        def keys(smiles):
            seen, out = set(), []
            for s in (smiles if smiles is not None else []):
                s = str(s)
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
            return out
        return dataset_df["smiles"].map(keys)
    raise ValueError(f"unknown dataset kind {kind!r}")


def seen_lookup(dataset_df: pd.DataFrame, kind: str) -> dict:
    """example_id -> bool: any of the row's drugs appeared in train+valid rows."""
    keys = drug_keys(dataset_df, kind)
    train_mask = dataset_df["split"].isin(["train", "valid"]).to_numpy()
    train_keys: set = set()
    for ks in keys[train_mask]:
        train_keys.update(ks)
    seen = keys.map(lambda ks: any(k in train_keys for k in ks))
    return dict(zip(dataset_df["example_id"].astype(str), seen))


# --------------------------------------------------------------------------- #
# Metrics (max-F1 threshold per set)
# --------------------------------------------------------------------------- #
def strat_metrics(y, proba, *, bootstrap_ci: int = 0) -> dict:
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba).astype(float)
    n, n_pos = int(len(y)), int(y.sum())
    base = {"n": n, "n_pos": n_pos}
    if n < MIN_STRATUM_N or len(np.unique(y)) < 2:
        return {**base, "roc_auc": float("nan"), "pr_auc": float("nan"),
                "f1": float("nan"), "f1_threshold": float("nan")}

    prec, rec, thr = precision_recall_curve(y, proba)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1s = np.nan_to_num(2 * prec * rec / (prec + rec))
    best = int(np.argmax(f1s))
    f1_threshold = float(thr[best]) if best < len(thr) else 1.0
    out = {
        **base,
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "f1": float(f1s[best]),
        "f1_threshold": f1_threshold,
    }
    if bootstrap_ci > 0:
        # CI for f1 at this set's own max-F1 threshold (matches the point estimate's convention).
        out.update(_bootstrap_metric_ci(y, proba, n_boot=bootstrap_ci, threshold=f1_threshold))
    return out


# --------------------------------------------------------------------------- #
# Per-experiment stratification
# --------------------------------------------------------------------------- #
def resolve_dataset(spec) -> tuple[str, str]:
    """(dataset_name, kind) for an experiment."""
    if spec.dataset:
        return spec.dataset, DATASETS[spec.dataset].kind
    name = _NATIVE_TO_DATASET[spec.native_benchmark]
    return name, DATASETS[name].kind


def _strata(preds: pd.DataFrame, seen_by_id: dict, label: str, bootstrap_ci: int) -> dict:
    """Map seen membership and score ALL / SEEN / UNSEEN from already-saved predictions."""
    preds = preds.copy()
    preds["seen"] = preds["example_id"].astype(str).map(seen_by_id)
    n_unmatched = int(preds["seen"].isna().sum())
    if n_unmatched:
        print(f"  ! {label}: {n_unmatched} predictions had no dataset match (dropped)")
        preds = preds.dropna(subset=["seen"])
    return {
        stratum: strat_metrics(sub["label"], sub["y_proba"], bootstrap_ci=bootstrap_ci)
        for stratum, sub in (("all", preds),
                             ("seen", preds[preds["seen"]]),
                             ("unseen", preds[~preds["seen"]]))
    }


def stratify_experiment(name: str, bootstrap_ci: int = 0) -> dict | None:
    spec = EXPERIMENTS[name]
    preds_path = RUNS_DIR / name / "predictions.parquet"
    if not preds_path.exists():
        return None

    dataset_name, kind = resolve_dataset(spec)
    dataset_df = pd.read_parquet(materialize(DATASETS[dataset_name]))
    seen_by_id = seen_lookup(dataset_df, kind)

    rec = {
        "experiment": name,
        "dataset": dataset_name,
        "drug_identity": "candidate_id_prefix" if kind == "dsm" else "smiles",
        "seen_rule": "any",
        **_strata(pd.read_parquet(preds_path), seen_by_id, name, bootstrap_ci),
    }
    (RUNS_DIR / name / "stratified.json").write_text(json.dumps(rec, indent=2))
    return rec


def stratify_embed_swap(bootstrap_ci: int = 0) -> list[dict]:
    """Stratify the embed_swap models (xgb_pca50/xgb_hint_emb and their mdtp variants), whose
    predictions live in runs/embed_swap/<target>/<model>_preds.parquet rather than as registered
    experiments. Mirrors run.py's _embed_swap_rows so `dsm stratify` covers every model."""
    out: list[dict] = []
    for target, ds_key in _EMBED_SWAP_DATASET.items():
        target_dir = RUNS_DIR / "embed_swap" / target
        pred_files = sorted(target_dir.glob("*_preds.parquet"))
        # Only the embedding-swap xgb models; hint_preds is the registered hint_* experiment.
        pred_files = [p for p in pred_files if p.name.startswith("xgb_")]
        if not pred_files:
            continue
        kind = DATASETS[ds_key].kind
        dataset_df = pd.read_parquet(materialize(DATASETS[ds_key]))
        seen_by_id = seen_lookup(dataset_df, kind)
        for pf in pred_files:
            model = pf.name[: -len("_preds.parquet")]      # e.g. xgb_pca50, xgb_hint_emb_mdtp
            name = f"{model}_{target}"                       # matches _FIRST_CLASS / _embed_swap_rows
            out.append({
                "experiment": name,
                "dataset": ds_key,
                "drug_identity": "candidate_id_prefix" if kind == "dsm" else "smiles",
                "seen_rule": "any",
                **_strata(pd.read_parquet(pf), seen_by_id, name, bootstrap_ci),
            })
    return out


def stratify_all(names: list[str] | None = None, bootstrap_ci: int = 0) -> list[dict]:
    if names is not None:
        return [r for r in (stratify_experiment(n, bootstrap_ci) for n in names) if r is not None]
    # Full sweep: registered experiments + the embed_swap models (so all results show up).
    recs = [r for r in (stratify_experiment(n, bootstrap_ci) for n in EXPERIMENTS) if r is not None]
    return recs + stratify_embed_swap(bootstrap_ci)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(v):
    if isinstance(v, float):
        return "   nan" if v != v else f"{v:.4f}"
    return str(v)


def _fmt_ci(m: dict, key: str) -> str:
    """`0.690 [0.652, 0.731]` when a bootstrap CI is present, else just the point estimate."""
    pt = m.get(key)
    lo, hi = m.get(f"{key}_lo"), m.get(f"{key}_hi")
    if isinstance(lo, float) and lo == lo and isinstance(hi, float) and hi == hi:
        return f"{_fmt(pt)} [{lo:.4f}, {hi:.4f}]"
    return _fmt(pt)


def print_table(records: list[dict]) -> None:
    cols = ["experiment", "stratum", "n", "n_pos", "roc_auc", "pr_auc", "f1", "f1_thr"]
    rows = []
    for r in records:
        for stratum in ("all", "seen", "unseen"):
            m = r[stratum]
            rows.append({
                "experiment": r["experiment"], "stratum": stratum,
                "n": m["n"], "n_pos": m["n_pos"],
                "roc_auc": _fmt_ci(m, "roc_auc"), "pr_auc": _fmt_ci(m, "pr_auc"),
                "f1": _fmt_ci(m, "f1"), "f1_thr": _fmt(m["f1_threshold"]),
            })
    w = {c: max(len(c), *(len(str(row[c])) for row in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    print("  ".join("-" * w[c] for c in cols))
    prev = None
    for row in rows:
        if prev is not None and row["experiment"] != prev:
            print()
        print("  ".join(str(row[c]).ljust(w[c]) for c in cols))
        prev = row["experiment"]

    print("\nseen − unseen ROC-AUC gap (memorization signal):")
    for r in records:
        s, u = r["seen"]["roc_auc"], r["unseen"]["roc_auc"]
        gap = (s - u) if (s == s and u == u) else float("nan")
        print(f"  {r['experiment']:22s} seen={_fmt(s)}  unseen={_fmt(u)}  gap={_fmt(gap)}")


def summary_frame(records: list[dict]) -> pd.DataFrame:
    out = []
    for r in records:
        for stratum in ("all", "seen", "unseen"):
            out.append({"experiment": r["experiment"], "dataset": r["dataset"],
                        "drug_identity": r["drug_identity"], "stratum": stratum, **r[stratum]})
    return pd.DataFrame(out)
