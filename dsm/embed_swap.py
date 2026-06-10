"""Train XGBoost on HINT's learned representation — the representation-vs-classifier swap.

The decisive experiment from HINT_VS_XGBOOST_ANALYSIS.md §3: hold HINT's representation
fixed and swap only the head. Runs on two kinds of target:
  - benchmark phases p1/p2/p3 (hint_p{n}, seen/unseen by SMILES identity),
  - our indication-level dataset di (ours_di, the full P1->approval task, seen/unseen by
    drug identity = candidate_id prefix).

Per target, build four prediction sets and stratify each seen vs unseen:

  hint          HINT itself (100-d -> interaction GCN)        — embed_swap's own HINT run
  xgb_full      XGB on 2215-d ECFP4+MACCS + disease features  — reuses xgb_bench_p{n}/xgb_di_md
  xgb_hint_emb  XGB on HINT's trained 100-d (50 MPNN+50 GRAM) — the core swap
  xgb_pca50     XGB on PCA-50(fingerprint) + PCA-50(disease)  — symmetric bottleneck control

Reading:
  xgb_hint_emb ~= hint        -> the gap is HINT's representation/bottleneck, not its head.
  xgb_hint_emb >> hint        -> the gap is HINT's classifier/optimization (GCN, ~5 epochs).
  xgb_pca50    -> hint level  -> the 50-d bottleneck alone explains most of the cost.

One HINT training per target yields both the `hint` predictions and the dumped embeddings.
Composes existing primitives (MoleculeFP, the dsm disease encoders, build_model, stratify,
run_experiment); no new registered experiments.
Run with `python -m dsm.embed_swap [target] [--force]` (target in {p1,p2,p3,di}; omit = all).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import run as run_mod
from . import stratify as strat
from .config import PROJECT_ROOT
from .datasets import materialize
from .experiments import DATASETS
from .model import build_model
from .models import hint_adapter
from .models.sklearn_adapter import MoleculeFP

RUNS_DIR = PROJECT_ROOT / "runs"
EMBED_DIR = RUNS_DIR / "embed_swap"

# target -> (canonical dataset key, reused full-feature xgb experiment).
# Both xgb baselines and HINT use the same mol+disease inputs as the swap. The seen/unseen
# drug-identity rule is derived from DATASETS[ds_key].kind (smiles for benchmark, candidate_id
# for ours).
TARGETS = {
    "p1": ("hint_p1", "xgb_bench_p1"),
    "p2": ("hint_p2", "xgb_bench_p2"),
    "p3": ("hint_p3", "xgb_bench_p3"),
    "di": ("ours_di", "xgb_di_md"),     # indication-level: full P1 -> approval
}
MODELS = ("hint", "xgb_full", "xgb_hint_emb", "xgb_pca50")
STRATA = ("all", "seen", "unseen")


# --------------------------------------------------------------------------- #
# XGB on an arbitrary feature matrix (mirrors sklearn_adapter.run's core)
# --------------------------------------------------------------------------- #
def _fit_predict_xgb(X_train, y_train, X_test, test_df, *, seed=0, inner_val_size=0.1):
    from sklearn.model_selection import train_test_split

    inner_idx, val_idx = train_test_split(
        np.arange(len(X_train)), test_size=inner_val_size, stratify=y_train, random_state=seed)
    X_inner, X_val = X_train[inner_idx], X_train[val_idx]
    y_inner, y_val = y_train[inner_idx], y_train[val_idx]

    n_pos = int(y_inner.sum())
    spw = (len(y_inner) - n_pos) / n_pos if n_pos else 1.0
    clf = build_model("xgb", scale_pos_weight=spw, random_state=seed)
    clf.fit(X_inner, y_inner, X_val=X_val, y_val=y_val)
    y_proba = clf.predict_proba(X_test)[:, 1]

    return pd.DataFrame({
        "example_id": test_df["example_id"].astype(str).values,
        "label": test_df["label"].to_numpy(dtype=np.int8),
        "phase": test_df["phase"].astype(str).values,
        "y_proba": y_proba.astype(float),
    })


def _xgb_hint_emb(emb_path) -> pd.DataFrame:
    """XGB on HINT's dumped 100-d (50 MPNN + 50 GRAM) vectors."""
    df = pd.read_parquet(emb_path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    train_df = df[df["split"].isin(["train", "valid"])].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    X_train = train_df[emb_cols].to_numpy(dtype=np.float32)
    X_test = test_df[emb_cols].to_numpy(dtype=np.float32)
    return _fit_predict_xgb(X_train, train_df["label"].to_numpy(dtype=int), X_test, test_df)


def _xgb_pca50(canonical_path) -> pd.DataFrame:
    """Symmetric control: XGB on PCA-50(fingerprint) + PCA-50(disease) = 100-d.

    The molecule fingerprint (deterministic) and the disease block (dataset-dispatched by the
    sklearn adapter: ICD multi-hot on the benchmark, DiseaseGroup on our data; vocab fit on
    train) are each reduced to 50-d by a PCA fit on the train slice, mirroring HINT's 50+50
    learned bottleneck."""
    from sklearn.decomposition import PCA

    from .models.sklearn_adapter import _make_encoders, _takes_y

    df = pd.read_parquet(canonical_path)
    train_mask = df["split"].isin(["train", "valid"]).to_numpy()
    test_mask = (df["split"] == "test").to_numpy()
    train_df = df[train_mask].reset_index(drop=True)
    y_train = train_df["label"].to_numpy(dtype=int)

    mol = MoleculeFP().transform(df)                      # (n, 2215), deterministic
    disease_encs = _make_encoders(["disease"], df)        # ICD multi-hot OR DiseaseGroup
    for e in disease_encs:
        e.fit(train_df, y_train) if _takes_y(e) else e.fit(train_df)
    disease = np.hstack([e.transform(df) for e in disease_encs])

    blocks = []
    for mat in (mol, disease):
        pca = PCA(n_components=min(50, mat.shape[1]), random_state=0)
        pca.fit(mat[train_mask])
        blocks.append(pca.transform(mat))
    X = np.hstack(blocks).astype(np.float32)

    test_df = df[test_mask].reset_index(drop=True)
    return _fit_predict_xgb(X[train_mask], y_train, X[test_mask], test_df)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _stratify_preds(model_name: str, target: str, preds: pd.DataFrame, seen_by_id: dict) -> dict:
    p = preds.copy()
    p["seen"] = p["example_id"].astype(str).map(seen_by_id)
    n_unmatched = int(p["seen"].isna().sum())
    if n_unmatched:
        print(f"  ! {target}/{model_name}: {n_unmatched} predictions had no dataset match (dropped)")
        p = p.dropna(subset=["seen"])
    rec = {"model": model_name, "target": target}
    for stratum, sub in (("all", p), ("seen", p[p["seen"]]), ("unseen", p[~p["seen"]])):
        rec[stratum] = strat.strat_metrics(sub["label"], sub["y_proba"])
    return rec


def run_embed_swap(targets: list[str] | None = None, force: bool = False) -> list[dict]:
    targets = targets if targets else list(TARGETS)
    records: list[dict] = []
    for target in targets:
        ds_key, xgb_full_exp = TARGETS[target]
        kind = DATASETS[ds_key].kind          # "hint_benchmark" (SMILES) | "dsm" (candidate_id)
        canonical = materialize(DATASETS[ds_key])
        out_dir = EMBED_DIR / target
        out_dir.mkdir(parents=True, exist_ok=True)
        hint_preds = out_dir / "hint_preds.parquet"
        emb_path = out_dir / "embeddings.parquet"

        # 1. HINT: predictions + dumped 100-d embeddings (one training).
        if force or not emb_path.exists() or not hint_preds.exists():
            hint_adapter.run(dataset_path=canonical, features=["mol", "disease"],
                             out_path=hint_preds, dump_embeddings=emb_path)

        # 2. full-feature xgb baseline (reuse the registered experiment).
        full_preds = run_mod.RUNS_DIR / xgb_full_exp / "predictions.parquet"
        if force or not full_preds.exists():
            run_mod.run_experiment(xgb_full_exp)

        # 3. the two new xgb prediction sets.
        emb_pred_df = _xgb_hint_emb(emb_path)
        emb_pred_df.to_parquet(out_dir / "xgb_hint_emb_preds.parquet", index=False)
        pca_pred_df = _xgb_pca50(canonical)
        pca_pred_df.to_parquet(out_dir / "xgb_pca50_preds.parquet", index=False)

        # 4. stratify all four (seen/unseen by the dataset's drug-identity rule).
        seen_by_id = strat.seen_lookup(pd.read_parquet(canonical), kind)
        preds_by_model = {
            "hint": pd.read_parquet(hint_preds),
            "xgb_full": pd.read_parquet(full_preds),
            "xgb_hint_emb": emb_pred_df,
            "xgb_pca50": pca_pred_df,
        }
        for model_name in MODELS:
            records.append(_stratify_preds(model_name, target, preds_by_model[model_name], seen_by_id))
    return records


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def summary_frame(records: list[dict]) -> pd.DataFrame:
    out = []
    for r in records:
        for stratum in STRATA:
            out.append({"target": r["target"], "model": r["model"], "stratum": stratum, **r[stratum]})
    return pd.DataFrame(out)


def _fmt(v) -> str:
    if isinstance(v, float):
        return "   nan" if v != v else f"{v:.4f}"
    return str(v)


def print_table(records: list[dict]) -> None:
    """Per (target, model): all/seen/unseen ROC-AUC + the seen−unseen gap (memorization signal)."""
    cols = ["target", "model", "all", "seen", "unseen", "gap", "n_unseen"]
    rows = []
    for r in records:
        s, u = r["seen"]["roc_auc"], r["unseen"]["roc_auc"]
        gap = (s - u) if (s == s and u == u) else float("nan")
        rows.append({
            "target": r["target"], "model": r["model"],
            "all": _fmt(r["all"]["roc_auc"]), "seen": _fmt(s), "unseen": _fmt(u),
            "gap": _fmt(gap), "n_unseen": str(r["unseen"]["n"]),
        })
    w = {c: max(len(c), *(len(row[c]) for row in rows)) for c in cols}
    print("ROC-AUC by stratum (the swap: compare xgb_hint_emb vs hint vs xgb_pca50)\n")
    print("  ".join(c.ljust(w[c]) for c in cols))
    print("  ".join("-" * w[c] for c in cols))
    prev = None
    for row in rows:
        if prev is not None and row["target"] != prev:
            print()
        print("  ".join(row[c].ljust(w[c]) for c in cols))
        prev = row["target"]


def main(argv=None) -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ap = argparse.ArgumentParser(prog="python -m dsm.embed_swap", description=__doc__)
    ap.add_argument("target", nargs="?", choices=list(TARGETS),
                    help="benchmark phase p1/p2/p3 or indication-level di; omit for all")
    ap.add_argument("--force", action="store_true", help="retrain even if outputs exist")
    args = ap.parse_args(argv)

    targets = [args.target] if args.target else None
    records = run_embed_swap(targets, force=args.force)
    if not records:
        print("no records produced.")
        return
    print_table(records)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    summary = RUNS_DIR / "embed_swap_summary.csv"
    summary_frame(records).to_csv(summary, index=False)
    print(f"\nwrote {summary}")


if __name__ == "__main__":
    main()
