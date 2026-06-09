"""Single-group feature ablation of the headline xgb model, on seen vs unseen drugs.

Trains XGBoost on each feature group ALONE (molecule/disease/admet/target/pathway),
evaluates every variant on all/seen/unseen drugs, and reports each against the full
5-group model (`xgb_di_2019`). A group strong on UNSEEN drugs genuinely generalizes;
a group strong only on SEEN drugs is memorization-driven.

Composes the existing pipeline — `run.run_experiment` (train -> predictions) and
`stratify.stratify_experiment` (seen/unseen + max-F1) — it adds no modeling code.
Driven by `dsm ablation`.
"""

from __future__ import annotations

import pandas as pd

from . import run as run_mod
from . import stratify as strat
from .experiments import ALL_GROUPS

FULL = "xgb_di_2019"                       # 5-group baseline (Δ reference)
FAMILY = [f"abl_{g}" for g in ALL_GROUPS]  # single-group variants
STRATA = ("all", "seen", "unseen")
_METRICS = ("roc_auc", "pr_auc", "f1")


def run_ablation(force: bool = False) -> list[dict]:
    """Ensure the family is trained, stratify each, and tag the variant label."""
    records = []
    for name in [FULL] + FAMILY:
        preds = run_mod.RUNS_DIR / name / "predictions.parquet"
        if force or not preds.exists():
            run_mod.run_experiment(name)
        rec = strat.stratify_experiment(name)
        if rec is None:
            continue
        rec["variant"] = "full" if name == FULL else name.removeprefix("abl_")
        records.append(rec)
    return records


def _full_record(records: list[dict]) -> dict | None:
    return next((r for r in records if r["variant"] == "full"), None)


def summary_frame(records: list[dict]) -> pd.DataFrame:
    """Long table: one row per (variant, stratum) with metrics and Δ-vs-full."""
    full = _full_record(records)
    rows = []
    # full first, then single groups, ranked by unseen ROC (best generalizer top).
    ordered = [r for r in records if r["variant"] == "full"]
    ordered += sorted(
        [r for r in records if r["variant"] != "full"],
        key=lambda r: -(r["unseen"].get("roc_auc") or float("-inf")),
    )
    for r in ordered:
        for stratum in STRATA:
            m, fm = r[stratum], (full[stratum] if full else {})
            row = {"variant": r["variant"], "stratum": stratum,
                   "n": m["n"], "n_pos": m["n_pos"]}
            for k in _METRICS:
                row[k] = m.get(k)
                row[f"d_{k}"] = (None if (m.get(k) is None or fm.get(k) is None)
                                 else m[k] - fm[k])
            rows.append(row)
    return pd.DataFrame(rows)


def _fmt(v, signed=False):
    if v is None or (isinstance(v, float) and v != v):
        return "   nan"
    return f"{v:+.4f}" if signed else f"{v:.4f}"


def print_ablation(df: pd.DataFrame) -> None:
    cols = ["variant", "stratum", "n", "roc_auc", "d_roc_auc",
            "pr_auc", "d_pr_auc", "f1", "d_f1"]
    fmt = []
    for _, r in df.iterrows():
        fmt.append({
            "variant": r["variant"], "stratum": r["stratum"], "n": str(int(r["n"])),
            "roc_auc": _fmt(r["roc_auc"]), "d_roc_auc": _fmt(r["d_roc_auc"], True),
            "pr_auc": _fmt(r["pr_auc"]), "d_pr_auc": _fmt(r["d_pr_auc"], True),
            "f1": _fmt(r["f1"]), "d_f1": _fmt(r["d_f1"], True),
        })
    w = {c: max(len(c), *(len(row[c]) for row in fmt)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    print("  ".join("-" * w[c] for c in cols))
    prev = None
    for row in fmt:
        if prev is not None and row["variant"] != prev:
            print()
        print("  ".join(row[c].ljust(w[c]) for c in cols))
        prev = row["variant"]
