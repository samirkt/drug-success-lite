#!/usr/bin/env python3
"""Validate that the web tool's confidence signal tracks real accuracy.

The tool shows a confidence band (Out-of-domain / Borderline / In-domain = Low / Medium / High)
from an applicability-domain `support_score` (weakest link of nearest-neighbour Tanimoto to TRAIN
molecules and #TRAIN programs in the disease's ICD category). This script stratifies the held-out
test set by that confidence and reports, per stratum, the Brier score (and a base-rate baseline +
Brier skill, ECE, and ROC-AUC — because raw Brier alone is confounded by each stratum's class
balance). If the signal is real, higher-confidence strata show higher skill / lower ECE / higher ROC.

The AD score is computed against the TRAIN pool only, so it's non-circular for test rows; predictions
are taken override-free (we bypass predict_one's exact-match), evaluating the model, not leaked labels.

    uv run python scripts/confidence_check.py     # prints the table + writes figures/confidence_check.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dsm.config import PROJECT_ROOT
from dsm.cohort import load_cohort, training_support
from dsm.datasets import materialize
from dsm.experiments import DATASETS
from dsm.serve import load_predictor

BANDS = ["Out-of-domain", "Borderline", "In-domain"]      # low -> high confidence
BAND_LABEL = {"Out-of-domain": "Low", "Borderline": "Medium", "In-domain": "High"}
OUT = PROJECT_ROOT / "figures" / "confidence_check.png"


def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def _metrics(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    base = y.mean()
    brier = float(np.mean((p - y) ** 2))
    baseline = float(base * (1 - base))                 # Brier of predicting the stratum's own rate
    skill = float(1 - brier / baseline) if baseline > 0 else float("nan")
    roc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return {"n": len(y), "pos": float(base), "brier": brier, "baseline": baseline,
            "skill": skill, "ece": _ece(y, p), "roc": roc}


def _ci(y, p, fn, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(fn(yy, pp))
    return (float("nan"), float("nan")) if not vals else (
        float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


def _brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def _skill(y, p):
    b = np.asarray(y, float).mean() * (1 - np.asarray(y, float).mean())
    return float(1 - _brier(y, p) / b) if b > 0 else float("nan")


def _roc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def main() -> None:
    art = load_predictor()
    pipe, clf, cal = art["pipeline"], art["clf"], art.get("calibrator")
    load_cohort()  # build train-only AD references once
    df = pd.read_parquet(materialize(DATASETS["ours_di"]))
    test = df[df["split"] == "test"].reset_index(drop=True)
    y = test["label"].to_numpy(int)

    raw = clf.predict_proba(pipe.transform(test))[:, 1]
    pred = cal.predict(raw) if cal is not None else raw

    # per-row applicability-domain confidence (train-only; ignore exact_match self-match)
    score = np.full(len(test), np.nan)
    band = np.array([None] * len(test), dtype=object)
    for i, r in enumerate(test.itertuples(index=False)):
        sm = r.smiles[0] if len(r.smiles) else ""
        s = training_support(sm, list(r.icd_codes))
        if s.get("band") is not None and s.get("support_score") is not None:
            score[i], band[i] = s["support_score"], s["band"]

    avail = ~np.isnan(score)
    print(f"test n={len(y)}  with confidence={int(avail.sum())}  approval rate={y.mean():.3f}")
    yv, pv, sv, bv = y[avail], pred[avail], score[avail], band[avail]

    # ---- by UI band ----
    print("\n=== Brier by confidence band ===")
    hdr = f"{'band':14s} {'n':>5s} {'pos':>5s} {'brier':>16s} {'baseline':>8s} {'skill':>16s} {'ece':>6s} {'roc':>16s}"
    print(hdr)
    band_rows = []
    for b in BANDS:
        m = bv == b
        if m.sum() < 2 or len(np.unique(yv[m])) < 2:
            print(f"{BAND_LABEL[b]+' ('+b+')':14s} {int(m.sum()):>5d}  (too few / single-class)")
            continue
        mm = _metrics(yv[m], pv[m])
        bl, bh = _ci(yv[m], pv[m], _brier)
        kl, kh = _ci(yv[m], pv[m], _skill)
        rl, rh = _ci(yv[m], pv[m], _roc)
        band_rows.append((b, mm, (rl, rh)))
        print(f"{BAND_LABEL[b]+' ('+b+')':14s} {mm['n']:>5d} {mm['pos']:>5.2f} "
              f"{mm['brier']:.3f} [{bl:.3f},{bh:.3f}] {mm['baseline']:>8.3f} "
              f"{mm['skill']:+.3f} [{kl:+.3f},{kh:+.3f}] {mm['ece']:>6.3f} "
              f"{mm['roc']:.3f} [{rl:.3f},{rh:.3f}]")

    # ---- by support_score decile (finer, equal-n) ----
    print("\n=== by support_score decile (low -> high confidence) ===")
    print(f"{'decile':6s} {'range':>13s} {'n':>5s} {'pos':>5s} {'brier':>6s} {'skill':>7s} {'ece':>6s} {'roc':>6s}")
    qbins = pd.qcut(sv, 10, duplicates="drop")
    dec_rows = []
    for cat in qbins.categories:
        m = np.asarray(qbins == cat)
        if m.sum() < 2 or len(np.unique(yv[m])) < 2:
            continue
        mm = _metrics(yv[m], pv[m])
        mid = float(sv[m].mean())
        dec_rows.append((mid, mm))
        print(f"{len(dec_rows):>6d} [{cat.left:.2f},{cat.right:.2f}] {mm['n']:>5d} {mm['pos']:>5.2f} "
              f"{mm['brier']:>6.3f} {mm['skill']:>+7.3f} {mm['ece']:>6.3f} {mm['roc']:>6.3f}")

    # ---- collapsed: Low (out-of-domain) vs High (borderline + in-domain) ----
    print("\n=== collapsed: Low (out-of-domain) vs High (borderline + in-domain) ===")
    collapsed_rows = []
    for name, m in [("Low\n(out-of-domain)", bv == "Out-of-domain"),
                    ("High\n(borderline+in-domain)", bv != "Out-of-domain")]:
        if m.sum() < 2 or len(np.unique(yv[m])) < 2:
            continue
        mm = _metrics(yv[m], pv[m])
        rl, rh = _ci(yv[m], pv[m], _roc)
        collapsed_rows.append((name, mm, (rl, rh)))
        print(f"  {name.replace(chr(10), ' '):26s} n={mm['n']:>4d} pos={mm['pos']:.2f} "
              f"ROC={mm['roc']:.3f} [{rl:.3f},{rh:.3f}] skill={mm['skill']:+.3f} ece={mm['ece']:.3f}")

    _plot([(BAND_LABEL[b], mm, ci) for b, mm, ci in band_rows], dec_rows, collapsed_rows)


def _roc_bars(ax, rows, title):
    x = np.arange(len(rows))
    roc = [m["roc"] for _, m, _ in rows]
    lo = [m["roc"] - ci[0] for _, m, ci in rows]
    hi = [ci[1] - m["roc"] for _, m, ci in rows]
    ax.bar(x, roc, width=0.6, color="#3c6ea5", yerr=[lo, hi], capsize=4)
    ax.axhline(0.5, color="#c0504d", ls="--", lw=1, label="chance (0.5)")
    for xi, (_, m, _) in zip(x, rows):
        ax.annotate(f"{m['roc']:.2f}\n(n={m['n']})", (xi, m["roc"]), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _, _ in rows])
    ax.set_ylim(0.45, 0.92)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="y", color="#eee", lw=0.8)
    ax.set_axisbelow(True)


def _plot(band_disp, dec_rows, collapsed_rows):
    fig, (axb, axd, axc) = plt.subplots(1, 3, figsize=(14, 4.6))

    _roc_bars(axb, band_disp, "A  ROC by confidence band")
    axb.legend(fontsize=8, frameon=False, loc="lower left")

    xs = [mid for mid, _ in dec_rows]
    axd.plot(xs, [m["roc"] for _, m in dec_rows], "o-", color="#3c6ea5")
    axd.axhline(0.5, color="#c0504d", ls="--", lw=1)
    axd.set_xlim(0, 1)
    axd.set_ylim(0.45, 0.92)
    axd.set_xlabel("support_score (confidence) — decile mean")
    axd.set_ylabel("ROC-AUC")
    axd.set_title("B  ROC vs support score (deciles)", loc="left", fontsize=11, fontweight="bold")
    axd.grid(color="#eee", lw=0.8)
    axd.set_axisbelow(True)

    _roc_bars(axc, collapsed_rows, "C  ROC: Low vs High (collapsed)")

    fig.suptitle("Does the web tool's confidence track ROC-AUC?  (held-out test)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
