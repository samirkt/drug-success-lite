#!/usr/bin/env python3
"""Scan one molecule against every known disease in our dataset and rank them by the served model —
i.e. propose the indications the drug is most likely to win approval in. When the molecule is a
dataset drug, overlay its ACTUAL approvals/failures so you can eyeball whether real approvals land
high. Works for any molecule (a SMILES with no known outcomes still gets a ranked proposal list).

Ranking modes (--rank):
  prob  (default) — by the calibrated approval probability. Top proposals are dominated by
                    generally-high-approval diseases (the disease prior).
  lift            — by prob / disease base rate, surfacing molecule-SPECIFIC signal (lift > 1 =
                    the model rates this molecule above the disease's historical average).

Default run scans the top-50 unseen test-set drugs (molecule never trained on), ranked by #approvals,
and draws a compact "fingerprint wall". Small runs (<=12 molecules) get a detailed two-panel view.

Standalone — reuses the served abl_md model + the ours_di dataset; changes nothing else.

    uv run python drug_disease_scan.py                          # top-50 unseen test drugs
    uv run python drug_disease_scan.py --rank lift --n 50
    uv run python drug_disease_scan.py --drug imatinib --smiles "CC(=O)Oc1ccccc1C(=O)O"
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dsm.datasets import materialize
from dsm.experiments import DATASETS
from dsm.serve import load_predictor

OUT = "figures/drug_disease_scan.png"
C_APPR, C_FAIL, C_GRID = "#2e7d32", "#c0504d", "#b9c2cc"
BASE_PRIOR = 10     # pseudocount shrinking each disease's base rate toward global (for lift)
DEFAULT_N = 100     # default # drugs to scan
DETAIL_MAX = 12     # <= this many molecules -> detailed 2-panel; else compact wall


def _key(codes) -> tuple:
    return tuple(sorted({str(c) for c in codes}))


def _score(pred, base, mode):
    """Ranking score: raw calibrated probability, or lift = prob / (smoothed) disease base rate."""
    pred = np.asarray(pred, dtype=float)
    return pred if mode == "prob" else pred / np.asarray(base, dtype=float)


def build_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Distinct diseases keyed by ICD-code set (the model's input space), with a representative
    indication label, historical approval rate, and a smoothed `base` rate (shrunk toward the global
    rate so small-n diseases don't give extreme lift / divide-by-zero)."""
    global_base = float(df["label"].mean())
    g = df.assign(_key=df["icd_codes"].map(_key))
    g = g[g["_key"].map(len) > 0]
    rows = []
    for key, sub in g.groupby("_key"):
        ind = sub["indication"].mode()
        n, appr = len(sub), float(sub["label"].sum())
        rows.append({"key": key, "icd_codes": list(key),
                     "indication": (ind.iloc[0] if len(ind) else key[0]),
                     "hist_rate": appr / n, "hist_n": int(n),
                     "base": (appr + BASE_PRIOR * global_base) / (n + BASE_PRIOR)})
    return pd.DataFrame(rows)


def make_scorer(art: dict, grid: pd.DataFrame):
    """Return f(smiles_list) -> calibrated prob over the whole grid. The disease block is
    molecule-independent, so precompute it ONCE; per molecule only the (tiled) molecule block is
    recomputed. Falls back to full per-row transforms if the model isn't molecule+ICD-disease.
    Bypasses predict_one's exact-match override by construction."""
    pipe, clf, cal = art["pipeline"], art["clf"], art.get("calibrator")
    encs, reds = pipe.encoders, pipe.pcas
    n = len(grid)
    mol_i = next((i for i, e in enumerate(encs) if e.__class__.__name__ == "MoleculeFP"), None)
    others = [i for i in range(len(encs)) if i != mol_i]
    fast = mol_i is not None and all(getattr(encs[i], "column", None) == "icd_codes" for i in others)

    def _cal(raw):
        return cal.predict(raw) if cal is not None else raw

    if not fast:  # safe fallback (e.g. a served model with target/pathway groups)
        return lambda smiles: _cal(clf.predict_proba(
            pipe.transform(pd.DataFrame({"smiles": [smiles] * n,
                                         "icd_codes": list(grid["icd_codes"])})))[:, 1])

    gdf = pd.DataFrame({"icd_codes": list(grid["icd_codes"])})
    dblocks = {i: (reds[i].transform(encs[i].transform(gdf)) if reds is not None
                   else encs[i].transform(gdf)) for i in others}

    def score(smiles):
        mb = encs[mol_i].transform(pd.DataFrame({"smiles": [smiles]}))
        mb = reds[mol_i].transform(mb) if reds is not None else mb
        parts = [None] * len(encs)
        parts[mol_i] = np.repeat(mb, n, axis=0)
        for i in others:
            parts[i] = dblocks[i]
        return _cal(clf.predict_proba(np.hstack(parts).astype(np.float32))[:, 1])

    return score


def select_default(df: pd.DataFrame, n: int) -> list[str]:
    """Unseen test-set drugs (molecule never in train), balanced between failure-rich and
    approval-rich so both outcomes are well represented. Interleaves the two rankings (failures
    first) and dedupes up to n."""
    train_drugs = set(df[df["split"].isin(["train", "valid"])]["drugbank_id"])
    test = df[(df["split"] == "test") & (~df["drugbank_id"].isin(train_drugs))]
    by_fail = list(test[test["label"] == 0].groupby("drugbank_id").size().sort_values(ascending=False).index)
    by_appr = list(test[test["label"] == 1].groupby("drugbank_id").size().sort_values(ascending=False).index)
    out, seen = [], set()
    f_it, a_it = iter(by_fail), iter(by_appr)
    while len(out) < n:
        progressed = False
        for it in (f_it, a_it):  # one failure-rich, then one approval-rich, alternating
            for d in it:
                if d not in seen:
                    seen.add(d)
                    out.append(d)
                    progressed = True
                    break
        if not progressed:
            break
    return out[:n]


def molecule_from_drug(df: pd.DataFrame, query: str, train_drugs: set):
    """Resolve a drug name / drugbank_id to (display name, smiles list, unseen?, actuals df)."""
    q = query.strip().lower()
    sub = df[df["drug_name"].astype(str).str.lower() == q]
    if sub.empty:
        sub = df[df["drugbank_id"].astype(str).str.lower() == q]
    if sub.empty:
        sub = df[df["drug_name"].astype(str).str.lower().str.contains(q, regex=False)]
    if sub.empty:
        print(f"  ! '{query}' not found in dataset — skipping")
        return None
    smiles = list(sub.iloc[0]["smiles"])
    name = str(sub.iloc[0]["drug_name"]) or query
    unseen = sub.iloc[0]["drugbank_id"] not in train_drugs
    actuals = pd.DataFrame({
        "indication": sub["indication"].astype(str).values,
        "key": sub["icd_codes"].map(_key).values,
        "label": sub["label"].astype(int).values,
    }).drop_duplicates("key")
    return name, smiles, unseen, actuals


def report(name, grid, pred, score, mode, actuals, top: int) -> None:
    lbl = "prob" if mode == "prob" else "lift"
    sfmt = (lambda v: f"{v:6.3f}") if mode == "prob" else (lambda v: f"{v:6.2f}")
    order = np.argsort(-score)
    print(f"\n=== {name} — top {top} proposed diseases (by {lbl}) ===")
    for i in order[:top]:
        r = grid.iloc[i]
        print(f"  {r['indication'][:40]:40s} {lbl}={sfmt(score[i])} prob={pred[i]:.3f} base={r['base']:.2f}")
    if actuals is None or actuals.empty:
        return
    k2s, k2p = dict(zip(grid["key"], score)), dict(zip(grid["key"], pred))
    for _, a in actuals.iterrows():
        s = k2s.get(a["key"])
        if s is None:
            continue
        pct = float((score <= s).mean())
        print(f"  actual: {a['indication'][:34]:34s} {lbl}={sfmt(s)} prob={k2p[a['key']]:.3f} "
              f"pctile={pct*100:3.0f}%  {'APPROVED' if a['label'] == 1 else 'failed'}")


def _aggregate(mols, grid):
    """Across all molecules with actuals: percentile lists for approvals and failures."""
    appr, fail = [], []
    for m in mols:
        if m["actuals"] is None or m["actuals"].empty:
            continue
        k2s = dict(zip(grid["key"], m["score"]))
        for _, a in m["actuals"].iterrows():
            s = k2s.get(a["key"])
            if s is None:
                continue
            pct = float((m["score"] <= s).mean())
            (appr if a["label"] == 1 else fail).append(pct)
    return np.array(appr), np.array(fail)


def _xlabel(mode):
    return "predicted approval probability" if mode == "prob" else "lift over disease base rate (×)"


def plot_detailed(mols, grid, mode, out, top=3):
    """Two panels per molecule (strip + ranked curve) — for small runs."""
    n = len(mols)
    xmax = max(0.05, max(m["score"].max() for m in mols) * 1.05)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.7 * n), squeeze=False,
                             gridspec_kw={"width_ratios": [1.1, 1.0]})
    rng = np.random.default_rng(0)
    for r, m in enumerate(mols):
        score, actuals = m["score"], m["actuals"]
        k2s = dict(zip(grid["key"], score))
        ax, axc = axes[r, 0], axes[r, 1]
        ax.scatter(score, rng.uniform(-0.35, 0.35, len(score)), s=6, color=C_GRID, alpha=0.30,
                   linewidths=0, zorder=1)
        ax.axvline(np.median(score), color="#999", lw=1, ls="--", zorder=2)
        if mode == "lift":
            ax.axvline(1.0, color="#c98", lw=1, zorder=2)
        if actuals is not None and not actuals.empty:
            for _, a in actuals.iterrows():
                s = k2s.get(a["key"])
                if s is None:
                    continue
                y, c, mk = (0.6, C_APPR, "o") if a["label"] == 1 else (-0.6, C_FAIL, "X")
                ax.scatter(s, y, s=55, color=c, marker=mk, zorder=3,
                           edgecolors="white" if a["label"] == 1 else "none")
        ax.set_xlim(0, xmax)
        ax.set_ylim(-1, 1)
        ax.set_yticks([0.6, 0, -0.6])
        ax.set_yticklabels(["approved", "all diseases", "failed"], fontsize=8)
        ax.set_title(m["title"], fontsize=10, fontweight="bold", loc="left")
        if r == n - 1:
            ax.set_xlabel(_xlabel(mode))
        ax.grid(axis="x", color="#eee", lw=0.8)
        ax.set_axisbelow(True)

        order = np.argsort(-score)
        ys = score[order]
        xs = np.arange(len(ys)) / max(1, len(ys) - 1)
        axc.plot(xs, ys, color="#3c6ea5", lw=1.8)
        if actuals is not None and not actuals.empty:
            for _, a in actuals.iterrows():
                s = k2s.get(a["key"])
                if s is None:
                    continue
                axc.scatter(float((score > s).mean()), s, s=50, zorder=3,
                            color=C_APPR if a["label"] == 1 else C_FAIL,
                            marker="o" if a["label"] == 1 else "X",
                            edgecolors="white" if a["label"] == 1 else "none")
        for i in order[:top]:
            axc.annotate(grid.iloc[i]["indication"][:26], (0.0, score[i]), fontsize=7,
                         xytext=(4, 0), textcoords="offset points", va="center", color="#555")
        axc.set_xlim(-0.01, 1)
        axc.set_ylim(0, xmax)
        axc.set_title("ranked diseases (left = top proposal)", fontsize=9, loc="left")
        if r == n - 1:
            axc.set_xlabel("disease rank (fraction)")
        axc.grid(color="#eee", lw=0.8)
        axc.set_axisbelow(True)
    _finish(fig, mode, out)


def plot_wall(mols, grid, mode, out):
    """Compact fingerprint wall — one thin row per molecule, for large runs. Each row: grey bar =
    the 5th-95th pct of the molecule's disease-score landscape, '|' = its median, green ●/red ✕ =
    real approvals/failures."""
    appr_pct, fail_pct = _aggregate(mols, grid)
    n = len(mols)
    xmax = max(0.05, max(m["score"].max() for m in mols) * 1.05)
    fig = plt.figure(figsize=(9.5, 0.32 * n + 2.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, max(3, 0.32 * n)], hspace=0.12)
    axa, ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # aggregate: where real approvals vs failures fall in the score ranking, across all molecules
    bins = np.linspace(0, 1, 21)
    if len(appr_pct):
        axa.hist(appr_pct, bins=bins, color=C_APPR, alpha=0.6, label=f"approvals (n={len(appr_pct)})")
    if len(fail_pct):
        axa.hist(fail_pct, bins=bins, color=C_FAIL, alpha=0.6, label=f"failures (n={len(fail_pct)})")
    axa.set_xlim(0, 1)
    axa.set_xlabel("percentile of real outcome within the drug's disease ranking (1.0 = top)",
                   fontsize=8)
    axa.set_ylabel("# outcomes", fontsize=8)
    axa.legend(fontsize=8, frameon=False)
    axa.set_title("Aggregate: do real approvals rank above failures?", fontsize=10,
                  fontweight="bold", loc="left")
    axa.tick_params(labelsize=8)

    for r, m in enumerate(mols):
        y = n - 1 - r
        score = m["score"]
        lo, med, hi = np.percentile(score, [5, 50, 95])
        ax.plot([lo, hi], [y, y], color=C_GRID, lw=4, alpha=0.55, solid_capstyle="butt", zorder=1)
        ax.plot(med, y, marker="|", color="#888", ms=7, zorder=2)
        if m["actuals"] is not None and not m["actuals"].empty:
            k2s = dict(zip(grid["key"], score))
            for _, a in m["actuals"].iterrows():
                s = k2s.get(a["key"])
                if s is None:
                    continue
                ax.scatter(s, y, s=30, zorder=3, linewidths=0.5,
                           color=C_APPR if a["label"] == 1 else C_FAIL,
                           marker="o" if a["label"] == 1 else "X",
                           edgecolors="white" if a["label"] == 1 else "none")
    if mode == "lift":
        ax.axvline(1.0, color="#c98", lw=1, zorder=0)
    ax.set_yticks(range(n))
    ax.set_yticklabels([m["name"][:26] for m in mols][::-1], fontsize=6.5)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(_xlabel(mode))
    ax.grid(axis="x", color="#eee", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"Per-drug fingerprint — grey = disease landscape (5–95%), tick = median  "
                 f"({n} drugs)", fontsize=10, fontweight="bold", loc="left")
    _finish(fig, mode, out, tight=False)


def _finish(fig, mode, out, tight=True):
    if tight:
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.suptitle(f"Drug → disease proposals vs real outcomes  (ranked by {mode})",
                     fontsize=13, fontweight="bold", y=0.995)
    fig.legend(handles=[Line2D([0], [0], marker="o", color=C_APPR, lw=0, label="real approval"),
                        Line2D([0], [0], marker="X", color=C_FAIL, lw=0, label="real failure"),
                        Line2D([0], [0], marker="o", color=C_GRID, lw=0, label="untested disease")],
               loc="upper center", bbox_to_anchor=(0.5, 0.965 if tight else 1.0), ncol=3,
               frameon=False, fontsize=9)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drug", action="append", default=[], help="drug name or drugbank_id (repeatable)")
    ap.add_argument("--smiles", action="append", default=[], help="raw SMILES, proposals only (repeatable)")
    ap.add_argument("--rank", choices=["prob", "lift"], default="prob", help="ranking score")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="# default drugs when none given")
    ap.add_argument("--top", type=int, default=8, help="# proposed diseases to print per drug")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    art = load_predictor()
    df = pd.read_parquet(materialize(DATASETS["ours_di"]))
    grid = build_grid(df)
    base = grid["base"].to_numpy()
    score_smiles = make_scorer(art, grid)
    train_drugs = set(df[df["split"].isin(["train", "valid"])]["drugbank_id"])
    print(f"model_roc={art.get('metrics', {}).get('roc_auc'):.3f}  grid={len(grid)} diseases  rank={args.rank}")

    queries = args.drug or ([] if args.smiles else select_default(df, args.n))
    mols = []
    for q in queries:
        res = molecule_from_drug(df, q, train_drugs)
        if res is None:
            continue
        name, smiles, unseen, actuals = res
        a, f = int((actuals["label"] == 1).sum()), int((actuals["label"] == 0).sum())
        mols.append({"name": name, "smiles": smiles, "actuals": actuals,
                     "title": f"{name}  ({'unseen' if unseen else 'seen'}; {a} approved / {f} failed)"})
    for s in args.smiles:
        mols.append({"name": s[:26], "smiles": [s], "actuals": None, "title": f"SMILES: {s[:40]}"})

    if not mols:
        raise SystemExit("no molecules to scan")
    for m in mols:
        m["pred"] = score_smiles(m["smiles"])
        m["score"] = _score(m["pred"], base, args.rank)
        report(m["name"], grid, m["pred"], m["score"], args.rank, m["actuals"], args.top)

    appr_pct, fail_pct = _aggregate(mols, grid)
    if len(appr_pct):
        msg = f"\nAGGREGATE median percentile — approvals: {np.median(appr_pct)*100:.0f}% (n={len(appr_pct)})"
        if len(fail_pct):
            msg += f"  | failures: {np.median(fail_pct)*100:.0f}% (n={len(fail_pct)})"
        print(msg)

    (plot_detailed if len(mols) <= DETAIL_MAX else plot_wall)(mols, grid, args.rank, args.out)


if __name__ == "__main__":
    main()
