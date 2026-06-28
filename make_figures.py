#!/usr/bin/env python3
"""Build paper Figures 1-3 for the drug-indication approval-forecast project.

Numbers are loaded LIVE from the repo so the figures always reflect the latest training — point
estimates + bootstrap 95% CIs from `runs/<exp>/metrics.json`, seen/unseen from `dsm.stratify`, and
cohort stats from the materialized `ours_di` dataset. Only the schematic text (Figure 1A/1B) and the
label→experiment wiring are hard-coded. Re-run after retraining; no manual number edits.

    uv run python make_figures.py     # writes figures/figure{1,2,3}.{png,pdf}
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from dsm.datasets import materialize
from dsm.experiments import DATASETS
from dsm.stratify import stratify_experiment

OUTDIR = "figures"
RUNS = Path("runs")


# ----------------------------------------------------------------------------- #
# Live data loaders — numbers come straight from runs/ + the dataset
# ----------------------------------------------------------------------------- #
def _metrics(name: str):
    """(roc, roc_lo, roc_hi, pr, pr_lo, pr_hi) from runs/<name>/metrics.json (CIs -> point if absent)."""
    o = json.loads((RUNS / name / "metrics.json").read_text())["overall"]
    roc, pr = o["roc_auc"], o["pr_auc"]
    return (roc, o.get("roc_auc_lo", roc), o.get("roc_auc_hi", roc),
            pr, o.get("pr_auc_lo", pr), o.get("pr_auc_hi", pr))


def _seen_unseen(name: str, n_boot: int = 1000):
    r = stratify_experiment(name, bootstrap_ci=n_boot)
    return r["seen"], r["unseen"]


def _cohort() -> dict:
    df = pd.read_parquet(materialize(DATASETS["ours_di"]))
    test = df[df["split"] == "test"]
    return {
        "total": len(df),
        "approved": int(df["label"].sum()),
        "failed": int((df["label"] == 0).sum()),
        "test_pairs": len(test),
        "test_pos": int(test["label"].sum()),
        "unique_drugs": int(df["drugbank_id"].nunique()),
        "unique_indications": int(df["indication"].nunique()),
        "avg_ind_per_drug": float(df.groupby("drugbank_id")["indication"].nunique().mean()),
    }

# ----------------------------------------------------------------------------- #
# DATA — schematic text is fixed; all numbers are loaded from the repo below.
# ----------------------------------------------------------------------------- #
COHORT = _cohort()
POS_RATE = COHORT["test_pos"] / COHORT["test_pairs"]  # PR-AUC random baseline

# Figure 1A — prediction-unit comparison.
F1_HEADERS = ["Prior task", "Prediction unit", "Limitation"]
F1_ROWS = [
    ["LOA analysis", "disease / phase / modality rate", "not individualized"],
    ["Drug-level approval", "drug", "not disease-specific"],
    ["Trial-outcome prediction", "trial",
     "not final approval, often trial-feature dependent"],
]
# 'This work' is a separate callout (it resolves the limitations, so it isn't a 'Limitation' row).
F1_THISWORK = ("drug–indication pair   →   individualized, disease-specific, "
               "approval-level (trial-agnostic)")

# Figure 1B — dataset pipeline (left -> right); features are their own block.
F1_PIPELINE = [
    "Clinical trials (ClinicalTrials.gov / AACT)",
    "Drug–indication pairs + FDA approval label",
    "Filter: require SMILES + ICD-10",
    "Attach features: molecule, disease, target / pathway / ADMET",
    "Temporal split: train ≤ 2019 / test > 2019",
]

# Figure 2 — task-specific model vs transferred baselines.
# Wiring only: (label, task origin, experiment, is_transfer); numbers loaded from runs/.
_F2 = [
    ("ChemAP", "built for drug approval", "chemap_di_2019", True),
    ("HINT adapted", "built for trial outcomes", "hint_di_2019", True),
    ("molecule only", "", "abl_molecule", False),
    ("molecule + disease", "", "abl_md", False),
    ("full model", "", "abl_mdtpa", False),
]
# -> (label, origin, roc, roc_lo, roc_hi, pr, pr_lo, pr_hi, is_transfer)
F2_MODELS = [(lbl, origin, *_metrics(exp), is_t) for lbl, origin, exp, is_t in _F2]

# Figure 3A — feature-set ladder. Wiring only: (label, experiment, kind); ROC+CI from runs/,
# sorted ascending so the "ladder" holds however the numbers move.
_F3 = [
    ("pathway", "abl_pathway", "single"),
    ("target", "abl_target", "single"),
    ("disease", "abl_disease", "single"),
    ("ADMET", "abl_admet", "single"),
    ("molecule", "abl_molecule", "single"),
    ("molecule + disease", "abl_md", "cumulative"),
    ("molecule + disease + target", "abl_mdt", "cumulative"),
    ("full", "abl_mdtpa", "cumulative"),
]
F3_LADDER = sorted(
    [(lbl, (m := _metrics(exp))[0], m[1], m[2], kind) for lbl, exp, kind in _F3],
    key=lambda r: r[1],
)

# Figure 3B — seen vs unseen drugs for the full model (abl_mdtpa), from dsm.stratify.
_seen, _unseen = _seen_unseen("abl_mdtpa")
F3_SEEN_UNSEEN = [
    ("seen drugs", _seen["roc_auc"], _seen["roc_auc_lo"], _seen["roc_auc_hi"], _seen["n"]),
    ("unseen drugs", _unseen["roc_auc"], _unseen["roc_auc_lo"], _unseen["roc_auc_hi"], _unseen["n"]),
]

# colours
C_TRANSFER = "#c0504d"     # transferred baselines (don't transfer well)
C_TASK = "#3c6ea5"         # task-specific models
C_SINGLE = "#8aa0b6"       # single feature group
C_CUM = "#3c6ea5"          # cumulative feature set
C_HEADER = "#33415c"
C_HL = "#dbe7ff"           # highlight ("This work")
C_BOX = "#eef2f7"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "savefig.dpi": 300,
    "figure.dpi": 120,
})


# ----------------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------------- #
def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def _panel_tag(ax, tag: str, title: str) -> None:
    ax.set_title(f"  {title}", loc="left", fontsize=12, fontweight="bold")
    ax.text(-0.01, 1.06, tag, transform=ax.transAxes, fontsize=15, fontweight="bold",
            va="bottom", ha="right")


def _hdot_ci(ax, labels, vals, los, his, colors, *, value_fmt="{:.3f}", xlim=None,
             ref=None, ref_label=None):
    """Horizontal dot plot with asymmetric 95% CI whiskers; first item at top."""
    n = len(labels)
    ys = list(range(n))[::-1]
    if ref is not None and (xlim is None or xlim[0] <= ref <= xlim[1]):
        ax.axvline(ref, color="#999999", lw=1.0, ls="--", zorder=0)
        if ref_label:
            ax.text(ref, -0.5, f"{ref_label} ", color="#888888", fontsize=7.5,
                    va="bottom", ha="right", rotation=90)
    for y, v, lo, hi, c in zip(ys, vals, los, his, colors):
        ax.plot([lo, hi], [y, y], color=c, lw=2.2, solid_capstyle="round", zorder=2)
        ax.plot([v], [y], "o", color=c, ms=8, zorder=3)
        ax.text(hi + (0.006 if xlim is None else (xlim[1] - xlim[0]) * 0.01), y,
                value_fmt.format(v), va="center", ha="left", fontsize=9, color="#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, n - 0.4)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="x", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


# ----------------------------------------------------------------------------- #
# Figure 1 — task & dataset schematic
# ----------------------------------------------------------------------------- #
def figure1() -> None:
    fig = plt.figure(figsize=(12, 13.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.62, 0.9], hspace=0.30)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))
    fig.suptitle("Figure 1 — Defining the drug–indication approval-forecast task",
                 fontsize=15, fontweight="bold", y=0.995)

    # --- A: prediction-unit comparison table ---
    axA.axis("off")
    _panel_tag(axA, "A", "Prediction-unit comparison")
    col_edges = [0.01, 0.25, 0.58, 0.99]
    wrap = [20, 26, 34]
    top, bot = 0.90, 0.40
    nrows = len(F1_ROWS) + 1
    h = (top - bot) / nrows

    def cell(x0, x1, y0, text, *, header=False, bold=False):
        wrapped = textwrap.fill(text, width=wrap[ci])
        axA.text(x0 + 0.012, y0 + h / 2, wrapped, ha="left", va="center",
                 fontsize=9.5, color="white" if header else "#222222",
                 fontweight="bold" if (header or bold) else "normal")

    # header band
    axA.add_patch(Rectangle((col_edges[0], top - h), col_edges[-1] - col_edges[0], h,
                            color=C_HEADER, zorder=1))
    for ci, (x0, x1) in enumerate(zip(col_edges[:-1], col_edges[1:])):
        cell(x0, x1, top - h, F1_HEADERS[ci], header=True)
    # body
    for r, row in enumerate(F1_ROWS):
        y0 = top - (r + 2) * h
        is_this = row[0] == "This work"
        if is_this:
            axA.add_patch(Rectangle((col_edges[0], y0), col_edges[-1] - col_edges[0], h,
                                    color=C_HL, zorder=0))
        for ci, (x0, x1) in enumerate(zip(col_edges[:-1], col_edges[1:])):
            cell(x0, x1, y0, row[ci], bold=(is_this and ci == 0))
    # gridlines
    for i in range(nrows + 1):
        axA.plot([col_edges[0], col_edges[-1]], [top - i * h] * 2, color="#cfd6df", lw=0.8, zorder=2)
    for x in col_edges:
        axA.plot([x, x], [bot, top], color="#cfd6df", lw=0.8, zorder=2)
    # 'This work' as a highlighted callout below the table (the resolution, not a limitation row).
    cy0, cyh = 0.06, 0.24
    axA.add_patch(FancyBboxPatch((col_edges[0], cy0), col_edges[-1] - col_edges[0], cyh,
                                 boxstyle="round,pad=0.004,rounding_size=0.025",
                                 fc=C_HL, ec=C_HEADER, lw=1.3, zorder=1))
    axA.text(col_edges[0] + 0.02, cy0 + cyh / 2, "This work", fontsize=12, fontweight="bold",
             va="center", ha="left", color=C_HEADER, zorder=2)
    axA.text(col_edges[0] + 0.15, cy0 + cyh / 2, F1_THISWORK, fontsize=10,
             va="center", ha="left", color="#222222", zorder=2)
    axA.set_xlim(0, 1)
    axA.set_ylim(0, 1)

    # --- B: dataset construction pipeline ---
    axB.axis("off")
    _panel_tag(axB, "B", "Dataset construction pipeline")
    axB.set_xlim(0, 1)
    axB.set_ylim(0, 1)
    n = len(F1_PIPELINE)
    gap = 0.045
    bw = (1.0 - (n - 1) * gap) / n
    yb, bh = 0.26, 0.48
    for i, stage in enumerate(F1_PIPELINE):
        x0 = i * (bw + gap)
        axB.add_patch(FancyBboxPatch((x0, yb), bw, bh,
                                     boxstyle="round,pad=0.004,rounding_size=0.025",
                                     fc=C_BOX, ec=C_HEADER, lw=1.2))
        axB.text(x0 + bw / 2, yb + bh / 2, textwrap.fill(stage, width=19),
                 ha="center", va="center", fontsize=8.2)
        if i < n - 1:
            axB.annotate("", xy=(x0 + bw + gap, yb + bh / 2), xytext=(x0 + bw, yb + bh / 2),
                         arrowprops=dict(arrowstyle="-|>", color=C_HEADER, lw=1.8,
                                         mutation_scale=18))

    # --- C: final cohort summary cards ---
    axC.axis("off")
    _panel_tag(axC, "C", "Final cohort summary")
    axC.set_xlim(0, 1)
    axC.set_ylim(0, 1)
    cards = [
        (f"{COHORT['total']:,}", "drug–indication pairs"),
        (f"{COHORT['approved']:,} / {COHORT['failed']:,}", "approved / failed"),
        (f"{COHORT['test_pairs']:,}", f"test pairs ({COHORT['test_pos']} positive)"),
        (f"{COHORT['unique_drugs']:,}", "unique drugs"),
        (f"{COHORT['unique_indications']:,}", "unique indications"),
        (f"{COHORT['avg_ind_per_drug']:.2f}", "avg indications / drug"),
    ]
    ncol, gx, gy = 3, 0.015, 0.06
    cw = (1.0 - (ncol - 1) * gx) / ncol
    ch, ytop = 0.40, 0.84
    for i, (val, lab) in enumerate(cards):
        r, c = divmod(i, ncol)
        x0 = c * (cw + gx)
        y0 = ytop - r * (ch + gy) - ch
        axC.add_patch(FancyBboxPatch((x0, y0), cw, ch,
                                     boxstyle="round,pad=0.004,rounding_size=0.03",
                                     fc=C_BOX, ec="#c9d3df", lw=1.0))
        axC.text(x0 + cw / 2, y0 + ch * 0.62, val, ha="center", va="center",
                 fontsize=15, fontweight="bold", color=C_HEADER)
        axC.text(x0 + cw / 2, y0 + ch * 0.22, textwrap.fill(lab, width=22),
                 ha="center", va="center", fontsize=8.4, color="#555555")

    _save(fig, "figure1")


# ----------------------------------------------------------------------------- #
# Figure 2 — main result
# ----------------------------------------------------------------------------- #
def figure2() -> None:
    fig, (axR, axP) = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("Figure 2 — Drug–indication model vs transferred baselines",
                 fontsize=14, fontweight="bold", y=1.02)

    labels = [f"{m[0]}\n({m[1]})" if m[1] else m[0] for m in F2_MODELS]
    colors = [C_TRANSFER if m[8] else C_TASK for m in F2_MODELS]

    _hdot_ci(axR, labels, [m[2] for m in F2_MODELS], [m[3] for m in F2_MODELS],
             [m[4] for m in F2_MODELS], colors, xlim=(0.40, 0.83),
             ref=0.5, ref_label="random")
    axR.set_title("A   ROC-AUC", loc="left")
    axR.set_xlabel("ROC-AUC (95% CI)")

    _hdot_ci(axP, labels, [m[5] for m in F2_MODELS], [m[6] for m in F2_MODELS],
             [m[7] for m in F2_MODELS], colors, xlim=(0.15, 0.60),
             ref=POS_RATE, ref_label=f"base rate ({POS_RATE:.2f})")
    axP.set_title("B   PR-AUC", loc="left")
    axP.set_xlabel("PR-AUC (95% CI)")
    axP.set_yticklabels([])

    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color=C_TRANSFER, lw=0, ms=8,
                      label="transferred baseline"),
               Line2D([0], [0], marker="o", color=C_TASK, lw=0, ms=8,
                      label="our model")]
    axR.legend(handles=handles, loc="upper right", fontsize=8.5, frameon=False)

    fig.tight_layout()
    _save(fig, "figure2")


# ----------------------------------------------------------------------------- #
# Figure 3 — feature signal & generalization
# ----------------------------------------------------------------------------- #
def figure3() -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   gridspec_kw={"width_ratios": [1.6, 1.0]})
    fig.suptitle("Figure 3 — Feature signal and generalization (full model)",
                 fontsize=14, fontweight="bold", y=1.02)

    labels = [r[0] for r in F3_LADDER]
    colors = [C_SINGLE if r[4] == "single" else C_CUM for r in F3_LADDER]
    _hdot_ci(axA, labels, [r[1] for r in F3_LADDER], [r[2] for r in F3_LADDER],
             [r[3] for r in F3_LADDER], colors, xlim=(0.55, 0.80))
    axA.set_title("A   Feature-set ladder", loc="left")
    axA.set_xlabel("ROC-AUC (95% CI)")
    from matplotlib.lines import Line2D
    axA.legend(handles=[Line2D([0], [0], marker="o", color=C_SINGLE, lw=0, ms=8,
                               label="single feature group"),
                        Line2D([0], [0], marker="o", color=C_CUM, lw=0, ms=8,
                               label="cumulative feature set")],
               loc="upper right", fontsize=8.5, frameon=False)

    su_labels = [f"{r[0]}\n(n={r[4]:,})" for r in F3_SEEN_UNSEEN]
    _hdot_ci(axB, su_labels, [r[1] for r in F3_SEEN_UNSEEN], [r[2] for r in F3_SEEN_UNSEEN],
             [r[3] for r in F3_SEEN_UNSEEN], [C_TASK, C_TRANSFER], xlim=(0.55, 0.85))
    axB.set_title("B   Seen vs unseen drugs", loc="left")
    axB.set_xlabel("ROC-AUC (95% CI)")
    drop = F3_SEEN_UNSEEN[0][1] - F3_SEEN_UNSEEN[1][1]
    axB.text(0.98, 0.04, f"Δ = {drop:.3f}\n(drops but does not collapse)",
             transform=axB.transAxes, ha="right", va="bottom", fontsize=8.5, color="#555555")

    fig.tight_layout()
    _save(fig, "figure3")


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    figure1()
    figure2()
    figure3()
    print(f"wrote figure1/2/3 (.png + .pdf) to ./{OUTDIR}/")
