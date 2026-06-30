"""Generate Figure 1 — drug-indication approval forecast: dataset + model.

A two-act schematic for the paper:
  (a) Dataset construction pipeline — third-party data sources (grey) feed into our
      processing spine, with the Qwen LLM approval-label-extraction step highlighted.
  (b) Approval-forecast model — inputs (molecule + disease) -> featurization -> PCA
      bottleneck -> XGBoost -> Platt calibration -> calibrated per-(drug, indication)
      probability, with one real individualized example.

Pure layout (no data dependency); headline cohort numbers are constants below so the
figure always renders. Writes vector + raster to a *new* filename so the existing
figures/figure1.* are left untouched for comparison.

    uv run python make_figure1.py   ->  figures/figure1_concept.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).parent / "figures"

# ── palette (restrained, clinical) ───────────────────────────────────────────
INK = "#1A1A1A"; MID = "#555555"
NAVY = "#1B3A6B"
GREY_FC = "#EEF1F4"; GREY_EC = "#9AA3AD"
AMBER_FC = "#FBEEDD"; AMBER_EC = "#C9772E"; AMBER_TX = "#8A4B12"
CRIM = "#8B1A1A"
WHITE = "#FFFFFF"

# ── locked cohort headline numbers (stable; wire to runs/ later if desired) ───
PAIRS, DRUGS, INDS, POS, NTEST = "14,134", "2,313", "4,190", "17.5%", "2,637"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def box(ax, cx, cy, w, h, text, *, fc, ec, tc=INK, fs=8.0, fw="normal", lw=1.1):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=fc,
                           edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight=fw, linespacing=1.35, zorder=3)
    return (cx, cy, w, h)


def anc(b, side):
    cx, cy, w, h = b
    return {"L": (cx - w / 2, cy), "R": (cx + w / 2, cy),
            "T": (cx, cy + h / 2), "B": (cx, cy - h / 2)}[side]


def arrow(ax, p0, p1, *, color=MID, lw=1.1, ms=9):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, shrinkA=1, shrinkB=1, zorder=1))


def panel_timeline(ax):
    """Act a — where the prediction sits in drug development: scored at preclinical entry,
    forecasting eventual FDA approval after all trial phases complete (trial-agnostic)."""
    ax.text(0.0, 0.99, "a", fontsize=13, fontweight="bold", color=INK, va="top")
    ax.text(0.032, 0.982, "Where the prediction is made", fontsize=10, fontweight="bold",
            color=INK, va="top")

    ty, th = 0.50, 0.24
    # (center, width, label, kind): preclinical = entry point; phases de-emphasized; approval = label
    stages = [
        (0.075, 0.135, "Discovery /\nPreclinical", "pre"),
        (0.235, 0.115, "Phase 1", "ph"),
        (0.375, 0.115, "Phase 2", "ph"),
        (0.515, 0.115, "Phase 3", "ph"),
        (0.66,  0.115, "FDA\nreview", "ph"),
        (0.85,  0.15,  "Approved", "appr"),
    ]
    for cx, w, label, kind in stages:
        if kind == "appr":
            box(ax, cx, ty, w, th, label, fc=NAVY, ec=NAVY, tc=WHITE, fw="bold", fs=8.4)
        elif kind == "pre":
            box(ax, cx, ty, w, th, label, fc=WHITE, ec=NAVY, fs=7.8)
        else:
            box(ax, cx, ty, w, th, label, fc=GREY_FC, ec=GREY_EC, tc=MID, fs=7.8)
    cs = [s[0] for s in stages]; ws = [s[1] for s in stages]
    for i in range(len(stages) - 1):
        arrow(ax, (cs[i] + ws[i] / 2, ty), (cs[i + 1] - ws[i + 1] / 2, ty), ms=8, lw=1.0)

    # prediction marker above the preclinical entry
    px = 0.075
    ax.annotate("", xy=(px, ty + th / 2 + 0.02), xytext=(px, 0.76),
                arrowprops=dict(arrowstyle="-|>", color=AMBER_EC, lw=1.7))
    ax.text(px, 0.80, "Prediction made here", fontsize=8.2, color=AMBER_TX,
            fontweight="bold", ha="center", va="bottom")

    # forecast-horizon arrow below the track: preclinical -> approval
    fy = 0.20
    arrow(ax, (px, fy), (0.85, fy), color=NAVY, lw=1.4, ms=12)
    ax.text((px + 0.85) / 2, fy + 0.10, "forecasts eventual FDA approval   ·   trial-agnostic",
            fontsize=8.0, color=NAVY, ha="center", va="bottom", style="italic")

    # what is / isn't available at prediction time
    ax.text(px - 0.075, fy - 0.16,
            "known now:  molecule (SMILES) + indication (ICD-10)          "
            "not used:  trial readouts / endpoints",
            fontsize=7.3, color=MID, ha="left", va="center")


def main() -> None:
    fig = plt.figure(figsize=(7.4, 9.2))
    fig.suptitle("Figure 1  ·  Drug–indication approval forecast: task, dataset and model",
                 y=0.99, fontsize=11.5, fontweight="bold", color=INK)

    axT = fig.add_axes([0.02, 0.78, 0.96, 0.17])
    axA = fig.add_axes([0.02, 0.42, 0.96, 0.31])
    axB = fig.add_axes([0.02, 0.04, 0.96, 0.31])
    for ax in (axT, axA, axB):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    panel_timeline(axT)

    # ══ Panel b — dataset construction ═══════════════════════════════════════
    axA.text(0.0, 1.03, "b", fontsize=13, fontweight="bold", color=INK, va="top")
    axA.text(0.032, 1.022, "Dataset construction", fontsize=10, fontweight="bold",
             color=INK, va="top")

    # encoding legend (top-right)
    def leg(x, fc, ec, label):
        axA.add_patch(Rectangle((x, 0.985), 0.022, 0.032, facecolor=fc, edgecolor=ec, lw=1.0))
        axA.text(x + 0.028, 1.001, label, fontsize=7.2, color=MID, va="center")
    leg(0.45, GREY_FC, GREY_EC, "third-party source")
    leg(0.66, WHITE, NAVY, "our step")
    leg(0.80, AMBER_FC, AMBER_EC, "Qwen LLM")

    # processing spine
    sy, sw, sh = 0.45, 0.165, 0.15
    xs = [0.10, 0.295, 0.49, 0.685, 0.88]
    n1 = box(axA, xs[0], sy, sw, sh, "ClinicalTrials.gov\n/ AACT", fc=GREY_FC, ec=GREY_EC, tc=MID, fs=7.8)
    n2 = box(axA, xs[1], sy, sw, sh, "Drug–indication\npairs", fc=WHITE, ec=NAVY)
    n3 = box(axA, xs[2], sy, sw, sh, "Qwen LLM\napproval-label\nextraction", fc=AMBER_FC,
             ec=AMBER_EC, tc=AMBER_TX, fw="bold", fs=7.6, lw=1.4)
    n4 = box(axA, xs[3], sy, sw, sh, "Require\nSMILES + ICD-10", fc=WHITE, ec=NAVY)
    n5 = box(axA, xs[4], sy, sw, sh, "Attach multimodal\nfeatures", fc=WHITE, ec=NAVY)
    for a, b in [(n1, n2), (n2, n3), (n3, n4), (n4, n5)]:
        arrow(axA, anc(a, "R"), anc(b, "L"))

    # third-party source tributaries (feed from above)
    uy, uw, uh = 0.80, 0.175, 0.135
    sfda = box(axA, xs[2], uy, uw, uh, "FDA approvals\n+ trial outcomes", fc=GREY_FC, ec=GREY_EC, tc=MID, fs=7.3)
    sstr = box(axA, xs[3], uy, uw, uh, "ChEMBL · DrugBank\nNLM ICD-10-CM", fc=GREY_FC, ec=GREY_EC, tc=MID, fs=7.3)
    sfea = box(axA, xs[4], uy, uw, uh, "Open Targets · Reactome\nDrugBank ADMET · MeSH", fc=GREY_FC, ec=GREY_EC, tc=MID, fs=6.9)
    for s, n in [(sfda, n3), (sstr, n4), (sfea, n5)]:
        arrow(axA, anc(s, "B"), anc(n, "T"))

    # terminal locked cohort
    ty = 0.13
    term = box(axA, 0.59, ty, 0.80, 0.135,
               f"Locked cohort   ·   {PAIRS} drug–indication pairs   ·   {DRUGS} drugs × {INDS} indications   ·   {POS} approved\n"
               f"temporal split:   train ≤ 2019    /    held-out test > 2019   (n = {NTEST})",
               fc=NAVY, ec=NAVY, tc=WHITE, fs=7.6, fw="bold")
    arrow(axA, anc(n5, "B"), (xs[4], ty + 0.135 / 2))

    # ══ Panel c — model ══════════════════════════════════════════════════════
    axB.text(0.0, 1.05, "c", fontsize=13, fontweight="bold", color=INK, va="top")
    axB.text(0.032, 1.04, "Approval-forecast model", fontsize=10, fontweight="bold",
             color=INK, va="top")

    my = 0.66
    # two inputs (molecule + disease) converging
    i_mol = box(axB, 0.075, 0.78, 0.13, 0.13, "Drug\n(SMILES)", fc=WHITE, ec=NAVY, fs=7.8)
    i_dis = box(axB, 0.075, 0.56, 0.13, 0.13, "Disease\n(ICD-10)", fc=WHITE, ec=NAVY, fs=7.8)
    feat = box(axB, 0.275, my, 0.165, 0.18, "Featurize\nfingerprints +\nICD multi-hot", fc=WHITE, ec=NAVY, fs=7.6)
    pca  = box(axB, 0.475, my, 0.145, 0.18, "PCA-50 / block\n(50 + 50)", fc=WHITE, ec=NAVY, fs=7.8)
    xgb  = box(axB, 0.635, my, 0.105, 0.18, "XGBoost", fc=WHITE, ec=NAVY, fs=8.2)
    cal  = box(axB, 0.79,  my, 0.145, 0.18, "Platt\ncalibration\n(5-fold OOF)", fc=WHITE, ec=NAVY, fs=7.6)
    out  = box(axB, 0.935, my, 0.12, 0.18, "Calibrated\nP(approval)", fc=NAVY, ec=NAVY, tc=WHITE, fw="bold", fs=7.8)
    arrow(axB, anc(i_mol, "R"), anc(feat, "L"))
    arrow(axB, anc(i_dis, "R"), anc(feat, "L"))
    for a, b in [(feat, pca), (pca, xgb), (xgb, cal), (cal, out)]:
        arrow(axB, anc(a, "R"), anc(b, "L"))

    # individualized example (held-out test)
    axB.text(0.0, 0.30, "Individualized output — same molecule, different indications:",
             fontsize=8.2, fontweight="bold", color=INK, va="center")

    def exrow(y, fill, drug, ind, p, ok):
        axB.add_patch(Rectangle((0.0, y - 0.02), 0.024, 0.04,
                                facecolor=(NAVY if fill else WHITE), edgecolor=NAVY, lw=1.2))
        axB.text(0.04, y, drug, fontsize=8.0, color=INK, va="center")
        arrow(axB, (0.175, y), (0.215, y), color=MID, lw=1.0, ms=8)
        axB.text(0.225, y, ind, fontsize=8.0, color=INK, va="center")
        axB.text(0.99, y, p, fontsize=8.0, color=(NAVY if ok else CRIM), va="center", ha="right")
    exrow(0.16, True,  "Cannabidiol", "Lennox–Gastaut syndrome", "p = 0.51   approved", True)
    exrow(0.05, False, "Cannabidiol", "schizophrenia",           "p = 0.25   not approved", False)

    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figure1_concept.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT/'figure1_concept.pdf'} and .png")


if __name__ == "__main__":
    main()
