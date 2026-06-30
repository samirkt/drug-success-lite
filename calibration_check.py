#!/usr/bin/env python3
"""Validate that the served model's predicted probabilities are accurate.

Loads the saved abl_md model (the exact one the web tool serves), scores the held-out TEST split,
and writes figures/calibration.png with two stacked panels:

  TOP  — reliability diagram. x = probability the model predicted; y = fraction of those programs
         that ACTUALLY got approved. The dashed diagonal is perfect (predicted == observed).
         A point ABOVE the line means the model predicted too LOW; BELOW means too HIGH. Two
         curves: the raw model score and the calibrated (served) probability — closer to the
         diagonal is better. Brier (lower=better) is in the legend.
  BOTTOM — histogram: how many test programs fall at each predicted probability (so you can see
         where the predictions actually sit; most cluster low).

    uv run python calibration_check.py
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dsm.datasets import materialize
from dsm.experiments import DATASETS

ARTIFACT = "runs/abl_md/model.joblib"
N_BINS = 10            # equal-frequency (quantile) bins
OUT = "figures/calibration.png"


def reliability(y, p, n_bins=N_BINS):
    """Equal-frequency reliability table + ECE/MCE. Returns (DataFrame, ece, mce)."""
    df = pd.DataFrame({"y": y, "p": p})
    # quantile bins; dropna handles ties collapsing bin edges
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"),
                                             n=("y", "size"))
    w = g["n"] / g["n"].sum()
    gap = (g["pred"] - g["obs"]).abs()
    ece = float((w * gap).sum())          # expected calibration error (weighted mean gap)
    mce = float(gap.max())                # maximum calibration error (worst bin)
    return g.reset_index(drop=True), ece, mce


def main() -> None:
    from sklearn.metrics import brier_score_loss

    art = joblib.load(ARTIFACT)
    pipe, clf, cal = art["pipeline"], art["clf"], art.get("calibrator")
    print(f"model: {ARTIFACT}  calibration={art.get('calibration')}  base_rate={art.get('base_rate'):.3f}")

    df = pd.read_parquet(materialize(DATASETS["ours_di"]))
    test = df[df["split"] == "test"].reset_index(drop=True)
    y = test["label"].to_numpy(int)
    raw = clf.predict_proba(pipe.transform(test))[:, 1]
    cal_p = cal.predict(raw) if cal is not None else raw
    base = y.mean()  # realized test-era approval rate

    tbl_raw, ece_raw, mce_raw = reliability(y, raw)
    tbl_cal, ece_cal, mce_cal = reliability(y, cal_p)
    brier_raw, brier_cal = brier_score_loss(y, raw), brier_score_loss(y, cal_p)
    for name, tbl, ece, mce, brier in [("raw model score", tbl_raw, ece_raw, mce_raw, brier_raw),
                                       ("calibrated probability", tbl_cal, ece_cal, mce_cal, brier_cal)]:
        print(f"\n=== {name} ===  Brier={brier:.4f}  ECE={ece:.4f}  MCE={mce:.4f}")
        print(tbl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    xmax = max(0.65, cal_p.max() * 1.05, raw.max() * 1.05)
    fig = plt.figure(figsize=(7.5, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.07)
    ax, axh = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # TOP: reliability diagram (raw vs calibrated)
    ax.plot([0, xmax], [0, xmax], ls="--", color="#888", lw=1.2, label="perfect calibration")
    ax.plot(tbl_raw["pred"], tbl_raw["obs"], "o-", color="#c4c4c4", lw=1.5, ms=5,
            label=f"raw model score  (Brier {brier_raw:.3f})")
    ax.plot(tbl_cal["pred"], tbl_cal["obs"], "o-", color="#3c6ea5", lw=2.3, ms=7,
            label=f"calibrated — served  (Brier {brier_cal:.3f})")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, xmax)
    ax.set_ylabel("Observed approval rate\n(fraction that actually got approved)")
    ax.tick_params(labelbottom=False)
    ax.grid(color="#eee", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.text(0.985, 0.04, "above the line → predicted too LOW\nbelow the line → predicted too HIGH",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, style="italic",
            color="#777")
    ax.set_title("Are the served model's probabilities accurate?\n"
                 f"held-out test set: n={len(y)}, {int(y.sum())} approved (base rate {base:.0%})",
                 fontsize=12, fontweight="bold")

    # BOTTOM: where the predictions actually sit
    axh.hist(cal_p, bins=np.linspace(0, xmax, 26), color="#9db4cc", edgecolor="white", lw=0.4)
    axh.set_xlim(0, xmax)
    axh.set_xlabel("Predicted approval probability (calibrated)")
    axh.set_ylabel("# programs")
    axh.grid(axis="y", color="#eee", lw=0.8)
    axh.set_axisbelow(True)

    os.makedirs("figures", exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
