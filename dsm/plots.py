"""Bar-plot renderers for `dsm results` and `dsm stratify`.

One metric on purpose — ROC-AUC, the headline discrimination number — with the
95% bootstrap CIs the tables already compute, drawn as error bars, plus a dashed
chance line at 0.5. That is the whole story and nothing else is plotted:

  - results : ROC-AUC per experiment  (how well each model discriminates).
  - stratify: seen vs unseen ROC-AUC  (the memorization gap).

Headless (Agg) so it runs without a display; each renderer returns the saved
path (or None if there was nothing scorable to draw).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CHANCE = 0.5
_SEEN_C, _UNSEEN_C = "#3b6fb0", "#c0573b"   # seen / unseen bar colors


def _ok(v) -> bool:
    """A real, plottable number (not None, not NaN)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def _err(pt, lo, hi) -> tuple[float, float]:
    """Asymmetric (lower, upper) error-bar lengths for a point estimate, clipped at 0."""
    if _ok(lo) and _ok(hi):
        return max(0.0, pt - lo), max(0.0, hi - pt)
    return 0.0, 0.0


def plot_results(rows: list[dict], out_path: Path, *, title: str = "ROC-AUC by experiment") -> Path | None:
    """Horizontal ROC-AUC bars (one per experiment, table order) with 95% CI whiskers."""
    rows = [r for r in rows if _ok(r.get("roc_auc"))]
    if not rows:
        return None
    labels = [r["experiment"] for r in rows]
    vals = [float(r["roc_auc"]) for r in rows]
    lower, upper = zip(*(_err(v, r.get("roc_auc_lo"), r.get("roc_auc_hi"))
                         for v, r in zip(vals, rows)))

    y = range(len(rows))
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.2))
    ax.barh(list(y), vals, color=_SEEN_C, height=0.7,
            xerr=[list(lower), list(upper)], error_kw={"ecolor": "#333", "elinewidth": 1, "capsize": 3})
    ax.axvline(CHANCE, ls="--", lw=1, color="gray")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()                       # first row on top, matching the table
    ax.set_xlabel("ROC-AUC (95% CI)")
    ax.set_xlim(0.4, 1.0)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_stratify(records: list[dict], out_path: Path,
                  *, title: str = "Seen vs unseen ROC-AUC") -> Path | None:
    """Grouped horizontal bars: seen vs unseen ROC-AUC per experiment with 95% CI whiskers.

    The `all` stratum is omitted on purpose — the seen↔unseen contrast is the story."""
    records = [r for r in records
               if _ok(r.get("seen", {}).get("roc_auc")) or _ok(r.get("unseen", {}).get("roc_auc"))]
    if not records:
        return None
    labels = [r["experiment"] for r in records]
    n = len(records)
    h = 0.38                                # half-gap between the paired bars

    fig, ax = plt.subplots(figsize=(7.5, 0.6 * n + 1.4))
    for offset, stratum, color in ((+h / 2, "seen", _SEEN_C), (-h / 2, "unseen", _UNSEEN_C)):
        ys, xs, lo, hi = [], [], [], []
        for i, r in enumerate(records):
            m = r.get(stratum, {})
            if not _ok(m.get("roc_auc")):
                continue
            ys.append(i + offset)
            xs.append(float(m["roc_auc"]))
            l, u = _err(xs[-1], m.get("roc_auc_lo"), m.get("roc_auc_hi"))
            lo.append(l)
            hi.append(u)
        ax.barh(ys, xs, height=h, color=color, label=stratum,
                xerr=[lo, hi], error_kw={"ecolor": "#333", "elinewidth": 1, "capsize": 2})
    ax.axvline(CHANCE, ls="--", lw=1, color="gray")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC (95% CI)")
    ax.set_xlim(0.4, 1.0)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
