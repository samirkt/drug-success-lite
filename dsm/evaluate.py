"""Metrics for binary classification, plus per-phase slicing for trial runs."""

from __future__ import annotations

import numpy as np


# Trial phases that map to a phase-success transition. Keyed by the string value
# of `trial_phase` as written to `trial_detail.parquet`.
PHASE_TRANSITIONS: dict[str, str] = {
    "Phase 1": "P1->P2",
    "Phase 2": "P2->P3",
    "Phase 3": "P3->approval",
}

# Metrics that get bootstrap confidence intervals.
_CI_METRICS: tuple[str, ...] = ("roc_auc", "pr_auc", "f1")


def _bootstrap_metric_ci(
    y_true,
    y_proba,
    *,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
    threshold: float = 0.5,
) -> dict:
    """Bootstrap-percentile CIs for ROC-AUC, PR-AUC and F1.

    Single-class resamples (AUC undefined) are skipped; if none survive, NaN.
    """
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {m: [] for m in _CI_METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        yp = y_proba[idx]
        y_pred = (yp >= threshold).astype(int)
        samples["roc_auc"].append(float(roc_auc_score(yt, yp)))
        samples["pr_auc"].append(float(average_precision_score(yt, yp)))
        samples["f1"].append(float(f1_score(yt, y_pred)))

    alpha = (1.0 - level) / 2.0
    out: dict = {}
    for m in _CI_METRICS:
        vals = samples[m]
        if vals:
            out[f"{m}_lo"] = float(np.quantile(vals, alpha))
            out[f"{m}_hi"] = float(np.quantile(vals, 1.0 - alpha))
        else:
            out[f"{m}_lo"] = float("nan")
            out[f"{m}_hi"] = float("nan")
    return out


def metrics(
    y_true,
    y_proba,
    *,
    threshold: float = 0.5,
    bootstrap_ci: int = 0,
    ci_seed: int = 0,
) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    multiclass = len(set(y_true.tolist())) > 1

    out = {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if multiclass else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_proba)) if multiclass else float("nan"),
        "f1": float(f1_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, np.clip(y_proba, 1e-7, 1 - 1e-7))) if multiclass else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    if bootstrap_ci > 0:
        out.update(
            _bootstrap_metric_ci(
                y_true, y_proba, n_boot=bootstrap_ci, seed=ci_seed, threshold=threshold,
            )
        )
    return out


def metrics_by_phase(
    y_true,
    y_proba,
    trial_phase,
    *,
    threshold: float = 0.5,
    bootstrap_ci: int = 0,
    ci_seed: int = 0,
) -> dict:
    """Per-phase metrics for trial-level runs.

    Slices `(y_true, y_proba)` by `trial_phase` into the three phase-transition
    cohorts (P1->P2 / P2->P3 / P3->approval) and returns a dict mapping the
    transition label to its full metrics dict. Empty cohorts return `{"n": 0}`.
    """
    phases = np.asarray(trial_phase)
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)
    out: dict = {}
    for phase_value, label in PHASE_TRANSITIONS.items():
        mask = phases == phase_value
        n = int(mask.sum())
        if n == 0:
            out[label] = {"n": 0, "n_pos": 0, "n_neg": 0}
            continue
        out[label] = metrics(
            y_true_arr[mask],
            y_proba_arr[mask],
            threshold=threshold,
            bootstrap_ci=bootstrap_ci,
            ci_seed=ci_seed,
        )
    return out
