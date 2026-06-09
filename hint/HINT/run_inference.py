"""
Run pre-trained HINT phase models over a CSV of trial-formatted rows,
stratified by phase, and report PR-AUC / F1 / ROC-AUC per phase.

Usage:
    uv run python HINT/run_inference.py --input <path.csv>
    uv run python HINT/run_inference.py --input rows.csv --out-json metrics.json --sample-num 20

The input CSV must include the columns:
    nctid, status, why_stop, label, phase, diseases, icdcodes, drugs, smiless, criteria

`phase` is matched against {Phase 1/I, Phase 2/II, Phase 3/III}, case-insensitive.
`icdcodes` is accepted as a flat list (e.g. "['Z63.72']") and rewrapped into
HINT's list-of-lists format on the fly.

Run from the repo root so the relative paths in HINT/dataloader.py resolve.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from random import choices, seed
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

sys.path.append(".")
from HINT.dataloader import csv_three_feature_2_dataloader  # noqa: E402

PHASE_TO_CKPT = {
    1: "save_model/phase_I.ckpt",
    2: "save_model/phase_II.ckpt",
    3: "save_model/phase_III.ckpt",
}

HINT_COLUMNS = [
    "nctid", "status", "why_stop", "label", "phase",
    "diseases", "icdcodes", "drugs", "smiless", "criteria",
]


def normalize_phase(value) -> Optional[int]:
    s = str(value).strip().lower().replace("phase", "").strip()
    return {"1": 1, "i": 1, "2": 2, "ii": 2, "3": 3, "iii": 3}.get(s)


def to_hint_icdcodes(cell: str) -> str:
    """
    HINT expects icdcodes as a list-of-lists serialized so that the parser in
    HINT/dataloader.py:icdcode_text_2_lst_of_lst can read it. The outer list
    holds one inner-list-string per disease. Two input shapes are accepted:
      - flat list, e.g. "['Z63.72']"        → wrap as "[\"['Z63.72']\"]"
      - already HINT-native list-of-lists,  → pass through unchanged
        e.g. "[\"['Z63.72']\", \"['F10.10']\"]"
    """
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return '["[]"]'
    if s.startswith('["') and s.endswith('"]'):
        return s
    return f'["{s}"]'


def write_phase_csv(df_phase: pd.DataFrame, path: str) -> None:
    out = df_phase.copy()
    for col in HINT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out["icdcodes"] = out["icdcodes"].map(to_hint_icdcodes)
    out = out[HINT_COLUMNS].fillna("")
    out.to_csv(path, index=False)


def _bootstrap_metrics(predict_all, label_all, sample_num: int) -> dict:
    n = len(predict_all)
    aucs, f1s, praucs = [], [], []
    seed(0)
    for _ in range(sample_num):
        idx = choices(range(n), k=n)
        y = [label_all[i] for i in idx]
        p = [predict_all[i] for i in idx]
        if len(set(y)) < 2:
            continue
        aucs.append(roc_auc_score(y, p))
        f1s.append(f1_score(y, [1 if x > 0.5 else 0 for x in p]))
        praucs.append(average_precision_score(y, p))

    def stat(arr):
        return (float(np.mean(arr)), float(np.std(arr))) if arr else (None, None)

    pr_m, pr_s = stat(praucs)
    f1_m, f1_s = stat(f1s)
    auc_m, auc_s = stat(aucs)
    return {
        "n_samples": n,
        "pr_auc_mean": pr_m, "pr_auc_std": pr_s,
        "f1_mean": f1_m, "f1_std": f1_s,
        "roc_auc_mean": auc_m, "roc_auc_std": auc_s,
    }


def evaluate_phase(df_phase: pd.DataFrame, ckpt_path: str,
                   batch_size: int, sample_num: int) -> dict:
    fd, tmp_csv = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        write_phase_csv(df_phase, tmp_csv)
        loader = csv_three_feature_2_dataloader(
            tmp_csv, shuffle=False, batch_size=batch_size,
        )
        device = torch.device("cpu")
        model = torch.load(ckpt_path, weights_only=False, map_location=device)
        model = model.to(device)
        if hasattr(model, "set_device"):
            model.set_device(device)
        model.eval()
        with torch.no_grad():
            _, predict_all, label_all, _ = model.generate_predict(loader)
        return _bootstrap_metrics(predict_all, label_all, sample_num)
    finally:
        os.unlink(tmp_csv)


def run(input_csv: str, sample_num: int = 20, batch_size: int = 32) -> dict:
    df = pd.read_csv(input_csv)
    missing = [c for c in HINT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")

    df["_phase"] = df["phase"].map(normalize_phase)
    summary = {}
    for phase in (1, 2, 3):
        sub = df[df["_phase"] == phase].drop(columns=["_phase"])
        key = f"phase_{phase}"
        if sub.empty:
            print(f"phase {phase}: no rows in input")
            continue
        ckpt = PHASE_TO_CKPT[phase]
        if not os.path.exists(ckpt):
            print(f"phase {phase}: missing checkpoint at {ckpt}")
            continue

        metrics = evaluate_phase(sub, ckpt, batch_size, sample_num)
        metrics["phase"] = phase
        metrics["checkpoint"] = ckpt
        summary[key] = metrics

        print(f"phase {phase} (n={metrics['n_samples']}, ckpt={ckpt}):")
        for label, m, s in [
            ("PR-AUC ", metrics["pr_auc_mean"], metrics["pr_auc_std"]),
            ("F1     ", metrics["f1_mean"], metrics["f1_std"]),
            ("ROC-AUC", metrics["roc_auc_mean"], metrics["roc_auc_std"]),
        ]:
            if m is None:
                print(f"  {label} n/a (single-class bootstrap samples)")
            else:
                print(f"  {label} {m:.4f} ± {s:.4f}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run HINT phase models on a CSV and report metrics.",
    )
    p.add_argument("--input", required=True, help="Path to CSV with HINT-format columns.")
    p.add_argument("--sample-num", type=int, default=20, help="Bootstrap resamples per phase.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-json", default=None, help="Optional path to write metrics as JSON.")
    args = p.parse_args()

    summary = run(args.input, sample_num=args.sample_num, batch_size=args.batch_size)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
