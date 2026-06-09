"""
Run pre-trained HINT phase models over a user-supplied HINT-format CSV.

This scores the input CSV directly, split by phase, using the phase-specific
checkpoints and the CSV's own labels. It reports only the base per-phase
metrics for the user dataset, plus diagnostics about overlap with the builtin
HINT benchmark.

Run from the repo root so relative paths (save_model/, data/) resolve.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.append(".")
sys.path.append("HINT")
from HINT.dataloader import csv_three_feature_2_dataloader
from run_inference import write_phase_csv

USER_CSV = "../../drug-success-report/outputs/features/hint_dataset.csv"
USER_CSV_OUT = "../../drug-success-report/outputs/features/hint_results.csv"
HINT_SPLITS = ["train", "valid", "test"]

PHASE_TO_CKPT = {
    "I": "save_model/phase_I.ckpt",
    "II": "save_model/phase_II.ckpt",
    "III": "save_model/phase_III.ckpt",
}


def norm_phase(v):
    s = str(v).strip().lower().replace("phase", "").strip()
    return {"1": "I", "i": "I", "2": "II", "ii": "II", "3": "III", "iii": "III"}.get(s)


def load_user_dataset(dataset):
    if isinstance(dataset, pd.DataFrame):
        return dataset.copy(), "DataFrame"
    if isinstance(dataset, (str, os.PathLike, Path)):
        dataset_path = os.fspath(dataset)
        return pd.read_csv(dataset_path), dataset_path
    raise TypeError("dataset must be a pandas DataFrame or a CSV path")


def load_hint_split(split):
    hint = pd.concat(
        [
            pd.read_csv(f"data/phase_{phase}_{split}.csv").assign(_split=f"{phase}_{split}")
            for phase in PHASE_TO_CKPT
        ],
        ignore_index=True,
    )
    hint["_phase"] = hint["phase"].map(norm_phase)
    return hint


def filter_user_to_hint_test_shared(user):
    hint_test = load_hint_split("test")
    shared_nctids = set(user["nctid"]) & set(hint_test["nctid"])

    df1, df2 = user[user["nctid"].isin(shared_nctids)].copy(), hint_test[hint_test["nctid"].isin(shared_nctids)].copy()

    return (
        df1,
        df2,
    )


def score_features(feature_df, ckpt):
    fd, tmp = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        write_phase_csv(feature_df, tmp)
        loader = csv_three_feature_2_dataloader(tmp, shuffle=False, batch_size=32)
        model = torch.load(ckpt, weights_only=False, map_location="cpu").to("cpu")
        if hasattr(model, "set_device"):
            model.set_device(torch.device("cpu"))
        model.eval()
        with torch.no_grad():
            _, preds, labels, _ = model.generate_predict(loader)
    finally:
        os.unlink(tmp)
    return np.asarray(preds, dtype=float), np.asarray(labels, dtype=int)


def point_metrics(y, p, thr=0.5):
    yhat = (p > thr).astype(int)
    return {
        "n": len(y),
        "pos_rate": float(y.mean()),
        "ROC-AUC": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
        "PR-AUC": average_precision_score(y, p) if len(set(y)) > 1 else float("nan"),
        "F1": f1_score(y, yhat),
        "Precision": precision_score(y, yhat, zero_division=0),
        "Recall":    recall_score(y, yhat, zero_division=0),
        "Accuracy": accuracy_score(y, yhat),
    }


def print_diagnostics(user):
    hint = pd.concat(
        [
            load_hint_split(split)
            for split in HINT_SPLITS
        ],
        ignore_index=True,
    )

    print()
    print(f"diagnostics: user rows={len(user)} hint rows={len(hint)}")
    print(f"user phase counts: {user['_phase'].value_counts(dropna=False).to_dict()}")
    print(f"hint phase counts: {hint['_phase'].value_counts(dropna=False).to_dict()}")

    user_indexed = user.set_index("nctid")
    hint_indexed = hint.drop_duplicates("nctid").set_index("nctid")
    shared_nctids = sorted(set(user_indexed.index) & set(hint_indexed.index))
    print(f"shared nctids: {len(shared_nctids)}")
    if not shared_nctids:
        return

    shared_user = user_indexed.loc[shared_nctids].reset_index()
    shared_hint = hint_indexed.loc[shared_nctids].reset_index()
    disease_match_mask = shared_user["diseases"].eq(shared_hint["diseases"])
    disease_matched_nctids = shared_user.loc[disease_match_mask, "nctid"].tolist()
    print(f"disease-matched shared rows: {len(disease_matched_nctids)}")

    supported_shared = [
        nctid for nctid in shared_nctids if hint_indexed.loc[nctid, "_phase"] in PHASE_TO_CKPT
    ]
    supported_matched = [
        nctid for nctid in disease_matched_nctids if hint_indexed.loc[nctid, "_phase"] in PHASE_TO_CKPT
    ]
    print(f"shared rows in supported phases: {len(supported_shared)}")
    print(f"disease-matched rows in supported phases: {len(supported_matched)}")

    user_labels = user_indexed.loc[supported_shared, "label"].astype(int)
    hint_labels = hint_indexed.loc[supported_shared, "label"].astype(int)
    disagreements = int((user_labels != hint_labels).sum())
    agreement = float((user_labels == hint_labels).mean()) if len(user_labels) else float("nan")
    print(f"label agreement on shared supported rows: {agreement:.3f} ({disagreements} disagreements)")

    for phase in PHASE_TO_CKPT:
        phase_user_total = int((user["_phase"] == phase).sum())
        phase_shared = [nctid for nctid in supported_shared if hint_indexed.loc[nctid, "_phase"] == phase]
        phase_matched = [nctid for nctid in supported_matched if hint_indexed.loc[nctid, "_phase"] == phase]
        print(
            f"phase {phase}: user_total={phase_user_total} "
            f"shared={len(phase_shared)} disease_matched={len(phase_matched)}"
        )
        if phase_shared:
            phase_user_labels = user_indexed.loc[phase_shared, "label"].astype(int)
            phase_hint_labels = hint_indexed.loc[phase_shared, "label"].astype(int)
            phase_agreement = float((phase_user_labels == phase_hint_labels).mean())
            print(f"phase {phase}: label agreement={phase_agreement:.3f}")


def run(
    dataset=USER_CSV,
    output=USER_CSV_OUT,
    show_diagnostics=True,
    test_only_shared=False,
    verbose=True,
):
    user, dataset_name = load_user_dataset(dataset)
    user["_phase"] = user["phase"].map(norm_phase)
    if verbose:
        print(f"loaded {len(user)} rows from {dataset_name}")
    sources = [("user", user)]
    if test_only_shared:
        user, hint_test_shared = filter_user_to_hint_test_shared(user)
        user.to_csv("filtered_user.csv", index=False)
        hint_test_shared.to_csv("filtered_hint_test.csv", index=False)
        sources = [("user", user), ("hint_test", hint_test_shared)]
        if verbose:
            print(
                f"filtered to {len(user)} rows shared with builtin HINT test split "
                f"({len(hint_test_shared)} matching builtin test rows)"
            )
    if show_diagnostics:
        print_diagnostics(user)

    rows = []
    for source, df in sources:
        for phase, ckpt in PHASE_TO_CKPT.items():
            sub = df[df["_phase"] == phase].drop(columns=["_phase"])
            if sub.empty:
                if verbose:
                    print(f"{source} phase {phase}: 0 rows, skipping")
                continue
            if verbose:
                print(f"{source} phase {phase}: scoring {len(sub)} rows with {ckpt}")
            preds, labels = score_features(sub, ckpt)
            rows.append({"source": source, "phase": phase, **point_metrics(labels, preds)})

    metrics_df = pd.DataFrame(rows).set_index(["source", "phase"])

    # Save results to CSV
    metrics_df.to_csv(output)

    if verbose:
        print(metrics_df.round(3))
    return metrics_df


def main():
    parser = argparse.ArgumentParser(description="Run HINT inference on a CSV or HINT-format dataset.")
    parser.add_argument(
        "--input",
        default=USER_CSV,
        help="Path to a HINT-format CSV. Defaults to the current USER_CSV constant.",
    )
    parser.add_argument(
        "--output",
        default=USER_CSV_OUT,
        help="Path to a HINT-format CSV. Defaults to the current USER_CSV constant.",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Skip overlap and label-agreement diagnostics against the builtin HINT benchmark.",
    )
    parser.add_argument(
        "--shared-with-hint-test-only",
        action="store_true",
        help="Only score input rows whose nctid appears in the builtin HINT test split.",
    )
    args = parser.parse_args()
    run(
        dataset=args.input,
        output=args.output,
        show_diagnostics=not args.no_diagnostics,
        test_only_shared=args.shared_with_hint_test_only,
        verbose=True,
    )


if __name__ == "__main__":
    main()
