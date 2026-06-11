"""Reproduction / faithfulness check for the wrapped ChemAP, reusing the exact load_students +
predict code path the dsm pipeline uses (run_experiment.py) plus ChemAP's own split/featurization.

Two checks:
  1. DrugApp seed-7 Drug split (in-distribution). Reports train/valid/test. If all three score
     ~equally high, the released checkpoint has effectively seen the whole CSV (no in-file split is
     truly held out), so we cannot reproduce the paper's held-out DrugApp number (AUROC 0.694 /
     AUPRC 0.851) from it — and a ~0.5 here would instead mean a broken wrapper.
  2. External set (FDA-2023 approved=1 + ClinicalTrials-2024 failed=0): molecules curated after
     DrugApp, so genuinely UNSEEN — the only leakage-free generalization check available. Reported
     both unfiltered and with ChemAP's 0.7-Tanimoto similarity filter against DrugApp (the checkpoint
     saw all of DrugApp, so we filter against the full set). External is positive-skewed (~87%), so
     read AUROC over AUPRC.

Run from chemap/:  uv run --project . python repro_drugapp.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from sklearn.metrics import average_precision_score, roc_auc_score

from src.Dataprocessing import DatasetSplit
from src.utils import calculate_tanimoto_similarity
from run_experiment import _seed, load_students, predict

RDLogger.DisableLog("rdApp.*")


def _metrics(tag, y_true, y_proba):
    print(f"[repro] {tag:26s} n={len(y_true):4d} pos={y_true.mean():.3f}  "
          f"AUROC={roc_auc_score(y_true, y_proba):.4f}  AUPRC={average_precision_score(y_true, y_proba):.4f}",
          flush=True)


def _valid_smiles(df):
    ok = df["SMILES"].map(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    return df[ok].reset_index(drop=True), int((~ok).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drugapp", default="dataset/DrugApp/All_training_feature_vectors.csv")
    ap.add_argument("--fda", default="dataset/FDA/FDA_2023_approved.csv")
    ap.add_argument("--clinical", default="dataset/ClinicalTrials/clinical_fail_2024_05.csv")
    ap.add_argument("--model-path", default="./model/ChemAP")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--similarity-cut", type=float, default=0.7)
    args = ap.parse_args()
    device = torch.device("cpu")

    _seed(args.seed)
    np.random.seed(args.seed)
    drugapp = pd.read_csv(args.drugapp)
    train, valid, test = DatasetSplit(drugapp, split="Drug").data_split()

    vocab, ecfp_student, smiles_student = load_students(args.model_path, True, args.seed, device)

    def pred(df):
        return predict(df.reset_index(drop=True), vocab, ecfp_student, smiles_student, device)

    print(f"== DrugApp seed-{args.seed} Drug split (in-distribution) ==")
    for name, d in (("DrugApp/train", train), ("DrugApp/valid", valid), ("DrugApp/test", test)):
        _metrics(name, d["Label"].to_numpy(int), pred(d))

    print("== External (FDA-2023 approved=1 + ClinicalTrials-2024 failed=0; truly unseen) ==")
    fda = pd.read_csv(args.fda)[["Drug Name", "SMILES"]].dropna()
    fda["Label"] = 1
    clin = pd.read_csv(args.clinical)[["Name", "SMILES"]].dropna()
    clin.columns = ["Drug Name", "SMILES"]
    clin["Label"] = 0
    ext, n_inv = _valid_smiles(pd.concat([fda, clin]).reset_index(drop=True))
    if n_inv:
        print(f"[repro] (dropped {n_inv} invalid-SMILES external rows)")
    _metrics("External/all", ext["Label"].to_numpy(int), pred(ext))

    # Paper-rigorous: drop external drugs with Tanimoto > cut to ANY DrugApp molecule.
    train_smiles = drugapp["SMILES"].tolist()
    keep = ext["SMILES"].map(
        lambda s: max(calculate_tanimoto_similarity(s, t) for t in train_smiles) <= args.similarity_cut)
    ext_f = ext[keep].reset_index(drop=True)
    _metrics(f"External/sim<={args.similarity_cut}", ext_f["Label"].to_numpy(int), pred(ext_f))
    print("(paper DrugApp-test: AUROC 0.694 / AUPRC 0.851. External is the unseen-molecule check.)")


if __name__ == "__main__":
    main()
