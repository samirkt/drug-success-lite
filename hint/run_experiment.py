"""The single HINT entry point: canonical example in -> canonical predictions out.

This replaces the old runs/dsm_e2e/{run.sh,split_tag.py,train.py} +
nctid2predict.pkl dance. It runs in the hint/ venv (cwd=hint/), reads one
dataset, trains a HINTModel reusing the existing encoders/model unchanged, and
writes the canonical predictions parquet (example_id,label,phase,y_proba) for the
test rows. The dsm side does everything else (label, split, row-filter, eval).

Two data sources:
  --dataset <canonical.parquet>   the dsm canonical example schema (our data, or
                                  benchmark-canonical). Valid is carved from
                                  split=='train' if the file has no 'valid' rows.
  --native-benchmark phase_I      HINT's own data/phase_I_{train,valid,test}.csv,
                                  verbatim (faithful published-benchmark repro).

Features: a comma list from {mol,disease,criteria}. mol+disease are always on
(this is HINTModel); `criteria` decides whether the real 1.1GB sentence embedding
is loaded (criteria on) or the empty stub (criteria off -> Protocol_Embedding
collapses to zeros). The ONLY canonical->HINT conversion (flat ICD -> nested
list-of-lists string) lives in `to_hint_cells` below.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
from random import Random

csv.field_size_limit(10 ** 9)

# Phase normalizer — mirrors dsm/datasets.py:normalize_phase (kept in sync by hand;
# this is the only intentional duplication, at the repo boundary).
_PHASE_MAP = {
    "phase 1": "P1", "phase 2": "P2", "phase 3": "P3", "phase 4": "P4",
    "phase 1/phase 2": "P1P2", "phase 2/phase 3": "P2P3", "early phase 1": "EarlyP1",
}


def normalize_phase(raw) -> str:
    s = (str(raw).strip().lower() if raw is not None else "")
    return _PHASE_MAP.get(s, "other") if s else "other"


def to_hint_cells(smiles_list, icd_list):
    """Build the exact string-reprs HINT's trial_collate_fn slices.

    smiles -> "['CC', 'CCC']"             (str of the list)
    icd    -> "[\"['C71.7', 'C71.9']\"]"  (str of [ str(flat_list) ]) == the only
              ICD conversion in the whole codebase; HINT's icdcode_text_2_lst_of_lst
              parses it back to [['C71.7','C71.9']].
    """
    smiles = [str(s) for s in (smiles_list if smiles_list is not None else []) if str(s).strip()]
    icd = [str(c) for c in (icd_list if icd_list is not None else []) if str(c).strip()]
    return str(smiles), str([str(icd)])


def carve_valid(rows, valid_frac, seed):
    """Stratified-by-label split of train rows into (train, valid)."""
    rng = Random(seed)
    by_label: dict = {}
    for r in rows:
        by_label.setdefault(int(r["label"]), []).append(r)
    train, valid = [], []
    for _, group in sorted(by_label.items()):
        rng.shuffle(group)
        n_va = int(len(group) * valid_frac)
        valid += group[:n_va]
        train += group[n_va:]
    rng.shuffle(train)
    return train, valid


# --------------------------------------------------------------------------- #
# Row loaders -> list of dicts {example_id,label,phase,smiles_cell,icd_cell,criteria}
# --------------------------------------------------------------------------- #
def rows_from_canonical(path, criteria_on):
    import pandas as pd

    df = pd.read_parquet(path)
    out = {"train": [], "valid": [], "test": []}
    for _, r in df.iterrows():
        smi_cell, icd_cell = to_hint_cells(list(r["smiles"]), list(r["icd_codes"]))
        out[r["split"]].append({
            "example_id": str(r["example_id"]),
            "label": int(r["label"]),
            "phase": str(r["phase"]),
            "smiles_cell": smi_cell,
            "icd_cell": icd_cell,
            "criteria": (str(r["criteria"]) if criteria_on else ""),
        })
    if not out["valid"]:
        out["train"], out["valid"] = carve_valid(out["train"], VALID_FRAC, SEED)
    return out["train"], out["valid"], out["test"]


def rows_from_native(phase_stem, criteria_on):
    """Read HINT's native phase CSVs verbatim (nested ICD, real criteria, 3-way split)."""
    out = []
    for split_name in ("train", "valid", "test"):
        path = os.path.join("data", f"{phase_stem}_{split_name}.csv")
        rows = list(csv.reader(open(path)))[1:]
        recs = []
        for r in rows:
            recs.append({
                "example_id": r[0],
                "label": int(r[3]),
                "phase": normalize_phase(r[4]),
                "smiles_cell": r[8],                 # already the list-repr HINT wants
                "icd_cell": r[6],                    # already nested list-of-lists string
                "criteria": (r[9] if criteria_on else ""),
            })
        out.append(recs)
    return out[0], out[1], out[2]


# --------------------------------------------------------------------------- #
# Build a HINT dataloader straight from row dicts (no temp CSV).
# --------------------------------------------------------------------------- #
def make_loader(rows, shuffle, batch_size=32):
    from torch.utils import data
    from HINT.dataloader import Trial_Dataset, trial_collate_fn

    ds = Trial_Dataset(
        nctid_lst=[r["example_id"] for r in rows],
        label_lst=[r["label"] for r in rows],
        smiles_lst=[r["smiles_cell"] for r in rows],
        icdcode_lst=[r["icd_cell"] for r in rows],
        criteria_lst=[r["criteria"] for r in rows],
    )
    return data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=trial_collate_fn)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="canonical example parquet")
    src.add_argument("--native-benchmark", help="phase stem, e.g. phase_I")
    ap.add_argument("--features", default="mol,disease")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    global VALID_FRAC, SEED
    VALID_FRAC, SEED = args.valid_frac, args.seed

    feats = {f.strip().lower() for f in args.features.split(",") if f.strip()}
    criteria_on = "criteria" in feats
    # Gate the 1.1GB embedding load BEFORE importing HINT.dataloader (module-level load).
    if not criteria_on:
        os.environ["HINT_SENTENCE2VEC"] = os.path.join("data", "sentence2embedding_stub.pkl")

    import torch
    torch.manual_seed(args.seed)
    from HINT.icdcode_encode import GRAM, build_icdcode2ancestor_dict
    from HINT.protocol_encode import Protocol_Embedding
    from HINT.molecule_encode import MPNN
    from HINT.model import HINTModel
    from HINT.learn_advancement import get_or_train_admet

    device = torch.device(args.device)

    if args.dataset:
        train_rows, valid_rows, test_rows = rows_from_canonical(args.dataset, criteria_on)
        tag = os.path.splitext(os.path.basename(args.dataset))[0]
    else:
        train_rows, valid_rows, test_rows = rows_from_native(args.native_benchmark, criteria_on)
        tag = args.native_benchmark
    print(f"[hint] {tag}: train={len(train_rows)} valid={len(valid_rows)} test={len(test_rows)} "
          f"criteria={'on' if criteria_on else 'off'}", flush=True)

    train_loader = make_loader(train_rows, shuffle=True)
    valid_loader = make_loader(valid_rows, shuffle=False)
    test_loader = make_loader(test_rows, shuffle=False)

    admet_model = get_or_train_admet(device)
    icd2anc = build_icdcode2ancestor_dict()
    gram = GRAM(embedding_dim=50, icdcode2ancestor=icd2anc, device=device)
    protocol = Protocol_Embedding(output_dim=50, highway_num=3, device=device)
    mpnn = MPNN(mpnn_hidden_size=50, mpnn_depth=3, device=device)
    model = HINTModel(
        molecule_encoder=mpnn, disease_encoder=gram, protocol_encoder=protocol,
        device=device, global_embed_size=50, highway_num_layer=2,
        prefix_name=f"run_{tag}", gnn_hidden_size=50,
        epoch=args.epochs, lr=args.lr, weight_decay=0,
    )
    model.init_pretrain(admet_model)
    model.learn(train_loader, valid_loader, test_loader)

    # Interaction.generate_predict returns (loss, predict, label, nctid) — key by nctid.
    _, predict_all, label_all, nctid_all = model.generate_predict(test_loader)
    assert len(predict_all) == len(test_rows), (len(predict_all), len(test_rows))
    phase_by_id = {r["example_id"]: r["phase"] for r in test_rows}

    import pandas as pd
    preds = pd.DataFrame({
        "example_id": [str(n) for n in nctid_all],
        "label": [int(round(x)) for x in label_all],
        "phase": [phase_by_id.get(str(n), "other") for n in nctid_all],
        "y_proba": [float(p) for p in predict_all],
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    preds.to_parquet(args.out, index=False)
    print(f"[hint] wrote {len(preds)} predictions -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
