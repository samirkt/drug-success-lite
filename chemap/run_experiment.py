"""The single ChemAP entry point: canonical example in -> canonical predictions out.

Mirrors hint/run_experiment.py's contract so the dsm `chemap` adapter can shell into the
chemap/ venv and get back the standard predictions parquet (example_id,label,phase,y_proba)
for the test rows. dsm does everything else (label, split, row-filter, eval, stratify).

ChemAP is used as a PRETRAINED BLACK BOX (transfer): we load its released DrugApp-trained
student checkpoints and run inference over each row's SMILES. ChemAP is SMILES-only, so we
take the first SMILES of each canonical `smiles` list and ignore disease/criteria/phase.

This imports ChemAP's own classes (src/) UNMODIFIED — featurization (ECFP-2048 +
SMILES-BERT tokens/adjacency), the two students, and the soft-vote are exactly ChemAP's.
The only ChemAP-side dependency we drop is torch_geometric (only ChemAP.py used it); we use
torch's stock DataLoader. Inference needs only the two student checkpoints — NOT ChemBERT or
the Teacher (the SMILES student's state_dict already carries its fine-tuned encoder).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rdkit import Chem, RDLogger

from src.Dataprocessing import External_Dataset
from src.models import FP_Student, SMILES_BERT, SMILES_Student
from src.utils import Vocab

RDLogger.DisableLog("rdApp.*")


def _seed(seed: int) -> None:
    """CPU-safe seeding. ChemAP's src.utils.seed_everything calls a CUDA-only RNG getter that
    crashes on CPU-only torch; we replicate its CPU-relevant parts without touching ChemAP."""
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ChemAP's published student-model hyperparameters (ChemAP.py defaults).
FP_DIMS = (1024, 128, 256)
FP_DROPS = (0.21, 0.11)
SEQ_LEN = 256


def first_smiles(cell) -> str | None:
    """First non-empty SMILES of a canonical `smiles` list (ChemAP is per-molecule)."""
    for s in (cell if cell is not None else []):
        s = str(s).strip()
        if s:
            return s
    return None


def rows_from_canonical(path):
    """Test rows only (pretrained transfer: no training on our data). One SMILES per row;
    drop null / RDKit-unparseable SMILES (External_Dataset would crash on a None fingerprint)."""
    df = pd.read_parquet(path)
    df = df[df["split"] == "test"].reset_index(drop=True)
    df["SMILES"] = df["smiles"].map(first_smiles)
    n_test = len(df)

    null_mask = df["SMILES"].isna()
    df = df[~null_mask].reset_index(drop=True)
    valid_mask = df["SMILES"].map(lambda s: Chem.MolFromSmiles(s) is not None)
    n_invalid = int((~valid_mask).sum())
    df = df[valid_mask].reset_index(drop=True)

    print(f"[chemap] test rows={n_test} dropped(null={int(null_mask.sum())}, "
          f"invalid={n_invalid}) -> scoring {len(df)}", flush=True)
    return df[["example_id", "label", "phase", "SMILES"]]


def _resolve_ckpt(model_path, kd_suffix, subdir, stems, seed):
    """Find a student checkpoint, tolerating both the ChemAP.py convention (*_predictor_{seed}.pt)
    and the released filenames (*_student_{seed}.pt)."""
    d = os.path.join(model_path, f"{subdir}{kd_suffix}")
    candidates = [os.path.join(d, f"{stem}_{seed}.pt") for stem in stems]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"ChemAP checkpoint missing in {d} (looked for: {', '.join(os.path.basename(c) for c in candidates)})\n"
        "Download the released student checkpoints from the authors (see chemap/README.md) into "
        "chemap/model/ChemAP/{ECFP_predictor,SMILES_predictor}/.")


def _remap_fp_keys(state_dict):
    """The released ECFP-student weights use the original ChemAP submodule names
    (`ecfp_enc`/`projector`); the vendored src/models.py renamed these to
    `encoder_1`/`encoder_2` (cosmetic — identical layers/dims). Remap so the released
    black-box weights load against the vendored FP_Student without touching ChemAP src."""
    rename = {"ecfp_enc.": "encoder_1.", "projector.": "encoder_2."}
    out = {}
    for k, v in state_dict.items():
        for old, new in rename.items():
            if k.startswith(old):
                k = new + k[len(old):]
                break
        out[k] = v
    return out


def load_students(model_path, kd, seed, device):
    kd_suffix = "" if kd else "_wo_KD"
    ecfp_ckpt = _resolve_ckpt(model_path, kd_suffix, "ECFP_predictor", ("ECFP_predictor", "ECFP_student"), seed)
    smi_ckpt = _resolve_ckpt(model_path, kd_suffix, "SMILES_predictor", ("SMILES_predictor", "SMILES_student"), seed)

    vocab = Vocab()
    ecfp_student = FP_Student(2048, FP_DIMS[0], FP_DIMS[1], FP_DIMS[2], FP_DROPS[0], FP_DROPS[1]).to(device)
    ecfp_student.load_state_dict(_remap_fp_keys(torch.load(ecfp_ckpt, map_location=device)))

    encoder = SMILES_BERT(len(vocab), max_len=SEQ_LEN, nhead=16, feature_dim=1024,
                          feedforward_dim=1024, nlayers=8, adj=True, dropout_rate=0)
    smiles_student = SMILES_Student(encoder, 1024).to(device)
    smiles_student.load_state_dict(torch.load(smi_ckpt, map_location=device))

    ecfp_student.eval()
    smiles_student.eval()
    return vocab, ecfp_student, smiles_student


def predict(df, vocab, ecfp_student, smiles_student, device):
    """ChemAP soft-vote probability per row (ChemAP.py:131-164), order-preserving."""
    dataset = External_Dataset(vocab, df, "custom", device, trainset=None)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    probs = []
    with torch.no_grad():
        for ecfp_2048, smi_input, smi_adj, smi_adj_mask, _y in loader:
            pos_num = torch.arange(SEQ_LEN).repeat(smi_input.size(0), 1).to(device)
            _, ecfp_out = ecfp_student(ecfp_2048)
            _, smi_out = smiles_student(smi_input, pos_num, smi_adj_mask, smi_adj)
            ecfp_prob = F.softmax(ecfp_out, dim=1)[:, 1]
            smi_prob = F.softmax(smi_out, dim=1)[:, 1]
            probs.append(((ecfp_prob + smi_prob) / 2).detach().cpu())
    return torch.cat(probs, dim=0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="canonical example parquet")
    ap.add_argument("--out", required=True, help="canonical predictions parquet to write")
    ap.add_argument("--model-path", default="./model/ChemAP", help="trained ChemAP student dir")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-kd", dest="kd", action="store_false",
                    help="load the non-distilled (_wo_KD) checkpoints")
    ap.add_argument("--features", default="mol", help="ignored; ChemAP is SMILES-only (adapter parity)")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    _seed(args.seed)

    df = rows_from_canonical(args.dataset)
    vocab, ecfp_student, smiles_student = load_students(args.model_path, args.kd, args.seed, device)
    y_proba = predict(df, vocab, ecfp_student, smiles_student, device)
    assert len(y_proba) == len(df), (len(y_proba), len(df))

    preds = pd.DataFrame({
        "example_id": df["example_id"].astype(str).values,
        "label": df["label"].to_numpy(dtype=np.int8),
        "phase": df["phase"].astype(str).values,
        "y_proba": y_proba.astype(float),
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    preds.to_parquet(args.out, index=False)
    print(f"[chemap] wrote {len(preds)} predictions -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
