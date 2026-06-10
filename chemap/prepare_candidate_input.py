"""Adapter: candidate_detail.parquet -> ChemAP custom-mode input CSV.

ChemAP's `custom` inference path (ChemAP.py:100-102) reads a CSV from ./dataset/
and only requires a `SMILES` column; External_Dataset generates all features on
the fly. This script bridges the parquet dataset to that interface WITHOUT touching
ChemAP: it renames the SMILES column, drops null / RDKit-invalid rows (the custom
path does no filtering and would crash on an unparseable SMILES), and carries
candidate_id / drug_name through so the output predictions stay joinable.
"""
import argparse
import os

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence RDKit parse warnings; we report counts ourselves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="candidate_detail.parquet",
                        help="input parquet path")
    parser.add_argument("--out", default="./dataset/candidate_detail_input.csv",
                        help="output CSV path (must live under ./dataset/ for ChemAP custom mode)")
    arg = parser.parse_args()

    df = pd.read_parquet(arg.parquet)
    n_total = len(df)

    # SMILES source: prefer standardized canonical, fall back to raw where canonical is null.
    if "smiles_canonical" in df.columns:
        smiles = df["smiles_canonical"]
        if "smiles" in df.columns:
            smiles = smiles.fillna(df["smiles"])
    elif "smiles" in df.columns:
        smiles = df["smiles"]
    else:
        raise KeyError("parquet has neither 'smiles_canonical' nor 'smiles' column")

    out = pd.DataFrame({"SMILES": smiles})
    for col in ("candidate_id", "drug_name"):  # pass-through for downstream joining
        if col in df.columns:
            out[col] = df[col].values

    # Drop null / empty SMILES.
    out["SMILES"] = out["SMILES"].astype("string").str.strip()
    null_mask = out["SMILES"].isna() | (out["SMILES"] == "")
    n_null = int(null_mask.sum())
    out = out[~null_mask].reset_index(drop=True)

    # Drop RDKit-unparseable SMILES — this is the filter that prevents the
    # None-fingerprint -> ragged-array crash in External_Dataset.
    valid_mask = out["SMILES"].map(lambda s: Chem.MolFromSmiles(s) is not None)
    n_invalid = int((~valid_mask).sum())
    out = out[valid_mask].reset_index(drop=True)

    os.makedirs(os.path.dirname(arg.out), exist_ok=True)
    out.to_csv(arg.out, index=False)

    print(f"rows in:        {n_total}")
    print(f"dropped null:   {n_null}")
    print(f"dropped invalid:{n_invalid}")
    print(f"rows written:   {len(out)}  -> {arg.out}")


if __name__ == "__main__":
    main()
