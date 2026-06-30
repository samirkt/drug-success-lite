"""Build a minimized, DrugBank-free copy of the cohort data the web tool serves.

The `/cohort` endpoint (`dsm/cohort.py`) reads two parquets:
  * `data/datasets/ours_di.parquet`        — per-program metadata (~170 columns, many unused)
  * `data/features/fingerprints.parquet`   — ECFP4 bit vectors, joined on example_id==candidate_id

For hosting we ship only what the cohort actually needs, and we strip every DrugBank reference:
  * keep only the columns the cohort reads,
  * replace the DrugBank-derived `drugbank_id` with an opaque `drug_uid` (internal dedup key only),
  * remap the join keys (`example_id` / `candidate_id`, which embed DrugBank ids like
    `db:DB11881__lymphoma`) to opaque surrogates (`ex00001`, …) using one shared mapping so the
    inner join still lines up.

Source parquets are read-only — outputs go to `deploy/space/data/...`, never in place.

    uv run python scripts/make_serving_data.py
"""

from __future__ import annotations

import logging

import pandas as pd

from dsm.config import DATASETS_DIR, DEFAULT_FINGERPRINTS, PROJECT_ROOT

logger = logging.getLogger(__name__)

SRC_DATASET = DATASETS_DIR / "ours_di.parquet"
SRC_FINGERPRINTS = DEFAULT_FINGERPRINTS

OUT_ROOT = PROJECT_ROOT / "deploy" / "space" / "data"
OUT_DATASET = OUT_ROOT / "datasets" / "ours_di.parquet"
OUT_FINGERPRINTS = OUT_ROOT / "features" / "fingerprints.parquet"

# The only dataset columns the cohort reads (besides the join key + drugbank_id, handled below).
# `split` (train/test) is needed for the applicability-domain references, which are computed against
# the model's training pool only; it carries no DrugBank reference, so it's safe to ship.
KEEP_COLUMNS = ["smiles", "drug_name", "indication", "label", "icd_codes", "split"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ds = pd.read_parquet(SRC_DATASET)
    fp = pd.read_parquet(SRC_FINGERPRINTS)[["candidate_id", "ecfp4"]]

    missing = [c for c in (["example_id", "drugbank_id"] + KEEP_COLUMNS) if c not in ds.columns]
    if missing:
        raise SystemExit(f"source dataset is missing expected columns: {missing}")

    # Mirror the cohort's inner join: keep only programs that have a fingerprint.
    fp_ids = set(fp["candidate_id"])
    ds = ds[ds["example_id"].isin(fp_ids)].reset_index(drop=True)
    logger.info("programs with fingerprints: %d / %d", len(ds), len(fp_ids))

    # Opaque, DrugBank-free join keys (shared mapping across both files).
    id_map = {old: f"ex{i:05d}" for i, old in enumerate(ds["example_id"])}

    # Opaque drug-level dedup key derived from drugbank_id (carries no DrugBank id).
    codes, _ = pd.factorize(ds["drugbank_id"])
    drug_uid = [f"d{c:05d}" for c in codes]

    out_ds = ds[KEEP_COLUMNS].copy()
    out_ds.insert(0, "example_id", [id_map[e] for e in ds["example_id"]])
    out_ds["drug_uid"] = drug_uid

    fp = fp[fp["candidate_id"].isin(id_map)].copy()
    fp["candidate_id"] = fp["candidate_id"].map(id_map)

    # Sanity: no DrugBank residue should survive.
    assert "drugbank_id" not in out_ds.columns
    assert not out_ds["example_id"].astype(str).str.contains("DB", case=True).any()
    assert not fp["candidate_id"].astype(str).str.contains("DB", case=True).any()

    OUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    OUT_FINGERPRINTS.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_parquet(OUT_DATASET, index=False)
    fp.to_parquet(OUT_FINGERPRINTS, index=False)

    logger.info("wrote %s (%d rows, columns=%s)", OUT_DATASET, len(out_ds), list(out_ds.columns))
    logger.info("wrote %s (%d rows)", OUT_FINGERPRINTS, len(fp))
    logger.info("dataset size: %.1f MB, fingerprints size: %.1f MB",
                OUT_DATASET.stat().st_size / 1e6, OUT_FINGERPRINTS.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
