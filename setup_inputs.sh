#!/usr/bin/env bash
# Populate the immutable dataset once, by copying the four parquets from the
# sibling drug-success-report-derived repo. Edit SRC for your machine.
#
# The dataset is treated as an IMMUTABLE INPUT: nothing in dsm/ writes to
# inputs/. Re-run this only to refresh from a new upstream snapshot.
set -euo pipefail

SRC="${1:-../drug-success-model/inputs}"
DST="$(cd "$(dirname "$0")" && pwd)/inputs"

mkdir -p "$DST/features"
cp "$SRC/candidate_detail.parquet"            "$DST/candidate_detail.parquet"
cp "$SRC/trial_detail.parquet"                "$DST/trial_detail.parquet"
cp "$SRC/features/fingerprints.parquet"       "$DST/features/fingerprints.parquet"
cp "$SRC/features/molformer_embeddings.parquet" "$DST/features/molformer_embeddings.parquet"

echo "Copied 4 parquets into $DST"
ls -lh "$DST" "$DST/features"
