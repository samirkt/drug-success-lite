# drug-success-lite

A stripped-down, bare-bones phase-transition success model. It does exactly one
thing: **load candidate + trial data → train a model → output metrics.** No
ablation, RFE, PDF reports, web serving, calibration, or dimensionality
reduction — just the data, the five feature groups, a model, and the numbers.

Extracted from `drug-success-model` to make experimentation cheap and fast.

## The dataset is an immutable input

The four parquets live under `inputs/` (gitignored) and are treated as
**read-only**: nothing in `dsm/` ever writes to `inputs/`. Populate them once:

```bash
./setup_inputs.sh                 # copies from ../drug-success-model/inputs
./setup_inputs.sh /path/to/inputs # or point at any snapshot
```

Expected layout:

```
inputs/
  candidate_detail.parquet
  trial_detail.parquet
  features/fingerprints.parquet
  features/molformer_embeddings.parquet
```

## Install & run

```bash
uv sync

# One model, all five feature groups, temporal split at 2019:
uv run python -m dsm train --granularity trial --time-split-year 2019 --output runs/t2019

# Drug-indication granularity:
uv run python -m dsm train --granularity drug_indication --time-split-year 2019 --output runs/di2019

# Pick specific groups (honored, unlike the old repo):
uv run python -m dsm train --groups molecule disease admet --output runs/mda

# Reproduce the old group-set sweep (one model per set):
uv run python -m dsm train --granularity both --sweep --time-split-year 2019 --output runs/sweep

uv run pytest tests/ -q
```

Outputs land under `runs/<name>/<granularity>/`:
`metrics.json`, `predictions.csv`, `feature_importances.csv`, plus a top-level
`metrics.csv` summarizing every run.

## Layout

```
dsm/
  config.py     # ModelingConfig / LabelConfig / FeatureConfig dataclasses
  dataset.py    # read-only parquet loaders + join + label  (immutable input)
  splits.py     # temporal or stratified-random train/test split
  encoders.py   # six generic sub-encoders (Scalar, MultiHot, DenseArray, ...)
  features.py   # five composite feature groups + registry
  model.py      # xgb + logreg wrappers behind a tiny registry
  evaluate.py   # metrics() + per-phase metrics + optional bootstrap CIs
  train.py      # train_one_run() + run-artifact writer
  cli.py        # `python -m dsm train ...`
```

## Granularities

- `drug_indication` — one row per candidate; label from `candidate.outcome`
  (Approved/Commercialized = 1, Failed Phase 1/2/3 = 0; Ongoing dropped).
- `trial` — one row per NCT; label from `trial_inferred_label`. Produces
  per-phase metrics (P1→P2, P2→P3, P3→approval).

## HINT comparison (best HINT, apples-to-apples)

Retrain the **best-performing HINT** (its three per-phase `HINTModel`s) on the
**exact same rows + split** as this repo, **eligibility-less** (no criteria — feature
parity, since this model has no criteria feature), and compare per-phase on the
shared test set. A single exported CSV is the contract; the sibling HINT repo
(`../hint_standalone/repo`) does the training.

```bash
# 1. Train your model (trial granularity) — predictions.csv keyed on nct_id+phase.
uv run python -m dsm train --granularity trial --time-split-year 2019 --output runs/t2019

# 2. Export the HINT-format CSV (10 cols + a train/test split tag from the SAME cutoff).
#    Only HINT-ingestible rows are emitted (valid SMILES + non-empty ICD-10); criteria blanked.
uv run python -m dsm export-hint --time-split-year 2019 --output runs/hint/hint_export.csv

# 3. In the HINT repo, retrain the 3 per-phase models on the export (NOT benchmark weights).
cd ../hint_standalone/repo
bash runs/dsm_best/run.sh /abs/path/to/drug-success-lite/runs/hint/hint_export.csv   # EPOCHS=5 default
#   -> runs/dsm_best/results/nctid2predict.pkl   (merged per-NCT test predictions)

# 4. Compare on the shared test nct_ids (inner-join), per-phase + overall.
cd -
uv run python -m dsm compare-hint \
  --predictions runs/t2019/trial/predictions.csv \
  --hint-predictions ../hint_standalone/repo/runs/dsm_best/results/nctid2predict.pkl \
  --output runs/hint/comparison.csv
```

**Why the comparison set is smaller than the full test set:** this model tolerates
missing features (every encoder has a `_missing` indicator), but HINT *requires* a
parseable SMILES and a non-empty ICD-10 list, and only models Phase 1/2/3.
`export-hint` reports exactly how many rows that drops (empty ICD codes dominate),
and `compare-hint` evaluates *both* models only on the shared `nct_id`s so neither is
scored on rows the other never saw. The split tag is driven off your
`--time-split-year`, so HINT trains/tests on your partition, not its native 2014
benchmark split.

`EPOCHS` defaults to 5. HINT keeps the best-validation checkpoint and validation loss
bottoms around epoch ~3 (then overfits), so 5 is plenty — more epochs don't change the
saved model. The benchmark `phase_*.ckpt` weights are never reused; this trains fresh
on your data.
