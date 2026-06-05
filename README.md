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

## HINT comparison (future)

Not built yet, but the structure is ready. Trial-granularity `predictions.csv`
is keyed on `nct_id` + `trial_phase`, so a future `compare_hint.py` can
inner-join the local predictions against a HINT model's per-NCT output (from a
sibling HINT repo) and group by phase — the same join the old pipeline used.
