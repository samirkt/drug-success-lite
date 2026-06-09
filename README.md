# drug-success-lite

A small research repo for clinical-trial / drug-approval success modeling. One
entry point, one data location, one standardized metrics format — for **both** our
XGBoost-style models and the **HINT** deep model (pulled in under `hint/`).

```bash
uv run python -m dsm list                  # datasets + experiments
uv run python -m dsm run xgb_di_2019        # -> runs/xgb_di_2019/metrics.json
uv run python -m dsm run hint_bench_p1_repro --epochs 5
uv run python -m dsm run --all
```

To add an experiment, add one line to `dsm/experiments.py` — no new code.

## How it fits together

```
data/                         ONE ground-truth data location
  candidate_detail.parquet  trial_detail.parquet  features/*.parquet   (our data)
  hint_benchmark/  ->  ../hint/data        (symlink; HINT's TOP benchmark CSVs)
  datasets/<name>.parquet                  canonical example files (built on demand)

dsm/                          our package (sklearn / xgboost, py3.11, numpy>=2)
  datasets.py    the SINGLE materializer -> canonical example parquet
  models/        adapters: sklearn_adapter (xgb/logreg, in-process) + hint_adapter (subprocess)
  experiments.py declarative dataset + experiment registry
  run.py         resolve experiment -> materialize -> adapter -> evaluate -> metrics.json
  evaluate.py    canonical predictions parquet -> metrics (overall + per-phase)
  features.py / encoders.py / dataset.py / splits.py / model.py / config.py

hint/                         pulled-in HINT (torch, py3.10, numpy<2 — its OWN uv venv)
  run_experiment.py           the single HINT entry: canonical in -> predictions out
  HINT/ ...                   encoders + model, reused unchanged
```

`dsm` and `hint` live in **incompatible Python environments**, so the only thing
that crosses between them is a parquet file. `dsm/models/hint_adapter.py` shells
`uv run --project hint python run_experiment.py ...`.

## The dsm ↔ HINT contract (two files, defined once)

Every modeling decision (label, train/test split, ICD format, criteria, row
filtering, phase naming) is made **once**, in `dsm/datasets.py`. Both model
families then consume one schema and emit another:

**Canonical example** — `data/datasets/<name>.parquet`:
`example_id, label, phase, smiles (list), icd_codes (flat list), criteria, split`
(+ rich admet/target/pathway columns for our data; absent for the benchmark).

**Canonical predictions** — `runs/<exp>/predictions.parquet`:
`example_id, label, phase, y_proba`. One evaluator turns this into
`runs/<exp>/metrics.json`. Comparing two models = run two experiments on the same
dataset and read two `metrics.json` (no bespoke join).

The molecule feature is **always** ECFP4+MACCS from the canonical `smiles`, and
the benchmark/our-data disease feature is the same `icd_codes` HINT's GRAM sees —
so xgb and HINT consume identical molecule+disease inputs on any shared dataset.
The flat-ICD → HINT's nested list-of-lists string is the *only* conversion, and it
lives in one function (`hint/run_experiment.py:to_hint_cells`), guarded by a
round-trip test.

## Data setup (immutable inputs)

Our parquets and HINT's benchmark are large, gitignored, and regenerable. Our four
parquets live under `data/`; HINT ships its TOP benchmark + assets under
`hint/data/` (exposed at `data/hint_benchmark/` via symlink so the native HINT
scripts' cwd-relative paths still resolve). `hint/data/sentence2embedding.pkl`
(1.1 GB) is needed only for the criteria-on reproduction path; the criteria-less
path uses the tiny `sentence2embedding_stub.pkl` automatically.

```bash
uv sync                       # dsm venv
uv sync --project hint        # HINT venv (torch, rdkit, numpy<2)
```

## Experiments

- **our data**: `xgb_di_2019` (all 5 groups), `xgb_di_md` (molecule+disease),
  `hint_di_2019` (HINT on the same rows).
- **benchmark comparison** (canonical, identical population): `xgb_bench_p{1,2,3}`
  vs `hint_bench_p{1,2,3}` — apples-to-apples on smiles+icd.
- **benchmark reproduction** (HINT native path, real 3-way split + real criteria):
  `hint_bench_p{1,2,3}_repro` — reproduces the published per-phase numbers.

```bash
uv run pytest tests/ -q
```

Metric to compare across datasets: **ROC-AUC** (PR-AUC/F1 move with base rate).
