# Plan: Stratify each model's test set by seen vs unseen drugs

## Context

The headline numbers (e.g. XGBoost 0.747 on our data) are suspected to be partly
**drug-identity memorization**: a drug seen in training is easy to score again.
To quantify this, we split every existing model's TEST set into **seen drugs** (the
drug appeared in that model's training rows) vs **unseen drugs**, and report
ROC-AUC, PR-AUC, and F1 on each stratum. If seen ≫ unseen, the headline is inflated
by leakage; if they're close, the signal is genuine.

This is **purely evaluation-side** — every experiment already has a saved
`runs/<exp>/predictions.parquet`, so **no retraining is needed**. We re-score the
existing predictions, sliced by drug-seen membership.

Confirmed decisions:
- **F1 threshold**: max-F1 per evaluation set (sweep thresholds, take the best F1)
  — computed independently for ALL / SEEN / UNSEEN so each is its own ceiling.
- **Multi-drug rule**: a test row is "seen" if **any** of its drugs appeared in
  training (the memorization-prone definition).
- **Scope**: the 6 experiments that currently have predictions on disk
  (xgb_di_2019, xgb_di_md, hint_di_2019, xgb_bench_p1, hint_bench_p1,
  hint_bench_p1_repro).

## Empirical grounding (already measured)

| dataset | drug identity | test seen / unseen | multi-drug |
|---|---|---|---|
| `ours_di` | `candidate_id` prefix before `__` (DrugBank id) | 2270 (86%) / 367 (14%) | none |
| `hint_p*` | the row's SMILES (no drug column) | P1: 273 (44%) / 354 (56%) | 50% of rows |

Predictions join 100% to their dataset on `example_id` (including the native-repro
nctids, which match `hint_p1` test exactly). Both strata have both classes, so
ROC/PR are defined everywhere.

## Approach

Drug identity is **per-row a list of drug keys** (to support the any-drug rule):
- `dsm` datasets: `[candidate_id.split("__")[0]]` (single DrugBank id).
- `hint_benchmark` datasets: the deduped list of the row's `smiles` (molecule
  identity; benchmark has no drug ids).

Membership window = **train + valid** (everything the model was fit/selected on;
test excluded). `train_keys = ⋃ drug_keys over rows with split ∈ {train, valid}`.
A test row is **seen** iff `any(k in train_keys for k in row.drug_keys)`.

Each experiment maps to one canonical dataset for membership:
- `spec.dataset` when set (xgb_di_2019→ours_di, hint_bench_p1→hint_p1, …).
- native-repro: `native_benchmark` phase → `hint_p{1,2,3}` (same underlying data;
  nctids match, train+valid drugs are identical to the native train/valid CSVs).

Per stratum (ALL / SEEN / UNSEEN) compute:
- `roc_auc` = `roc_auc_score`, `pr_auc` = `average_precision_score` (threshold-free),
- `f1` = **max over thresholds** via `precision_recall_curve` → `f1 = 2PR/(P+R)`,
  reported with the `f1_threshold` that achieves it,
- plus `n`, `n_pos`. Single-class / tiny strata (`n < 20`) report counts + NaN.

## Files — one new file, nothing else changed

### NEW — `stratify_seen_drug.py` (repo root, self-contained)
A single script run with `uv run python stratify_seen_drug.py`. It **edits no
existing file**; it only *imports* `dsm.experiments` (read-only) to learn which
dataset each experiment used. Contents:
- `drug_keys(dataset_df, kind) -> pd.Series[list[str]]` — kind-dispatched
  (`candidate_id` prefix before `__` for dsm datasets; the deduped `smiles` list
  for the benchmark).
- `seen_mask(dataset_df, kind)` — `train_keys` = union of drug_keys over
  split∈{train,valid}; each test row seen iff any of its keys ∈ `train_keys`.
- `strat_metrics(y, proba)` — `roc_auc_score`, `average_precision_score`, and
  max-F1 via `precision_recall_curve` (+ the achieving `f1_threshold`), n, n_pos.
- a small `_NATIVE_TO_DATASET = {"phase_I":"hint_p1", ...}` to resolve native-repro
  experiments to their canonical dataset for membership.
- main loop: for each `runs/*/predictions.parquet` whose experiment is known,
  resolve + materialize-if-needed its dataset (via `dsm.datasets.materialize` /
  `dsm.experiments.DATASETS`, read-only use), join predictions on `example_id`,
  and print a table + write `runs/stratified_summary.csv`.

Why a script, not a `dsm` subcommand: this is an occasional analysis over
already-saved predictions, so it doesn't need to live in the package or the CLI.
Zero churn to `datasets.py`, `cli.py`, `experiments.py`, adapters, or `hint/`; no
re-materialize (drug keys come from columns already present); no schema change.
If we later want it as a first-class `dsm stratify` command, the script's three
functions lift into `dsm/stratify.py` unchanged.

## Output format (`runs/<exp>/stratified.json`)
```json
{
  "experiment": "xgb_di_2019",
  "drug_identity": "candidate_id_prefix",   // or "smiles" for benchmark
  "seen_rule": "any",
  "all":    {"n":2637,"n_pos":624,"roc_auc":..,"pr_auc":..,"f1":..,"f1_threshold":..},
  "seen":   {"n":2270,"n_pos":510, ...},
  "unseen": {"n":367, "n_pos":103, ...}
}
```
Printed table groups the three strata per experiment so the seen−unseen ROC gap is
read at a glance.

## Verification
1. `uv run python stratify_seen_drug.py` prints all 6 experiments × 3 strata and
   writes `runs/stratified_summary.csv`.
2. Counts sanity: `xgb_di_2019` seen/unseen = 2270/367; `hint_bench_p1` = 273/354
   (match the measured splits). `seen.n + unseen.n == all.n` for every experiment.
3. The memorization read: expect `xgb_di_2019` seen ROC > unseen ROC; report the
   gap. (hint_bench_p1 and its repro join on the same hint_p1 membership.)
4. Native-repro check: `hint_bench_p1_repro` example_ids resolve against `hint_p1`
   with zero unmatched (already confirmed 100%).
5. Inline self-check at the bottom of the script (run under `__main__`): a
   synthetic multi-drug frame confirms the any-drug `seen_mask` rule and that
   `strat_metrics` returns the F1-maximizing threshold on a toy vector. (No change
   to `tests/`.)
