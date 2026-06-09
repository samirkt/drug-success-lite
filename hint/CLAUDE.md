# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

HINT (Hierarchical Interaction Network) — a deep learning model for clinical trial outcome prediction. Built around the **TOP** benchmark, which curates ClinicalTrials.gov + DrugBank + ICD-10 + MoleculeNet ADMET data into per-phase train/valid/test CSVs.

## Environment

The repo ships **two** environment specs:

- `conda.yml` — original 2020 pinning: Python 3.7, PyTorch 1.2, RDKit 2020.03. Python 3.7 is EOL and will not install on Apple Silicon. Treat this as historical reference only.
- `pyproject.toml` — uv-managed modern stack: Python 3.10, PyTorch 2.11, RDKit 2026.x, scikit-learn 1.7. **This is the supported way to run the code.** Created and verified on macOS arm64 (Apr 2026).

```bash
uv sync                         # creates .venv/ from pyproject.toml + uv.lock
uv run python HINT/learn_phaseI.py
```

All commands below assume the venv is active (or prefixed with `uv run`) and **the working directory is the repo root** (the scripts use relative paths like `data/...` and `save_model/...`, and `learn_*.py` does `sys.path.append('.')`).

### torch 1.2 → 2.x port notes

- All `torch.load(...)` calls in `HINT/learn_*.py` and `HINT/sponsor_inference.py` were patched to pass `weights_only=False`. PyTorch 2.6+ flipped this default to `True`, which rejects pickled user classes (`HINT.model.HINTModel`, `ADMET`, etc.) by design. The full HINT models are saved as whole nn.Modules, not state_dicts, so loading them needs the legacy unpickler.
- The original torch 1.2 checkpoints in `save_model/legacy/` (`admet_model.ckpt`, `phase_{I,II,III}.ckpt`) cannot be loaded by torch 2.x and were moved aside. The current `save_model/admet_model.ckpt` and `save_model/phase_I.ckpt` were retrained under the uv stack.
- `torch.autograd.Variable` is still used in `HINT/model.py` and `HINT/icdcode_encode.py`. It's a deprecated no-op wrapper in modern torch — works fine, no port needed.
- Seaborn's `distplot` (called from `HINT/utils.py`) emits a deprecation warning but still runs.

## Common commands

Train + evaluate per-phase models (each script trains, then writes `save_model/<base>.ckpt`; on subsequent runs it loads the checkpoint and only runs `bootstrap_test`):

```bash
uv run python HINT/learn_phaseI.py
uv run python HINT/learn_phaseII.py
uv run python HINT/learn_phaseIII.py
uv run python HINT/learn_indication.py     # phases I–III combined
```

Run the saved phase models over an external CSV, stratified by phase:

```bash
uv run python HINT/run_inference.py --input rows.csv --out-json metrics.json
```

The input must have HINT's 10 columns (`nctid, status, why_stop, label, phase, diseases, icdcodes, drugs, smiless, criteria`). `phase` accepts `Phase 1/2/3` or `Phase I/II/III`. `icdcodes` accepts both flat lists (`['Z63.72']`) and HINT's native list-of-lists format. The script splits by phase, dispatches to `save_model/phase_{I,II,III}.ckpt`, and reports PR-AUC / F1 / ROC-AUC per phase.

There is no test suite, lint config, or build step. The two Jupyter notebooks (`tutorial_HINT.ipynb`, `tutorial_benchmark.ipynb`) at the repo root are the interactive walkthroughs.

To force retraining, delete `save_model/<phase>.ckpt`. The ADMET pretraining checkpoint at `save_model/admet_model.ckpt` is shared across all phase scripts — delete it to also rerun pretraining.

Device defaults to CPU (`device = torch.device("cpu")` is hardcoded in each `learn_*.py`); change there to use CUDA.

## Data regeneration pipeline (benchmark/)

The repo ships with processed CSVs in `data/`, so retraining does NOT require regenerating data. The pipeline below only matters if changing the trial selection or refreshing from ClinicalTrials.gov. Run from the repo root, in order — each step writes inputs the next consumes:

```bash
mkdir -p raw_data && cd raw_data
wget https://clinicaltrials.gov/AllPublicXML.zip && unzip AllPublicXML.zip && cd ..
find raw_data/ -name 'NCT*.xml' | sort > data/all_xml      # ~370K trial IDs

python benchmark/collect_disease_from_raw.py   # disease text → ICD-10 (~2 hrs, hits ClinicalTable API)
python benchmark/drug2smiles.py                # drug name → SMILES via DrugBank
python benchmark/collect_raw_data.py | tee data_process.log   # → data/raw_data.csv
python benchmark/nctid2date.py                 # → data/nctid_date.txt
python benchmark/data_split.py                 # → data/phase_{I,II,III}_{train,valid,test}.csv
python benchmark/icdcode_encode.py             # → data/icdcode2ancestor_dict.pkl
python benchmark/protocol_encode.py            # BERT sentence embeddings → data/sentence2embedding.pkl
```

Each per-trial CSV row has: `nctid, status, why_stop, label (0/1), phase, diseases, icdcodes, drugs, smiless, criteria`.

## Architecture

The model is built up in three layers of inheritance in `HINT/model.py`:

1. **`Interaction`** (model.py:17) — concatenates outputs from the three encoders into a feature vector → linear → highway → logit. The simplest baseline.
2. **`HINT_nograph`** (model.py:397) — adds the ADMET-pretrained branches and per-aspect heads (efficacy, safety, etc.) without the trial-component graph.
3. **`HINTModel`** (model.py:503) — the full HINT: stacks a GCN over the encoder outputs, modeling interactions between drug / disease / protocol as a graph.

`learn_phase*.py` instantiate `HINTModel`; `Interaction` and `HINT_nograph` exist for ablations. `HINTModel_multi` (model.py:265) is for the multi-phase joint task in `learn_multiple_aim.py`.

**Three encoders feed every model**, and each lives in its own file:

- `molecule_encode.py` — `MPNN` message-passing network over molecule graphs (built from SMILES via RDKit). `ADMET` is a separate multi-task head trained against `data/ADMET/` to pretrain `MPNN` weights before HINT training.
- `icdcode_encode.py` — `GRAM` attention over the ICD-10 ancestor hierarchy. Requires `data/icdcode2ancestor_dict.pkl` (built by `benchmark/icdcode_encode.py` from the ontology files in `icdcode/`). `build_icdcode2ancestor_dict()` is the entry point.
- `protocol_encode.py` — `Protocol_Embedding` reads precomputed BioBERT sentence embeddings from `data/sentence2embedding.pkl` and pools them per trial.

`HINT/module.py` holds the reusable `Highway` and `GCN` blocks. `HINT/dataloader.py` defines `Trial_Dataset` and the `csv_three_feature_2_dataloader` factory used by every `learn_*.py`. There is also a `Trial_Dataset_Complete` variant that keeps the full row (status, phase, drug names, …) for interpretation/inference rather than training.

**Pretraining flow** (in every `learn_*.py`): if `save_model/admet_model.ckpt` is missing, train an `ADMET` model on the five MoleculeNet datasets in `data/ADMET/`, save it, then `model.init_pretrain(admet_model)` copies the MPNN weights into the HINT model before phase training begins. This pretraining is shared across phase scripts via the single checkpoint path.

## Path / cwd gotchas

- All scripts hardcode `data/`, `save_model/`, and `figure/` as relative paths — running from any other cwd will fail with FileNotFoundError or silently retrain.
- `learn_*.py` calls `os.makedirs("figure")` if missing; deleting `figure/` between runs is fine.
- `protocol_encode.py` (under `benchmark/`) and `HINT/protocol_encode.py` are different files — the benchmark one builds `sentence2embedding.pkl`, the HINT one consumes it. Don't conflate them.
- `icdcode/` (top level) holds raw ICD-10 + CCS reference files; `HINT/icdcode_encode.py` and `benchmark/icdcode_encode.py` are again distinct — benchmark builds the ancestor dict, HINT loads and uses it as `GRAM`.

## Metrics

`learn_*.py` reports PR-AUC, F1, ROC-AUC via `bootstrap_test`. Existing prediction CSVs are in `results/` and trained checkpoints in `save_model/` for all three phases.

### Reproduced on the uv stack (single seed, CPU)

| Phase | PR-AUC          | F1              | ROC-AUC         |
|-------|-----------------|-----------------|-----------------|
| I     | 0.5754 ± 0.0176 | 0.6804 ± 0.0176 | 0.5835 ± 0.0173 |
| II    | 0.6196 ± 0.0145 | 0.6109 ± 0.0123 | 0.6280 ± 0.0120 |
| III   | 0.8062 ± 0.0126 | 0.8001 ± 0.0096 | 0.6925 ± 0.0144 |

Phase I matches the in-script baseline comment in `learn_phaseI.py` (0.5645 / 0.6619 / 0.5760). The README's headline numbers (e.g. Phase I PR-AUC 0.745) come from the paper's full setup, not the default `learn_phase*.py` configuration.
