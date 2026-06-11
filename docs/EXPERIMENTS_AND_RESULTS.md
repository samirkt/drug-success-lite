# Experiments & Results — working notes

A running summary of the experiments in this repo and what they showed, written for
brainstorming. Three threads: (1) *why is HINT worse than XGBoost* (the representation-vs-
classifier swap), (2) *reconciling our numbers with the HINT paper*, (3) *integrating &
evaluating ChemAP*. Numbers are single-seed unless noted.

---

## Setup / shared context

**Datasets (canonical schema: `example_id, label, phase, smiles, icd_codes, criteria, split`)**
- `ours_di` — **drug-indication**, label = eventual approval (P1→approval). 14,134 rows
  (11,497 train / 2,637 test). Test: **23.7% positive**, **2,270 seen / 367 unseen** drugs.
  3,238 unique drugs overall (2,739 unique SMILES); 32.7% ever-succeed at the molecule level.
- `hint_p1/p2/p3` — HINT's TOP benchmark, phase-transition labels. Our canonical test sets are
  **byte-identical to HINT's native test CSVs** (627 / 1,654 / 1,146 rows; 55% / 55% / 75% pos).

**Seen/unseen:** a test drug is "seen" if it appeared in training. Drug identity = SMILES on the
benchmark, `candidate_id` prefix (`db:DB11881__lymphoma` → `db:DB11881`) on our data. The
seen−unseen gap is the **memorization signal**.

**Models / tooling**
- `xgb` — XGBoost on ECFP4(2048)+MACCS(167) fingerprints + disease features. Strong, tuned.
- `hint` — HINT (MPNN molecule + GRAM disease → interaction-GCN head). Our runs use mol+disease,
  criteria off, 5 epochs.
- `chemap` — ChemAP (SMILES-only drug-approval predictor), pretrained, run as a black box.
- Pipeline: `python -m dsm run <exp>` → predictions + metrics; `dsm stratify` → seen/unseen.

---

## Experiment 1 — Representation vs classifier swap (`dsm.embed_swap`)

**Question:** is HINT worse because of its *representation* (50-d MPNN+GRAM bottleneck) or its
*classifier* (interaction-GCN head)? **Method:** dump HINT's trained 100-d (50 MPNN ⊕ 50 GRAM)
vectors, train XGBoost on them (`xgb_hint_emb`). Controls: full-fingerprint XGB (`xgb_full`) and
XGB on PCA-50(fingerprint)⊕PCA-50(disease) (`xgb_pca50`). Ran on p1/p2/p3 + di.

**ROC-AUC by stratum (all / seen / unseen):**

| target | hint | xgb_full | xgb_hint_emb | xgb_pca50 |
|---|---|---|---|---|
| p1 | .568 / .572 / **.538** | .558 / .574 / .415 | .540 / .582 / **.256** | .529 / .531 / .552 |
| p2 | .621 / .614 / .687 | .599 / .592 / .674 | .631 / .624 / **.700** | .615 / .611 / .665 |
| p3 | .692 / .695 / .645 | .674 / .685 / .516 | .691 / .700 / .590 | .672 / .689 / .412 |
| **di** | .629 / .627 / .630 | **.736** / .757 / .625 | .716 / .724 / .669 | **.757** / .769 / .672 |

**Findings (di is the most reliable — 367 unseen rows vs 86–149 on benchmark):**
- **HINT's deficit is its classifier, not its features.** XGB on HINT's *own* embeddings jumps
  0.629 → **0.716** (same representation, different head).
- **HINT's learned representation isn't special.** `xgb_pca50` (0.757) ≥ `xgb_hint_emb` (0.716) at
  equal dimensionality — a plain PCA of explicit features matches/beats HINT's end-to-end 100-d.
- **The 50-d bottleneck — not HINT specifically — kills memorization.** `xgb_full` (2215-d) has the
  biggest seen−unseen gap and worst unseen; reducing to 50+50 (HINT-learned *or* PCA) shrinks the
  gap and improves unseen.
- **Benchmark phases are noisy / mixed:** p1 shows the dramatic memorization story (xgb below
  chance on unseen), p2 shows *none* (everyone better on unseen than seen), p3 partial.

**Caveats:** single seed; benchmark unseen strata are tiny (86–149) so p1's inversions are partly
noise. Full per-(target,model,stratum) metrics in `runs/embed_swap_summary.csv`.

---

## Experiment 2 — Reconciling with the HINT paper

**Observation:** our XGBoost beats HINT on the benchmark, but the paper reports HINT beating
XGBoost. **Resolution (analysis, not a new run):**
- **HINT reproduces fine.** Paper HINT Phase-I PR-AUC ≈ 0.567 matches our/HINT's default
  reproduction (~0.575). No paper-vs-repro gap.
- **The difference is the XGBoost baseline, not HINT.** Our XGB is stronger than the paper's
  (2215-bit fingerprint + tuning) and exploits the benchmark's heavy seen-drug overlap
  (memorization). HINT sits at ~0.567 in both.
- For `xgb_hint_emb` specifically, we hand XGBoost HINT's *own* learned representation — a model
  the paper's baselines never had — so beating them is expected and not "XGB > XGB".

**Takeaway:** the "XGB > HINT" result here is about a stronger baseline + memorization on a
seen-drug-heavy benchmark, not a failure of HINT to reproduce.

---

## Experiment 3 — ChemAP integration + transfer eval

**What:** vendored ChemAP into `chemap/`, wrapped it as a black-box `dsm` model adapter
(`chemap_di_2019`), and ran the **pretrained** ChemAP (released DrugApp weights) over `ours_di`.
ChemAP is SMILES-only and predicts *drug approval*. No retraining.

**Result on `ours_di` (n=2,637):**

| model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| xgb_di_md | **0.736** | 0.463 | 0.503 |
| hint_di_2019 | 0.595 | 0.328 | — |
| **chemap_di_2019** | **0.462** | 0.218 | 0.353 |

Seen 0.470 / unseen 0.451. **Below chance**, and not degenerate: predictions are well-spread
(0.004–0.996), but ChemAP calls **88% of rows "approve"** (mean prob 0.82) vs a 24% base rate, and
*failed* indications get a slightly **higher** mean score (0.856) than successful ones (0.816) →
mild anti-correlation.

**Interpretation:** the per-molecule-vs-per-indication mismatch. ChemAP scores a *molecule's*
general approvability and can't see the indication; our label is per-(drug, indication) success.
Broadly-approvable drugs pushed into many speculative indications produce many high-score / fail
rows, dragging the ranking below chance.

---

## Experiment 4 — ChemAP faithfulness / reproduction (`chemap/repro_drugapp.py`)

**Question:** is the 0.46 a wrapper bug or real transfer failure? **Method:** run our wrapped
ChemAP on ChemAP's own data and check against the paper (AUROC 0.694 / AUPRC 0.851).

| check | n | AUROC | AUPRC |
|---|---|---|---|
| DrugApp train | 2,498 | 0.967 | 0.976 |
| DrugApp valid | 312 | 0.953 | 0.971 |
| DrugApp test | 312 | 0.969 | 0.984 |
| **External, sim-filtered ≤0.7** | **26** | **0.662** | **0.853** |
| External, unfiltered | 40 | 0.491 | 0.694 |
| *paper (reported)* | — | *0.694* | *0.851* |

**Findings:**
- DrugApp train≈valid≈test (~0.96–0.97) → the released checkpoint has **seen the whole CSV**;
  no in-file split is truly held out (so we can't reproduce a held-out DrugApp number from it).
- On the **leakage-free external set** (FDA-2023 approved vs ClinicalTrials-2024 failed) with the
  paper's 0.7-Tanimoto filter, we get **0.662 / 0.853 ≈ paper 0.694 / 0.851**. → **Integration is
  faithful.**
- A broken wrapper would score ~0.5; we get strong, paper-matching signal. **So the `ours_di` 0.46
  is genuine negative transfer, not a bug.**

**Caveat:** external set is tiny (40 → 26 after filter); AUROC point estimate is noisy, but AUPRC
nails 0.851 and the whole picture is internally consistent.

---

## Consolidated leaderboard — `ours_di` test (n=2,637, 23.7% pos)

| model | inputs | ROC-AUC | PR-AUC | note |
|---|---|---|---|---|
| xgb_pca50 | PCA-50(FP)⊕PCA-50(disease) | 0.757 | 0.508 | embed_swap control |
| xgb_di_md / xgb_full | ECFP4+MACCS + disease | 0.736 | 0.463 | strong tuned baseline |
| xgb_hint_emb | HINT's 100-d embedding | 0.716 | 0.488 | classifier swap |
| hint_di_2019 | MPNN + GRAM → GCN | 0.595–0.629 | 0.328 | |
| chemap_di_2019 | SMILES (approval transfer) | 0.462 | 0.218 | below chance |

---

## ChemAP retraining feasibility (scoped, not run)

If we want ChemAP's *architecture* trained on di (molecule-level "ever-succeeded" label, 33% pos,
~2,500 train molecules):
- **Teacher** needs DrugApp's 186 clinical/patent/property features we don't have → train students
  **KD-free** (drops teacher).
- **ECFP student** — 9 MB (~2.3M params), SMILES-only, **cheap** (minutes on CPU).
- **SMILES student** — 204 MB (~50M params), needs the **ChemBERT** download; **GPU strongly
  advised** (~1–3 h GPU; ~half-day–days on CPU).
- Not black-box: needs data prep, training-script adaptation (Dataset assumes 186-col layout), and
  a training entry script.
- **Expectation:** with ~2,500 molecules and no KD/ChemBERT regularization, beating the existing
  `xgb_di_md` (0.736) is a high bar. Cheap first cut = **ECFP-student-only, KD-free**.

---

## Open questions / threads to pull

- **Multi-seed everything.** Single-seed benchmark unseen strata (86–149) are noisy; pool unseen
  across p1/p2/p3 and run ~3 HINT seeds (Tier 1 in `HINT_VS_XGBOOST_ANALYSIS.md`) to firm up the
  swap conclusions — especially p1's below-chance inversions.
- **HINT head ablation.** `xgb_hint_emb >> hint` says the GCN head hurts. Try a plain concat→MLP
  head on the same 50+50 embeddings (Tier 3) to confirm the interaction machinery is the cost.
- **Why does `xgb_pca50` ≥ `xgb_hint_emb`?** A linear projection beats HINT's end-to-end rep —
  is HINT's bottleneck under-trained (data-starved, ~1k–11k examples, 5 epochs) or genuinely worse?
- **ChemAP, the per-molecule eval.** Dedupe di to molecules + "ever-succeeded" and re-score from
  the existing `runs/chemap_di_2019/predictions.parquet` — removes the one-molecule-many-indications
  confound. *Prediction:* nudges toward ~0.5, won't reach the paper's level (task/population gap).
- **Is approvability orthogonal to indication success?** ChemAP's mild *anti*-correlation is
  interesting — does molecule-level approvability actively mislead for per-indication outcomes, and
  could a *negated* or residualized ChemAP feature add signal to the xgb baseline?
- **ChemAP as a feature, not a model.** Skipped for now: dump ChemAP's penultimate embedding and
  fold it into `xgb_di_md` (embed_swap-style) — does structure-only approvability add anything on
  top of fingerprints?
- **The real bar is `xgb_di_md` (0.736).** Nothing structure/representation-based has beaten the
  tuned fingerprint XGB on di except its own PCA control. What *would*?
