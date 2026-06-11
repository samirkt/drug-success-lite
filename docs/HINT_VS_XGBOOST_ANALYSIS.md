# Why is HINT worse than a basic XGBoost — even on HINT's own benchmark?

Analysis / experiment-design note. The premise "roughly equivalent features" is the
crux: the two models share the same raw **inputs** but produce very different
**representations** and carry different **inductive biases**. This doc separates
those, reframes the result with the seen/unseen evidence we already have, and lays
out experiments to validate the cause.

---

## 1. The inputs are the same; the features are not

Both models receive the same raw data — a list of SMILES and a list of ICD codes.
What they turn it into differs sharply:

| | XGBoost (sklearn adapter) | HINT |
|---|---|---|
| molecule | ECFP4(2048)+MACCS(167) = **2215 explicit binary bits** | **MPNN → 50-d learned** embedding, pretrained on ADMET |
| disease | ICD multi-hot, **top-200 explicit indicators** | **GRAM → 50-d learned** embedding over the ICD-10 hierarchy (vocab frozen) |
| criteria | none | Protocol_Embedding (blanked = zeros in the comparison runs) |
| classifier | 1000 gradient-boosted trees, early stopping | GCN "interaction graph" + highway head, ~5 epochs, ~1–1.8k trials |

So the molecule representation differs by **~40× in dimensionality** (2215 vs 50)
and, more importantly, in *kind*:
- XGBoost gets an **explicit, high-dimensional, near-unique fingerprint** that trees
  split on directly — and that effectively lets it **identify specific molecules**.
- HINT gets a **50-d learned bottleneck** that must be trained end-to-end on ~1000
  examples in ~5 CPU epochs.

That is not a fair-features comparison; it's a representation + inductive-bias
comparison. Two well-known facts make the outcome unsurprising:
1. **ECFP + trees is a famously strong molecular baseline** — high-dimensional
   fingerprints let a tree ensemble carve substructure→outcome associations (and
   memorize molecule identity). It routinely beats small GNNs on small datasets.
2. **Deep GNN/GCN models are data-hungry.** HINT has many parameters (MPNN + GRAM +
   interaction GCN + highway) and only ~1k training trials for ~5 epochs — almost
   certainly underfit / high-variance, consistent with the run-to-run swings seen.

---

## 2. The empirical reframe (evidence we already have)

The seen/unseen drug stratification (`dsm stratify`) says the "XGBoost wins on the
benchmark" story is really **"XGBoost wins on drugs it has already seen."**

**HINT benchmark, Phase I, by stratum (ROC-AUC):**

| stratum | XGBoost | HINT (reduced) | HINT (repro, all feats) |
|---|---|---|---|
| **seen** drugs (n=541) | **0.574** | 0.549 | 0.567 |
| **unseen** drugs (n=86) | **0.415** | **0.527** | **0.607** |

On *unseen* molecules XGBoost falls **below chance (0.415)**, while HINT holds
~0.53–0.61. XGBoost's headline benchmark edge lives entirely in the 541 seen rows —
its 2215-bit fingerprint is **memorizing molecule identity**, which HINT's 50-d MPNN
structurally cannot do. This mirrors the single-group ablation on our own data
(`molecule` had the largest seen−unseen gap; `admet`, not the fingerprint,
generalized best to unseen drugs).

**Likely real answer:** on *novel* drugs HINT mostly isn't worse — and is often
better. XGBoost's advantage is largely a **memorization capacity** that the
benchmark's seen-drug overlap rewards.

**Caveat:** n=86 unseen, one phase, one seed. This is the thing to validate, not yet
conclude.

---

## 3. Experiment design to validate (cheapest → decisive)

### Tier 1 — Confirm the memorization reframe (near-free; existing tooling)
- Run `xgb_bench` and `hint_bench` for **all three phases × ~3 HINT seeds**, then
  `dsm stratify` over them. Pool the unseen strata (P1+P2+P3 → a few hundred unseen
  rows, not 86). Also try the stricter exact-combination seen-rule.
- **Predicted if memorization is the cause:** XGBoost ≈ HINT (or worse) on pooled
  *unseen* drugs; XGBoost's whole edge sits in *seen*.

### Tier 2 — The decisive representation-vs-classifier swap
- Dump HINT's learned **50-d MPNN + 50-d GRAM** vectors (the pre-classifier
  representation) for train/test and **train XGBoost on those 100-d HINT
  embeddings.** Holds the representation fixed (HINT's), swaps only the classifier.
  - XGB-on-HINT-embeddings ≈ HINT → the gap is the **representation/bottleneck**,
    not the model head ("features ARE the difference").
  - XGB-on-HINT-embeddings ≫ HINT → the gap is HINT's **classifier/optimization**
    (interaction-GCN head, ~5 epochs, best-model-not-restored), not the features.
- Symmetric control: train XGBoost on a **50-d PCA / random projection of the
  fingerprint.** If it collapses toward HINT's level, that isolates how much the
  50-d bottleneck alone costs.

### Tier 3 — Pin down HINT's own deficit
- **Dimensionality sweep** for XGBoost: ECFP nBits ∈ {256, 512, 1024, 2048},
  ICD top-K ∈ {50, 200, all}. Degrading toward HINT as bits shrink → representation
  capacity is the driver.
- **Learning curve:** train both on 25/50/100% of benchmark train. Gap shrinking
  with data → HINT is data-starved.
- **GRAM vocabulary coverage audit:** fraction of benchmark test ICD codes absent
  from HINT's frozen `icdcode2ancestor_dict` and silently dropped. High → HINT's
  disease channel is crippled vs XGBoost's explicit multi-hot.
- **HINT head ablation:** bypass the interaction-GCN with a plain concat→linear on
  the same 50+50 embeddings; if it helps, the interaction machinery hurts on this
  small data.

---

## 4. What to run first

Run **Tier 1** (just more `dsm run` + `dsm stratify` across phases/seeds) to confirm
the unseen-drug reframe holds beyond n=86. If it does, the answer to the original
question is essentially:

> The features aren't comparable — XGBoost's fingerprint memorizes molecules; on
> truly novel drugs the two models are comparable and HINT is often better.

Then **Tier 2** (XGBoost on HINT's own embeddings) is the single clean experiment
that decisively attributes the remaining gap to **representation** vs **classifier**.

---

### Hypotheses being tested
- **H1 representation capacity** — 2215-d explicit fingerprint vs 50-d learned
  bottleneck (Tier 2 control, Tier 3 sweep).
- **H2 data starvation** — deep model on ~1k examples (Tier 3 learning curve).
- **H3 memorization** — fingerprint identifies molecules; benchmark overlap rewards
  it (Tier 1, Tier 2 swap).
- **H4 disease channel** — GRAM vocab freeze vs explicit ICD multi-hot (Tier 3 audit).
- **H5 head/optimization** — interaction-GCN, epochs, best-model-not-restored,
  single seed (Tier 2 swap, Tier 3 head ablation).
