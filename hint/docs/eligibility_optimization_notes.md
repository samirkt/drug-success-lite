# Optimizing eligibility criteria — conceptual notes

Notes on whether/how eligibility criteria can be treated as an *optimization* problem,
especially in relation to the criteria-blind, indication-level advancement model we set up
(see `HINT/learn_advancement.py`).

## The core knot: "input feature" vs "decision variable"

A variable can play two very different roles:

- **As an input feature**, eligibility is *observed data*. It's given, and the model uses it to
  predict an outcome. The model treats it as fixed.
- **As a decision variable in an optimization**, eligibility is something you *choose*. The
  question flips from "given these criteria, what happens?" to "what criteria *should* I write to
  make the best thing happen?"

Optimization is fundamentally about the second role, and this is the key point:

> **To optimize over a variable, your objective function must be *sensitive* to that variable.**

If the advancement model is blind to eligibility criteria, then changing the criteria changes the
predicted outcome by exactly zero. The gradient (how much the objective moves when you nudge the
criteria) is flat everywhere. An optimizer handed a flat objective has no signal — every choice
looks equally good, so there is nothing to optimize.

**Blunt answer:** you cannot meaningfully optimize eligibility while it is excluded from the model.
Optimizing it requires *some* model whose output responds to it. That model can be different from
the criteria-blind predictor, but it has to exist.

## Anatomy of an optimization problem

Every optimization problem has three parts:

1. **Decision variables** — what you get to choose. *Here: the eligibility criteria.*
2. **Objective function** — what you maximize/minimize. *Here: e.g. P(advancement), or a utility
   trading off advancement vs. enrollment speed, cost, patient safety.*
3. **Constraints** — what counts as a feasible/legal choice. *Here: criteria must be clinically
   valid, ethical, and enroll enough patients to power the trial.*

The predictive model lives **inside the objective function** — it maps a candidate set of criteria
to a predicted score. That is why the model must take criteria as input: it *is* the function the
optimizer climbs.

## Framings that make sense

- **Prescriptive / design optimization** (the main one): search the space of inclusion/exclusion
  rules to maximize an objective. Needs a criteria-aware model.
- **Eligibility as a constraint, not an objective**: criteria define *which patients are in the
  trial*. Optimize other design choices (sample size, endpoints, sites) *subject to* a fixed
  eligibility set — eligibility shapes the feasible region rather than being maximized.
- **Counterfactual / sensitivity analysis**: with a criteria-aware model, ask "if I loosen this one
  exclusion, how does predicted advancement move?" Optimization-flavored, and often the most
  practical starting point.

## The hard part: the decision space

Eligibility criteria are an ugly space to optimize over — discrete, structured, combinatorial
(sets of rules, some with thresholds: "age 18-65", "eGFR > 30", "no prior anti-TNF therapy"). As
free text the space is effectively infinite and non-differentiable, so you can't just take
gradients. Practical approaches:

- **Parameterize** criteria into a structured form (a fixed list of clinical variables with tunable
  thresholds) → decision variables become continuous/integer numbers you can search.
- **Black-box / derivative-free search** (Bayesian optimization, evolutionary algorithms) over a
  generative model of criteria, when you can't differentiate.
- **Counterfactual loops** — perturb one criterion at a time and re-score.

## How this reconciles with the criteria-blind model

The blind model and an optimizer can compose cleanly rather than conflict. Think of it as a
decomposition:

- The **criteria-blind advancement model** estimates a **prior**: "how likely is this
  drug-indication to advance, regardless of how the trial is run?"
- A separate **criteria-aware** model captures the **delta** — the part of the outcome attributable
  to trial design choices. You optimize over *that* piece.

So the blind model is the baseline; the design effect is what you actually have leverage over:

```
P(advance)  ≈  f(drug, indication)        # criteria-blind prior (current model)
             +  g(criteria | drug, indication)   # design effect — the thing you optimize
```

## Two caveats that matter a lot

- **Goodhart / gaming the model.** Optimizing criteria to maximize a *model's predicted*
  advancement will find criteria that fool the model rather than genuinely help. The harder you
  optimize, the more you exploit the model's errors — optimizing the map, not the territory.
- **Correlation vs. causation.** The model learns from *observational* data (trials as actually
  run). It captures that strict criteria *correlate* with outcomes, not the *causal effect* of
  tightening/loosening a rule. Optimization is an intervention, so it needs a causal estimate.
  Naive optimization over an observational model can point confidently in the wrong direction due
  to confounding (e.g. strict criteria may just correlate with experienced sponsors who succeed for
  unrelated reasons).

## Real-world grounding

This is an active research area. The cleanest example is **Trial Pathfinder** (Liu et al.,
*Nature*, 2021): using real-world EHR data, they simulated *relaxing* overly restrictive oncology
eligibility criteria and showed many trials could substantially expand their eligible population
without worsening the hazard ratio. That is essentially eligibility optimization — objective =
(eligible population size, outcome), decision = which criteria to loosen — evaluated against real
outcomes rather than a black-box predictor, which sidesteps both caveats above.

## Takeaway

Optimizing eligibility makes sense, but it is a **prescriptive** problem that needs a
criteria-aware objective (ideally a causal one), and it is a distinct project from the criteria-blind
predictor. The two fit together as **prior (blind baseline) + design-effect (the thing you
optimize)**.

Possible next step: a minimal criteria-aware "delta" model plus a counterfactual scoring loop on
top of the current repo.
