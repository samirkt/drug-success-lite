"""`python -m dsm train ...` — train models and output metrics.

Unlike the original repo, `--groups` is honored: by default one model is trained
on exactly the groups you pass (all five if omitted). `--sweep` reproduces the
old group-set sweep (one model per set).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import ALL_FEATURE_GROUPS, FeatureConfig, ModelingConfig
from .train import RunResult, train_one_run, write_run

logger = logging.getLogger(__name__)

# Reproduces the original repo's hard-coded sweep (one model per set).
_SWEEP_GROUP_SETS: tuple[tuple[str, ...], ...] = (
    ("molecule",),
    ("disease",),
    ("admet",),
    ("target",),
    ("pathway",),
    ("molecule", "disease", "admet"),
    ("molecule", "disease", "admet", "target", "pathway"),
)

# Granularity → the date column to split on.
_TIME_COLUMN = {
    "drug_indication": "earliest_start_date",
    "trial": "trial_start_date",
}

_SUMMARY_FIELDS = ("n", "n_pos", "roc_auc", "pr_auc", "f1", "brier", "balanced_accuracy")
_CI_FIELDS = ("roc_auc_lo", "roc_auc_hi", "pr_auc_lo", "pr_auc_hi", "f1_lo", "f1_hi")


def _summary_rows(result: RunResult) -> list[dict]:
    """One row for the overall test set, plus one per phase for trial runs."""
    groups = "+".join(result.groups)
    gran = result.config.training_granularity

    def row(level: str, m: dict) -> dict:
        d = {"granularity": gran, "level": level, "groups": groups,
             "model": result.config.model_name, "n_features": result.n_features}
        for f in _SUMMARY_FIELDS:
            d[f] = m.get(f)
        for f in _CI_FIELDS:
            if f in m:
                d[f] = m[f]
        return d

    rows = [row(gran, result.metrics)]
    for label, mp in result.per_phase_metrics.items():
        rows.append(row(f"trial:{label}", mp))
    return rows


def _print_table(rows: list[dict]) -> None:
    cols = ["granularity", "level", "groups", "n", "n_pos", "roc_auc", "pr_auc", "f1"]
    widths = {c: len(c) for c in cols}
    fmt_rows = []
    for r in rows:
        fr = {}
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                fr[c] = "nan" if v != v else f"{v:.4f}"
            else:
                fr[c] = "" if v is None else str(v)
            widths[c] = max(widths[c], len(fr[c]))
        fmt_rows.append(fr)
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    print(line)
    print("  ".join("-" * widths[c] for c in cols))
    for fr in fmt_rows:
        print("  ".join(fr[c].ljust(widths[c]) for c in cols))


def _build_config(args, granularity: str, groups: tuple[str, ...]) -> ModelingConfig:
    return ModelingConfig(
        features=FeatureConfig(enabled=tuple(groups), molecule_repr=args.molecule_repr),
        training_granularity=granularity,
        model_name=args.model,
        test_size=args.test_size,
        seed=args.seed,
        time_split_column=_TIME_COLUMN[granularity],
        time_split_year=args.time_split_year,
        bootstrap_ci=args.bootstrap_ci,
    )


def cmd_train(args) -> None:
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    granularities = (
        ["drug_indication", "trial"] if args.granularity == "both" else [args.granularity]
    )
    group_sets = _SWEEP_GROUP_SETS if args.sweep else [tuple(args.groups)]

    all_rows: list[dict] = []
    for gran in granularities:
        for groups in group_sets:
            tag = f"{gran}__{'+'.join(groups)}" if args.sweep else gran
            logger.info("=== training %s ===", tag)
            config = _build_config(args, gran, groups)
            result = train_one_run(config)
            write_run(result, output_root / tag)
            all_rows.extend(_summary_rows(result))

    summary = pd.DataFrame(all_rows)
    summary_path = output_root / "metrics.csv"
    summary.to_csv(summary_path, index=False)
    print()
    _print_table(all_rows)
    print(f"\nwrote summary -> {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dsm", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train model(s) and output metrics")
    t.add_argument(
        "--granularity", choices=["drug_indication", "trial", "both"],
        default="drug_indication",
    )
    t.add_argument(
        "--groups", nargs="+", default=list(ALL_FEATURE_GROUPS),
        metavar="GROUP", help="feature groups to use (ignored with --sweep)",
    )
    t.add_argument("--molecule-repr", choices=["fingerprint", "embedding"], default="fingerprint")
    t.add_argument("--model", choices=["xgb", "logreg"], default="xgb")
    t.add_argument("--time-split-year", type=int, default=None,
                   help="temporal split cutoff; omit for a stratified random split")
    t.add_argument("--test-size", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--bootstrap-ci", type=int, default=0,
                   help="bootstrap resamples for 95%% CIs on ROC-AUC/PR-AUC/F1 (0 = off)")
    t.add_argument("--sweep", action="store_true",
                   help="train one model per built-in group set instead of a single run")
    t.add_argument("--output", default="runs/latest")
    t.set_defaults(func=cmd_train)
    return p


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
