"""`python -m dsm` — the single entry point for running experiments.

    python -m dsm list                       # show datasets + experiments
    python -m dsm materialize ours_di        # build a canonical example parquet
    python -m dsm run xgb_di_2019            # run one experiment -> runs/<name>/metrics.json
    python -m dsm run hint_bench_p1_repro --epochs 5
    python -m dsm run --all                  # run every experiment

Every experiment writes runs/<name>/{predictions.parquet, metrics.json} in one
standardized format, whether the model is xgb/logreg (in-process) or HINT (shelled
into hint/'s venv). To add an experiment, edit `dsm/experiments.py` — no new code.
"""

from __future__ import annotations

import argparse
import logging

from .experiments import DATASETS, EXPERIMENTS
from .run import collect_results, materialize_dataset, run_experiment

logger = logging.getLogger(__name__)


def cmd_list(args) -> None:
    print("datasets:")
    for name, spec in DATASETS.items():
        extra = spec.phase_stem or f"{spec.granularity} <= {spec.time_split_year}"
        print(f"  {name:16s} {spec.kind:14s} {extra}")
    print("\nexperiments:")
    for name, spec in EXPERIMENTS.items():
        src = spec.dataset or f"native:{spec.native_benchmark}"
        print(f"  {name:22s} {spec.model:7s} {src:16s} features={','.join(spec.features)}")


def cmd_materialize(args) -> None:
    path = materialize_dataset(args.dataset, force=args.force)
    print(f"materialized {args.dataset} -> {path}")


def cmd_run(args) -> None:
    names = list(EXPERIMENTS) if args.all else [args.experiment]
    if not args.all and args.experiment is None:
        raise SystemExit("give an experiment name or --all (see `dsm list`)")
    for name in names:
        payload = run_experiment(
            name, epochs=args.epochs, bootstrap_ci=args.bootstrap_ci,
            force_materialize=args.force_materialize,
        )
        o = payload["overall"]
        print(f"{name:22s} ROC-AUC={o['roc_auc']:.4f} PR-AUC={o['pr_auc']:.4f} "
              f"F1={o['f1']:.4f} (n={payload['n']})")


def cmd_results(args) -> None:
    rows = collect_results()
    if not rows:
        print("no results yet — run an experiment first (see `dsm list`).")
        return
    cols = ["experiment", "model", "dataset", "n", "n_pos", "roc_auc", "pr_auc", "f1"]
    fmt = []
    for r in rows:
        fr = {}
        for c in cols:
            v = r.get(c)
            fr[c] = (f"{v:.4f}" if isinstance(v, float) and v == v
                     else ("" if v is None else str(v)))
        fmt.append(fr)
    w = {c: max(len(c), *(len(f[c]) for f in fmt)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    print("  ".join("-" * w[c] for c in cols))
    for f in fmt:
        print("  ".join(f[c].ljust(w[c]) for c in cols))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dsm", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list datasets and experiments").set_defaults(func=cmd_list)
    sub.add_parser("results", help="table of all runs/*/metrics.json").set_defaults(func=cmd_results)

    m = sub.add_parser("materialize", help="build a canonical example parquet")
    m.add_argument("dataset", help="dataset name (see `dsm list`)")
    m.add_argument("--force", action="store_true", help="rebuild even if it exists")
    m.set_defaults(func=cmd_materialize)

    r = sub.add_parser("run", help="run an experiment (or --all)")
    r.add_argument("experiment", nargs="?", help="experiment name (see `dsm list`)")
    r.add_argument("--all", action="store_true", help="run every experiment")
    r.add_argument("--epochs", type=int, default=None, help="override HINT epochs")
    r.add_argument("--bootstrap-ci", type=int, default=0,
                   help="bootstrap resamples for 95%% CIs (0 = off)")
    r.add_argument("--force-materialize", action="store_true",
                   help="rebuild the dataset parquet before running")
    r.set_defaults(func=cmd_run)
    return p


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
