"""Model adapters: canonical example parquet in -> canonical predictions parquet out.

Every adapter has the same signature

    run(dataset_path, features, out_path, **opts) -> Path

and writes `out_path` with columns example_id, label, phase, y_proba (test rows
only). `sklearn` runs xgb/logreg in-process; `hint` shells into the hint/ venv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import chemap_adapter, hint_adapter, sklearn_adapter

# model family -> runner
ADAPTERS: dict[str, Callable[..., Path]] = {
    "xgb": sklearn_adapter.run,
    "logreg": sklearn_adapter.run,
    "hint": hint_adapter.run,
    "chemap": chemap_adapter.run,
}


def run_model(model: str, dataset_path, features, out_path, **opts) -> Path:
    if model not in ADAPTERS:
        raise KeyError(f"unknown model {model!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[model](
        dataset_path=Path(dataset_path) if dataset_path is not None else None,
        features=list(features),
        out_path=Path(out_path),
        model=model,
        **opts,
    )
