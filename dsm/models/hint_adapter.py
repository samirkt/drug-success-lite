"""HINT adapter — dispatch into the hint/ venv via subprocess.

dsm and HINT live in incompatible Python environments (numpy>=2 vs numpy<2), so
the only thing that crosses the boundary is the canonical parquet. This adapter
shells `uv run --project hint python run_experiment.py ...` with cwd=hint/ (so
HINT's cwd-relative `data/` assets resolve) and lets that one script do the
canonical->HINT conversion, training, and prediction. It returns the canonical
predictions parquet HINT wrote.

Two modes (mirrored by run_experiment.py):
  - canonical : `--dataset <canonical.parquet>` (our data, or benchmark-canonical)
  - native    : `--native-benchmark phase_I` reads HINT's own phase CSVs verbatim
                (faithful reproduction of the published benchmark numbers).
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HINT_DIR = Path(__file__).resolve().parents[2] / "hint"


def run(*, dataset_path: Optional[Path], features: list[str], out_path: Path,
        native_benchmark: Optional[str] = None, epochs: int = 5, lr: float = 1e-3,
        seed: int = 42, device: str = "cpu", **_ignored) -> Path:
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv", "run", "--project", str(HINT_DIR),
        "python", "run_experiment.py",
        "--features", ",".join(f for f in features if f.lower() != "icd"),
        "--out", str(out_path),
        "--epochs", str(epochs), "--lr", str(lr),
        "--seed", str(seed), "--device", device,
    ]
    if native_benchmark:
        cmd += ["--native-benchmark", native_benchmark]
    else:
        if dataset_path is None:
            raise ValueError("hint adapter needs dataset_path (canonical) or native_benchmark")
        cmd += ["--dataset", str(Path(dataset_path).resolve())]

    # Drop the parent (dsm) venv from the env so uv targets hint/'s own venv cleanly.
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    logger.info("hint: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(HINT_DIR), check=True, env=env)
    if not out_path.exists():
        raise RuntimeError(f"hint run_experiment.py did not write {out_path}")
    return out_path
