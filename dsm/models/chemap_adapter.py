"""ChemAP adapter — dispatch into the chemap/ venv via subprocess.

Same isolation pattern as hint_adapter: dsm and ChemAP live in separate venvs, and the only
thing crossing the boundary is the canonical parquet. This shells
`uv run --project chemap python run_experiment.py ...` with cwd=chemap/ (so ChemAP's
cwd-relative `model/` and `src/` resolve) and returns the canonical predictions parquet it wrote.

ChemAP is a pretrained, SMILES-only black box: it has no training/native-benchmark path, so it
needs `dataset_path` (canonical). `features`/`epochs`/`lr`/`native_benchmark` are accepted for
adapter-signature parity and ignored.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CHEMAP_DIR = Path(__file__).resolve().parents[2] / "chemap"


def run(*, dataset_path: Optional[Path], out_path: Path, seed: int = 7,
        model: str = "chemap", **_ignored) -> Path:
    if dataset_path is None:
        raise ValueError("chemap adapter needs dataset_path (canonical); it has no native path")
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv", "run", "--project", str(CHEMAP_DIR),
        "python", "run_experiment.py",
        "--dataset", str(Path(dataset_path).resolve()),
        "--out", str(out_path),
        "--seed", str(seed),
    ]

    # Drop the parent (dsm) venv from the env so uv targets chemap/'s own venv cleanly.
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    logger.info("chemap: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(CHEMAP_DIR), check=True, env=env)
    if not out_path.exists():
        raise RuntimeError(f"chemap run_experiment.py did not write {out_path}")
    return out_path
