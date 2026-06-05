"""Generic, column-driven sub-encoders.

A composite feature group (see `features.py`) is just an ordered list of these
encoders. Each encoder knows which column(s) it reads, learns its state on the
inner-train slice via `fit`, and emits a fixed-width float matrix via
`transform`. All encoders tolerate their source column being absent on the
frame: they fall back to a deterministic zero/median width so the assembled
matrix stays stable across train/val/test slices.

Shared contract (duck-typed):

    is_available(df) -> bool
    fit(df) -> None
    transform(df) -> np.ndarray            # shape (len(df), width)
    feature_names() -> list[str]           # len == width
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from ._admet_columns import ADMET_COLUMNS, field_name

logger = logging.getLogger(__name__)


def stack_arrays(values: pd.Series, dim: int, dtype) -> tuple[np.ndarray, np.ndarray]:
    """Stack a pandas Series of numpy arrays into a 2D matrix.

    Rows with NaN / None are filled with zeros. Returns (matrix, missing_mask).
    """
    n = len(values)
    out = np.zeros((n, dim), dtype=dtype)
    missing = np.zeros(n, dtype=np.int8)
    for i, v in enumerate(values.values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            missing[i] = 1
            continue
        arr = np.asarray(v)
        if arr.shape[0] != dim:
            logger.warning(
                "stack_arrays: row %d has shape %s, expected (%d,) — treating as missing",
                i, arr.shape, dim,
            )
            missing[i] = 1
            continue
        out[i] = arr.astype(dtype, copy=False)
    return out, missing


def _iter_lists(values: pd.Series) -> list[list[str]]:
    """Yield a clean list[str] for each row, treating NaN/None as []."""
    out: list[list[str]] = []
    for v in values.values:
        if v is None:
            out.append([])
            continue
        if isinstance(v, float) and np.isnan(v):
            out.append([])
            continue
        try:
            out.append([str(x) for x in v if x is not None])
        except TypeError:
            out.append([])
    return out


class MultiHot:
    """Multi-hot encoder over the K most frequent tokens in TRAIN.

    Out-of-vocabulary tokens collapse into a `<prefix>_other_count` column;
    empty/missing rows are flagged by `<prefix>_missing`. `token_fn` optionally
    rewrites each row's raw token list before counting (e.g. MeSH tree-prefix
    truncation).
    """

    def __init__(
        self,
        column: str,
        prefix: str,
        top_k: int,
        token_fn: Optional[Callable[[list[str]], list[str]]] = None,
    ) -> None:
        self.column = column
        self.prefix = prefix
        self.top_k = top_k
        self._token_fn = token_fn
        self.vocab: list[str] = []
        self._index: dict[str, int] = {}

    def _tokens(self, df: pd.DataFrame) -> list[list[str]]:
        if self.column not in df.columns:
            return [[] for _ in range(len(df))]
        rows = _iter_lists(df[self.column])
        if self._token_fn is not None:
            rows = [self._token_fn(r) for r in rows]
        return rows

    def is_available(self, df: pd.DataFrame) -> bool:
        return self.column in df.columns

    def fit(self, df: pd.DataFrame) -> None:
        counts: Counter[str] = Counter()
        for tokens in self._tokens(df):
            counts.update(set(tokens))  # dedup within row
        self.vocab = [t for t, _ in counts.most_common(self.top_k)]
        self._index = {t: i for i, t in enumerate(self.vocab)}
        logger.info(
            "%s: fit on %d rows; vocab=%d (top_k=%d)",
            self.prefix, len(df), len(self.vocab), self.top_k,
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        rows = self._tokens(df)
        k = len(self.vocab)
        out = np.zeros((len(rows), k + 2), dtype=np.float32)
        for i, tokens in enumerate(rows):
            if not tokens:
                out[i, k + 1] = 1.0  # _missing
                continue
            other = 0
            for tok in set(tokens):
                idx = self._index.get(tok)
                if idx is None:
                    other += 1
                else:
                    out[i, idx] = 1.0
            out[i, k] = float(other)  # _other_count
        return out

    def feature_names(self) -> list[str]:
        return (
            [f"{self.prefix}_{tok}" for tok in self.vocab]
            + [f"{self.prefix}_other_count", f"{self.prefix}_missing"]
        )


class WeightedMap:
    """Score-valued multi-hot over a `list<struct<pathway_id, score>>` column.

    Vocabulary is the top-K most frequent `pathway_id`s in TRAIN. Each cell
    holds the row's `score` for that id (0.0 if absent); a `<prefix>_missing`
    column flags rows with an empty map.
    """

    def __init__(self, column: str, prefix: str, top_k: int) -> None:
        self.column = column
        self.prefix = prefix
        self.top_k = top_k
        self.vocab: list[str] = []
        self._index: dict[str, int] = {}

    def _pairs(self, df: pd.DataFrame) -> list[list[tuple[str, float]]]:
        if self.column not in df.columns:
            return [[] for _ in range(len(df))]
        out: list[list[tuple[str, float]]] = []
        for v in df[self.column].values:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out.append([])
                continue
            try:
                out.append([(str(d["pathway_id"]), float(d["score"])) for d in v])
            except (TypeError, KeyError, ValueError):
                out.append([])
        return out

    def is_available(self, df: pd.DataFrame) -> bool:
        return self.column in df.columns

    def fit(self, df: pd.DataFrame) -> None:
        counts: Counter[str] = Counter()
        for pairs in self._pairs(df):
            counts.update({pid for pid, _ in pairs})
        self.vocab = [t for t, _ in counts.most_common(self.top_k)]
        self._index = {t: i for i, t in enumerate(self.vocab)}
        logger.info(
            "%s: fit on %d rows; vocab=%d pathway IDs (top_k=%d)",
            self.prefix, len(df), len(self.vocab), self.top_k,
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        rows = self._pairs(df)
        k = len(self.vocab)
        out = np.zeros((len(rows), k + 1), dtype=np.float32)
        for i, pairs in enumerate(rows):
            if not pairs:
                out[i, k] = 1.0  # _missing
                continue
            for pid, score in pairs:
                idx = self._index.get(pid)
                if idx is not None:
                    out[i, idx] = np.float32(score)
        return out

    def feature_names(self) -> list[str]:
        return [f"{self.prefix}_{pid}" for pid in self.vocab] + [f"{self.prefix}_missing"]


class Scalar:
    """Numeric columns: median-impute + a paired `<col>_missing` indicator.

    Bools are cast to float. Columns absent at fit time are skipped; columns
    present at fit but absent at transform fall back to the learned median with
    the indicator set, keeping the output width deterministic.
    """

    def __init__(self, columns: Sequence[str]) -> None:
        self._columns = tuple(columns)
        self._kept_cols: list[str] = []
        self._medians: dict[str, float] = {}

    def is_available(self, df: pd.DataFrame) -> bool:
        return any(c in df.columns for c in self._columns)

    def fit(self, df: pd.DataFrame) -> None:
        self._kept_cols = []
        self._medians = {}
        for col in self._columns:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            self._medians[col] = float(series.median()) if series.notna().any() else 0.0
            self._kept_cols.append(col)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        if not self._kept_cols:
            return np.zeros((n, 0), dtype=np.float32)
        out = np.zeros((n, 2 * len(self._kept_cols)), dtype=np.float32)
        for i, col in enumerate(self._kept_cols):
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                out[:, 2 * i] = series.fillna(self._medians[col]).astype(np.float32).values
                out[:, 2 * i + 1] = series.isna().astype(np.float32).values
            else:
                out[:, 2 * i] = np.float32(self._medians[col])
                out[:, 2 * i + 1] = np.float32(1.0)
        return out

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for col in self._kept_cols:
            names.append(col)
            names.append(f"{col}_missing")
        return names


class DenseArray:
    """Fixed-width vector column (embedding / fingerprint bits) + `_missing`.

    `impute="mean"` fills missing rows with the train-set mean (embeddings);
    `impute="zero"` leaves them as zeros (fingerprints).
    """

    def __init__(
        self,
        columns: Sequence[tuple[str, int]],
        prefix: str,
        dtype=np.float32,
        impute: str = "zero",
    ) -> None:
        # columns: list of (column_name, dim); names emitted as <col>_<i>.
        self._columns = list(columns)
        self.prefix = prefix
        self._dtype = dtype
        self._impute = impute
        self._means: dict[str, np.ndarray] = {}

    def is_available(self, df: pd.DataFrame) -> bool:
        return all(c in df.columns for c, _ in self._columns)

    def fit(self, df: pd.DataFrame) -> None:
        self._means = {}
        if self._impute != "mean":
            return
        for col, dim in self._columns:
            if col not in df.columns:
                self._means[col] = np.zeros(dim, dtype=self._dtype)
                continue
            mat, missing = stack_arrays(df[col], dim, self._dtype)
            present = missing == 0
            self._means[col] = (
                mat[present].mean(axis=0).astype(self._dtype)
                if present.any()
                else np.zeros(dim, dtype=self._dtype)
            )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        blocks: list[np.ndarray] = []
        missing_any = np.zeros(n, dtype=np.int8)
        for col, dim in self._columns:
            if col in df.columns:
                mat, missing = stack_arrays(df[col], dim, self._dtype)
            else:
                mat = np.zeros((n, dim), dtype=self._dtype)
                missing = np.ones(n, dtype=np.int8)
            if self._impute == "mean" and missing.any():
                mat[missing == 1] = self._means.get(col, np.zeros(dim, dtype=self._dtype))
            missing_any |= missing
            blocks.append(mat)
        blocks.append(missing_any.reshape(-1, 1).astype(self._dtype))
        return np.hstack(blocks)

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for col, dim in self._columns:
            names.extend(f"{col}_{i}" for i in range(dim))
        names.append(f"{self.prefix}_missing")
        return names


class OneHot:
    """Low-cardinality categorical → one-hot + `<prefix>_missing`.

    Unseen-at-fit values map to the `_missing` column.
    """

    def __init__(self, column: str, prefix: str) -> None:
        self.column = column
        self.prefix = prefix
        self.vocab: list[str] = []

    def is_available(self, df: pd.DataFrame) -> bool:
        return self.column in df.columns

    def fit(self, df: pd.DataFrame) -> None:
        if self.column in df.columns:
            self.vocab = sorted(v for v in df[self.column].dropna().unique())
        else:
            self.vocab = []

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        if not self.vocab:
            return np.zeros((n, 0), dtype=np.float32)
        idx = {v: i for i, v in enumerate(self.vocab)}
        out = np.zeros((n, len(self.vocab) + 1), dtype=np.float32)
        col = df[self.column].values if self.column in df.columns else [None] * n
        for i, v in enumerate(col):
            j = idx.get(v) if not (v is None or (isinstance(v, float) and np.isnan(v))) else None
            out[i, j if j is not None else -1] = 1.0
        return out

    def feature_names(self) -> list[str]:
        if not self.vocab:
            return []
        return [f"{self.prefix}_{v}" for v in self.vocab] + [f"{self.prefix}_missing"]


def _percentile_columns() -> list[str]:
    """DataFrame column names for the DrugBank-approved-percentile ADMET features."""
    return [
        field_name(c)
        for c in ADMET_COLUMNS
        if c.endswith("_drugbank_approved_percentile")
    ]


class AdmetPercentiles:
    """ADMET DrugBank-approved-percentile columns.

    Drops columns exceeding `drop_null_threshold` nulls in TRAIN, median-imputes
    the rest, and emits a `<col>_missing` indicator only for columns whose null
    rate exceeds `indicator_threshold`.
    """

    def __init__(
        self,
        drop_null_threshold: float = 0.95,
        indicator_threshold: float = 0.05,
    ) -> None:
        self._drop_null_threshold = drop_null_threshold
        self._indicator_threshold = indicator_threshold
        self._kept_cols: list[str] = []
        self._indicator_cols: list[str] = []
        self._medians: dict[str, float] = {}
        self._all_candidate_cols: list[str] = _percentile_columns()

    def is_available(self, df: pd.DataFrame) -> bool:
        return any(c in df.columns for c in self._all_candidate_cols)

    def fit(self, df: pd.DataFrame) -> None:
        present = [c for c in self._all_candidate_cols if c in df.columns]
        n = len(df)
        self._kept_cols = []
        self._indicator_cols = []
        self._medians = {}
        for col in present:
            null_rate = df[col].isna().mean() if n else 1.0
            if null_rate > self._drop_null_threshold:
                continue
            self._kept_cols.append(col)
            self._medians[col] = float(df[col].median()) if df[col].notna().any() else 0.0
            if null_rate > self._indicator_threshold:
                self._indicator_cols.append(col)
        logger.info(
            "admet: fit on %d rows; %d/%d cols kept, %d indicators",
            n, len(self._kept_cols), len(present), len(self._indicator_cols),
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        if not self._kept_cols:
            return np.zeros((n, 0), dtype=np.float32)
        n_val = len(self._kept_cols)
        out = np.zeros((n, n_val + len(self._indicator_cols)), dtype=np.float32)
        for i, col in enumerate(self._kept_cols):
            series = df[col] if col in df.columns else pd.Series([np.nan] * n)
            out[:, i] = series.fillna(self._medians[col]).astype(np.float32).values
        for j, col in enumerate(self._indicator_cols):
            series = df[col] if col in df.columns else pd.Series([np.nan] * n)
            out[:, n_val + j] = series.isna().astype(np.float32).values
        return out

    def feature_names(self) -> list[str]:
        return list(self._kept_cols) + [f"{c}_missing" for c in self._indicator_cols]
