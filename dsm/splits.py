"""Train/test split — temporal or stratified-random.

`time_split_year` set → temporal split on `time_split_column`'s year (train
<= year < test). Otherwise → stratified shuffle split by `y`.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _year_of(v) -> Optional[int]:
    """Extract year from a date / datetime / pd.Timestamp / None."""
    if v is None:
        return None
    if hasattr(v, "year"):
        return int(v.year)
    return None


def split(
    df: pd.DataFrame,
    *,
    test_size: float,
    seed: int,
    time_split_column: Optional[str] = None,
    time_split_year: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) into df."""
    y = df["y"].values

    if time_split_year is not None:
        if not time_split_column or time_split_column not in df.columns:
            raise ValueError(f"time_split_column={time_split_column!r} not in DataFrame")
        years = df[time_split_column].apply(_year_of).values
        train_mask = np.array([yr is not None and yr <= time_split_year for yr in years])
        test_mask = np.array([yr is not None and yr > time_split_year for yr in years])
        n_skip = len(df) - train_mask.sum() - test_mask.sum()
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)
        train_y = y[train_idx]
        test_y = y[test_idx]
        logger.info(
            "split: temporal on %s; cutoff=%d; train=%d (pos=%d, %.1f%%) test=%d (pos=%d, %.1f%%) skipped_no_year=%d",
            time_split_column,
            time_split_year,
            len(train_idx),
            int(train_y.sum()),
            100.0 * train_y.mean() if len(train_y) else 0.0,
            len(test_idx),
            int(test_y.sum()),
            100.0 * test_y.mean() if len(test_y) else 0.0,
            n_skip,
        )
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(
                f"temporal split with cutoff={time_split_year} produced empty "
                f"train ({len(train_idx)}) or test ({len(test_idx)}) — "
                f"check the year distribution of {time_split_column}"
            )
        return train_idx, test_idx

    from sklearn.model_selection import train_test_split

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=seed,
    )
    logger.info(
        "split: stratified row-level; train=%d test=%d (test_size=%.2f, seed=%d)",
        len(train_idx),
        len(test_idx),
        test_size,
        seed,
    )
    return train_idx, test_idx
