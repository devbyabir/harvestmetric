"""Leakage-safe splitting: whole farms go to either train or test, never both."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config


class LeakageError(AssertionError):
    """Raised when a train/test split shares farms or identical records."""


def group_holdout_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    group_col: str = config.GROUP_COL,
    stratify_col: str = "crop_type",
    random_state: int = config.RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) positional indices.

    Uses StratifiedGroupKFold and keeps one fold as the untouched test set, so
    every row of a farm (including exact duplicates) lands on the same side and
    crop proportions stay roughly balanced.
    """
    n_splits = round(1 / test_fraction)
    if n_splits < 2:
        raise ValueError("test_fraction must be <= 0.5")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, df[stratify_col], groups=df[group_col]))
    assert_no_leakage(df.iloc[train_idx], df.iloc[test_idx], group_col)
    return train_idx, test_idx


def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame, group_col: str = config.GROUP_COL) -> None:
    shared_groups = set(train[group_col]) & set(test[group_col])
    if shared_groups:
        raise LeakageError(f"{len(shared_groups)} {group_col} values in both train and test")
    content_cols = [c for c in train.columns if c not in config.ID_COLS]
    merged = train[content_cols].drop_duplicates().merge(test[content_cols].drop_duplicates(), how="inner")
    if len(merged):
        raise LeakageError(f"{len(merged)} identical records (ignoring IDs) in both train and test")
