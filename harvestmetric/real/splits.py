"""Chronological and geographic splits, with leakage assertions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


class SplitLeakageError(AssertionError):
    """Raised when a split could leak information across its boundary."""


def observed(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel[config.TARGET].notna()]


def time_split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training years FIRST_MODEL_YEAR..TRAIN_END; untouched test years TEST_START..LAST_YIELD_YEAR."""
    obs = observed(panel)
    train = obs[obs.year.between(config.FIRST_MODEL_YEAR, config.TRAIN_END)].reset_index(drop=True)
    test = obs[obs.year.between(config.TEST_START, config.LAST_YIELD_YEAR)].reset_index(drop=True)
    assert_time_order(train, test)
    return train, test


def assert_time_order(train: pd.DataFrame, test: pd.DataFrame) -> None:
    if len(train) and len(test) and train.year.max() >= test.year.min():
        raise SplitLeakageError(f"training year {train.year.max()} is not before test year {test.year.min()}")


def expanding_window_folds(train: pd.DataFrame, blocks=config.CV_BLOCKS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each fold validates on a block of years and trains only on strictly earlier years."""
    folds = []
    for start, end in blocks:
        tr = np.flatnonzero(train.year.to_numpy() < start)
        va = np.flatnonzero(train.year.between(start, end).to_numpy())
        if len(tr) == 0 or len(va) == 0:
            raise ValueError(f"empty fold for block {start}-{end}")
        assert_time_order(train.iloc[tr], train.iloc[va])
        folds.append((tr, va))
    return folds


def district_folds(frame: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Leave-one-agricultural-district-out folds (NASS's 9 Iowa crop reporting districts).

    Whole districts are held out, so a held-out county never has a neighbour
    from the same district in training.
    """
    folds = []
    for code in sorted(frame.district_code.unique()):
        te = np.flatnonzero(frame.district_code.to_numpy() == code)
        tr = np.flatnonzero(frame.district_code.to_numpy() != code)
        if set(frame.fips.iloc[tr]) & set(frame.fips.iloc[te]):
            raise SplitLeakageError("a county appears on both sides of a geographic fold")
        folds.append((frame.district.iloc[te[0]], tr, te))
    return folds
