"""Interactive leakage simulator used by the app (no external data needed).

Generates a dataset in which the target is pure noise (or has a chosen amount
of real signal), duplicates every record, and compares row-level KFold with
farm-grouped KFold. It reproduces the mechanism behind the original R² ≈ 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_score


@dataclass
class SimulationResult:
    row_level_r2: float
    grouped_r2: float
    true_signal_r2: float     # R² an oracle using the true signal would reach
    rows: int
    farms: int


def make_dataset(n_farms: int, copies: int, signal: float, n_features: int = 8,
                 seed: int = 0) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    """signal in [0, 1]: share of target variance explained by feature 0."""
    if not 0 <= signal <= 1:
        raise ValueError("signal must be between 0 and 1")
    if n_farms < 10 or copies < 1:
        raise ValueError("need at least 10 farms and 1 copy")
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n_farms, n_features))
    z = (X[:, 0] - 0.5) / np.sqrt(1 / 12)
    clean = np.sqrt(signal) * z
    y = clean + np.sqrt(1 - signal) * rng.normal(0, 1, n_farms)
    farms = np.arange(n_farms)
    rep = np.tile(farms, copies)
    return (pd.DataFrame(X[rep], columns=[f"x{i}" for i in range(n_features)]),
            pd.Series(y[rep]), rep, clean[rep])


def simulate(n_farms: int = 300, copies: int = 10, signal: float = 0.0, seed: int = 0,
             n_estimators: int = 60) -> SimulationResult:
    X, y, groups, clean = make_dataset(n_farms, copies, signal, seed=seed)
    model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=seed)
    row = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=seed), scoring="r2").mean()
    grp = cross_val_score(model, X, y, groups=groups, cv=GroupKFold(5), scoring="r2").mean()
    oracle = 1 - np.var(y - clean) / np.var(y)
    return SimulationResult(float(row), float(grp), float(oracle), len(X), n_farms)
