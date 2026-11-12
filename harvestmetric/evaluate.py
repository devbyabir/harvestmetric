"""Regression metrics, bootstrap intervals, and breakdowns by crop and yield band."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "n": int(len(y_true)),
    }


def bootstrap_ci(y_true, y_pred, metric: str = "r2", n_boot: int = 2000,
                 seed: int = config.RANDOM_STATE, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval for a metric over test farms."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(regression_metrics(y_true[idx], y_pred[idx])[metric])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_improvement(y_true, pred_model, pred_base, n_boot: int = 2000,
                                 seed: int = config.RANDOM_STATE) -> dict[str, float]:
    """Bootstrap the RMSE reduction of a model vs a baseline on the same farms."""
    y, a, b = (np.asarray(v, float) for v in (y_true, pred_model, pred_base))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        rm = np.sqrt(np.mean((y[idx] - a[idx]) ** 2))
        rb = np.sqrt(np.mean((y[idx] - b[idx]) ** 2))
        diffs.append(rb - rm)
    diffs = np.asarray(diffs)
    return {
        "rmse_reduction_mean": float(diffs.mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "share_of_resamples_model_better": float((diffs > 0).mean()),
    }


def breakdown(frame: pd.DataFrame, by: str, pred_col: str = "predicted",
              true_col: str = config.TARGET) -> pd.DataFrame:
    rows = []
    for level, g in frame.groupby(by, observed=True):
        m = regression_metrics(g[true_col], g[pred_col])
        m["bias"] = float((g[pred_col] - g[true_col]).mean())
        rows.append({by: level, **m})
    return pd.DataFrame(rows)


def yield_band(y: pd.Series) -> pd.Series:
    return pd.cut(y, bins=[2000, 3000, 4000, 5000, 6000.01], right=False,
                  labels=["2000-3000", "3000-4000", "4000-5000", "5000-6000"])
