"""Empirical prediction intervals from expanding-window CV residuals.

What these intervals are
    A symmetric band of +/- q around each point forecast, where q is an upper
    empirical quantile of the absolute residuals from expanding-window CV
    inside the training years. Each residual came from a model fitted only on
    earlier years, so no test-year information enters the band.

What they are not
    They are NOT split-conformal intervals and carry no finite-sample coverage
    guarantee. The same CV residuals were used to choose the hyperparameters
    and the model, so the calibration set is not independent of model
    selection. Residuals within a season are also strongly correlated (one
    weather shock hits many counties) and the error distribution shifts over
    time, so exchangeability does not hold. Coverage therefore has to be
    checked empirically, overall and per year, on the untouched test years.

The quantile uses the rank ceil((n + 1) * level), the same finite-sample
rank a split-conformal procedure would use; here it is only a slightly
conservative choice of empirical quantile, not a guarantee.
"""

from __future__ import annotations

import numpy as np


def empirical_quantile(residuals, level: float) -> float:
    """Upper empirical quantile of absolute residuals at rank ceil((n + 1) * level)."""
    r = np.sort(np.abs(np.asarray(residuals, float)))
    n = len(r)
    if n == 0:
        raise ValueError("no calibration residuals")
    k = int(np.ceil((n + 1) * level))
    return float(r[min(k, n) - 1])


def intervals(pred, q: float) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, float)
    return pred - q, pred + q


def coverage(y, lo, hi) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= lo) & (y <= hi)))
