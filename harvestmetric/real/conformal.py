"""Split-conformal prediction intervals calibrated on time-ordered residuals.

Calibration residuals come from expanding-window validation inside the
training period: each was produced by a model trained only on earlier years.
That keeps calibration on the correct side of the time split. Residuals within
a year are correlated (one weather shock hits many counties), so coverage is
reported per year as well as overall.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(residuals, level: float) -> float:
    """Finite-sample split-conformal quantile of absolute residuals."""
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
