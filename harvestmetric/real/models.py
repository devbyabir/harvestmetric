"""Baselines and the two candidate models for Part B.

All estimators take the full panel frame as X, so baselines, models and the
saved forecaster share one interface and one preprocessing path.

Model design: yield = county trend (from past yields) + deviation. Learned
models predict the deviation from season-to-date weather and last year's
deviation; the trend term is computed from history only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .prepare import WEATHER_FEATURES

DEVIATION_FEATURES = ["prev_dev", "prev_missing", *WEATHER_FEATURES]
REQUIRED_COLUMNS = ["trend_pred", "hist_mean", "prev_yield", "latest_yield", *DEVIATION_FEATURES]


class _ColumnBaseline(RegressorMixin, BaseEstimator):
    column = ""

    def fit(self, X: pd.DataFrame, y=None):
        self.n_features_in_ = 1
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].to_numpy(float)


class HistoricalMeanBaseline(_ColumnBaseline):
    """Mean of the county's published yields over the previous 30 years."""
    column = "hist_mean"


class TrendBaseline(_ColumnBaseline):
    """County-specific linear trend fitted to the previous 30 years, extrapolated to year t."""
    column = "trend_pred"


class PreviousYearBaseline(RegressorMixin, BaseEstimator):
    """Last year's published county yield; if unpublished, the most recent published year."""

    def fit(self, X: pd.DataFrame, y=None):
        self.n_features_in_ = 1
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["prev_yield"].fillna(X["latest_yield"]).to_numpy(float)


class TrendDeviationRegressor(RegressorMixin, BaseEstimator):
    """Predict yield as trend_pred + f(deviation features)."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y):
        self.model_ = clone(self.estimator).fit(X[DEVIATION_FEATURES], np.asarray(y, float) - X["trend_pred"])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["trend_pred"].to_numpy(float) + self.model_.predict(X[DEVIATION_FEATURES])


def candidates() -> dict[str, tuple[object, list[dict]]]:
    """name -> (estimator, list of parameter settings to try). Small, justified set:

    * three simple baselines a farm analyst would actually use;
    * Ridge: linear weather effects, stable with ~3,000 rows;
    * gradient boosting: allows thresholds (heat >30 °C, drought) and interactions.
    """
    ridge = TrendDeviationRegressor(Pipeline([("scale", StandardScaler()), ("m", Ridge())]))
    hgb = TrendDeviationRegressor(HistGradientBoostingRegressor(random_state=config.RANDOM_STATE))
    return {
        "Historical mean": (HistoricalMeanBaseline(), [{}]),
        "Previous year": (PreviousYearBaseline(), [{}]),
        "County trend": (TrendBaseline(), [{}]),
        "Trend + Ridge": (ridge, [{"estimator__m__alpha": a} for a in (0.1, 1.0, 10.0, 100.0)]),
        "Trend + gradient boosting": (hgb, [
            {"estimator__learning_rate": lr, "estimator__max_depth": d, "estimator__max_iter": n,
             "estimator__min_samples_leaf": 40}
            for lr in (0.03, 0.1) for d in (2, 3) for n in (200, 400)]),
    }


def is_baseline(name: str) -> bool:
    return not name.startswith("Trend +")
