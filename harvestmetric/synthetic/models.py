"""Candidate models and their (small) hyperparameter grids."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from . import config
from .features import FeatureSet, build_preprocessor


class GroupMeanRegressor(RegressorMixin, BaseEstimator):
    """Predict the training mean of the target within each level of one column.

    A transparent baseline: "expect the average yield for this crop".
    Unseen levels fall back to the overall training mean.
    """

    def __init__(self, column: str = "crop_type"):
        self.column = column

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.means_ = y.groupby(X[self.column]).mean().to_dict()
        self.global_mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].map(self.means_).fillna(self.global_mean_).to_numpy(dtype=float)


def candidate_models(fs: FeatureSet) -> dict[str, tuple[object, dict]]:
    """name -> (estimator, param_grid). Pipelines include preprocessing so the
    exact same transformation is applied in CV, final training and inference."""
    rs = config.RANDOM_STATE

    def pipe(model):
        return Pipeline([("prep", build_preprocessor(fs)), ("model", model)])

    return {
        "Mean baseline": (DummyRegressor(strategy="mean"), {}),
        "Crop-mean baseline": (GroupMeanRegressor("crop_type"), {}),
        "Ridge": (pipe(Ridge()), {"model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}),
        "Random forest": (
            pipe(RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=rs)),
            {"model__max_depth": [3, 6, None], "model__min_samples_leaf": [5, 20]},
        ),
        "Gradient boosting": (
            pipe(HistGradientBoostingRegressor(random_state=rs)),
            {"model__learning_rate": [0.03, 0.1], "model__max_depth": [2, 4],
             "model__l2_regularization": [0.0, 1.0]},
        ),
    }
