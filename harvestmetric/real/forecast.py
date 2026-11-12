"""Inference for Part B: forecasts with intervals from the saved model.

The app and tests call `forecast_rows`, which validates the panel rows and
applies exactly the estimator object that was trained (preprocessing lives
inside it), so training and inference cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .models import REQUIRED_COLUMNS


class ForecastInputError(ValueError):
    """Raised with a user-facing message when rows cannot be forecast."""


def validate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in rows.columns]
    if missing:
        raise ForecastInputError(f"Missing required columns: {missing}")
    needed = [c for c in REQUIRED_COLUMNS if c not in ("prev_yield",)]  # prev_yield may be unpublished
    bad = rows[needed].isna().any(axis=1) | ~np.isfinite(rows[needed].to_numpy(float)).all(axis=1)
    if bad.any():
        raise ForecastInputError(f"{int(bad.sum())} row(s) have missing or non-finite features "
                                 "(county needs >= 15 years of published yields and complete weather)")
    if (rows["year"] < config.FIRST_MODEL_YEAR).any() or (rows["year"] > config.FORECAST_YEAR).any():
        raise ForecastInputError(f"year must be between {config.FIRST_MODEL_YEAR} and {config.FORECAST_YEAR}")
    return rows


def load_artifact(path: str | Path = config.MODEL_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run: python -m harvestmetric.real.train")
    return joblib.load(path)


def forecast_rows(rows: pd.DataFrame, artifact: dict, which: str = "model",
                  allow_unsupported: bool = False) -> pd.DataFrame:
    """Return rows with forecast and 80/90% interval columns (bu/ac).

    which="model" uses the production refit (all published years);
    which="evaluated_model" uses the model frozen at 2011 that produced the test metrics.
    """
    if not artifact.get("prediction_supported") and not allow_unsupported:
        raise RuntimeError("Evaluation did not show a reliable gain over baselines; forecasts are disabled.")
    rows = validate_rows(rows.copy())
    rows["forecast"] = artifact[which].predict(rows)
    for level, q in artifact["interval_half_widths"].items():
        rows[f"lo{int(level * 100)}"] = rows["forecast"] - q
        rows[f"hi{int(level * 100)}"] = rows["forecast"] + q
    return rows
