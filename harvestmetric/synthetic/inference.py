"""Validated inference with the saved at-sowing pipeline.

The saved artifact is the full sklearn Pipeline (preprocessor + model), so the
transformations applied here are exactly those fitted during training.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from . import config
from .features import AT_SOWING


class InputError(ValueError):
    """Raised with a user-facing message when an input is invalid."""


def validate_input(record: Mapping[str, Any]) -> dict[str, Any]:
    """Check one at-sowing record; return a cleaned copy or raise InputError listing every problem."""
    errors: list[str] = []
    clean: dict[str, Any] = {}
    missing = [c for c in AT_SOWING.columns if c not in record or record[c] in (None, "")]
    if missing:
        raise InputError(f"Missing required field(s): {', '.join(missing)}")
    for col in AT_SOWING.categorical:
        text = str(record[col])
        if text not in config.CATEGORIES[col]:
            errors.append(f"{col} must be one of {config.CATEGORIES[col]} (got {text!r})")
        clean[col] = text
    for col in AT_SOWING.numeric:
        try:
            value = float(record[col])
        except (TypeError, ValueError):
            errors.append(f"{col} must be a number (got {record[col]!r})")
            continue
        lo, hi = (1, 12) if col == "sowing_month" else config.NUMERIC_RANGES[col]
        if not lo <= value <= hi:
            errors.append(f"{col} must be between {lo} and {hi} {config.UNITS.get(col, '')}".rstrip()
                          + f" (got {value:g})")
        if col == "sowing_month" and value != int(value):
            errors.append("sowing_month must be a whole month number")
        clean[col] = value
    if errors:
        raise InputError("; ".join(errors))
    return clean


def load_artifact(path: str | Path = config.MODEL_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run: python -m harvestmetric.synthetic.train")
    artifact = joblib.load(path)
    if artifact.get("feature_set") != AT_SOWING.name:
        raise ValueError("Model artifact was trained on a different feature set")
    return artifact


def predict(records: list[Mapping[str, Any]] | Mapping[str, Any], artifact: dict | None = None,
            allow_unsupported: bool = False) -> list[float]:
    """Predict yield (kg/ha) for one or more validated records.

    Refuses to predict when evaluation did not support predictions, unless
    allow_unsupported=True (used by tests and diagnostics).
    """
    artifact = artifact or load_artifact()
    if not artifact.get("prediction_supported") and not allow_unsupported:
        raise RuntimeError("Validated evaluation did not support predictions from this model.")
    if isinstance(records, Mapping):
        records = [records]
    frame = pd.DataFrame([validate_input(r) for r in records], columns=AT_SOWING.columns)
    return [float(v) for v in artifact["model"].predict(frame)]
