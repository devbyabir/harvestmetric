"""Small, testable analytics helpers used by the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class Comparison:
    label_a: str
    label_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    diff: float          # mean_a - mean_b, kg/ha
    ci_low: float
    ci_high: float

    @property
    def conclusive(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def compare_groups(a: pd.Series, b: pd.Series, label_a: str = "A", label_b: str = "B",
                   n_boot: int = 4000, seed: int = config.RANDOM_STATE) -> Comparison:
    """Difference in mean yield with a percentile bootstrap 95% interval."""
    xa, xb = np.asarray(a, float), np.asarray(b, float)
    if len(xa) < 2 or len(xb) < 2:
        raise ValueError("Each group needs at least 2 farms for a comparison.")
    rng = np.random.default_rng(seed)
    diffs = (rng.choice(xa, (n_boot, len(xa))).mean(1) - rng.choice(xb, (n_boot, len(xb))).mean(1))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return Comparison(label_a, label_b, len(xa), len(xb), float(xa.mean()), float(xb.mean()),
                      float(xa.mean() - xb.mean()), float(lo), float(hi))


SCENARIOS = {
    "Rice: drip irrigation vs no irrigation": (
        {"crop_type": "Rice", "irrigation_type": "Drip"}, {"crop_type": "Rice", "irrigation_type": "None"}),
    "Wheat: organic vs inorganic fertiliser": (
        {"crop_type": "Wheat", "fertilizer_type": "Organic"}, {"crop_type": "Wheat", "fertilizer_type": "Inorganic"}),
    "Soybean vs cotton (all regions)": ({"crop_type": "Soybean"}, {"crop_type": "Cotton"}),
    "Maize: healthy vs severe disease": (
        {"crop_type": "Maize", "crop_disease_status": "None"}, {"crop_type": "Maize", "crop_disease_status": "Severe"}),
    "North India vs East Africa (all crops)": ({"region": "North India"}, {"region": "East Africa"}),
}


def apply_filter(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for col, value in spec.items():
        mask &= (df[col] == value).to_numpy()
    return df[mask]


def describe_filter(spec: dict) -> str:
    return ", ".join(f"{k.replace('_', ' ')} = {v}" for k, v in spec.items()) or "all farms"
