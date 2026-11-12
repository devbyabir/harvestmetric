"""Descriptive comparisons of observed county-years (used by the app's Scenarios page).

These describe associations in historical data. They are not causal effects
and they are not model forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


def with_deviation(panel: pd.DataFrame) -> pd.DataFrame:
    """Add yield deviation from the county's pre-season trend, in bu/ac and percent."""
    out = panel[panel[config.TARGET].notna()].copy()
    out["dev_bu_ac"] = out[config.TARGET] - out["trend_pred"]
    out["dev_pct"] = 100 * out["dev_bu_ac"] / out["trend_pred"]
    out["prcp_jun_jul"] = out["prcp_jun"] + out["prcp_jul"]
    return out


@dataclass
class GroupComparison:
    n_a: int
    n_b: int
    years_a: int
    years_b: int
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float

    @property
    def conclusive(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def compare_by_year_bootstrap(frame: pd.DataFrame, mask_a, mask_b, value: str = "dev_pct",
                              n_boot: int = 3000, seed: int = config.RANDOM_STATE) -> GroupComparison:
    """Difference in mean `value` (A - B) with a bootstrap that resamples whole years.

    County-years in the same season share one weather shock, so resampling
    rows independently would overstate certainty.
    """
    a, b = frame[np.asarray(mask_a)], frame[np.asarray(mask_b)]
    if a.year.nunique() < 3 or b.year.nunique() < 3:
        raise ValueError("Each group needs county-years from at least 3 different years.")
    years = np.unique(frame.year)
    rng = np.random.default_rng(seed)
    ga, gb = a.groupby("year")[value], b.groupby("year")[value]
    sums_a, cnt_a = ga.sum().reindex(years, fill_value=0), ga.size().reindex(years, fill_value=0)
    sums_b, cnt_b = gb.sum().reindex(years, fill_value=0), gb.size().reindex(years, fill_value=0)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(years), len(years))
        na, nb = cnt_a.to_numpy()[pick].sum(), cnt_b.to_numpy()[pick].sum()
        if na == 0 or nb == 0:
            continue
        diffs.append(sums_a.to_numpy()[pick].sum() / na - sums_b.to_numpy()[pick].sum() / nb)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return GroupComparison(len(a), len(b), a.year.nunique(), b.year.nunique(), float(a[value].mean()),
                           float(b[value].mean()), float(a[value].mean() - b[value].mean()), float(lo), float(hi))
