"""Build the county-year panel used for modelling.

Run: python -m harvestmetric.real.prepare

One row per (county FIPS, year) for FIRST_MODEL_YEAR..FORECAST_YEAR. Every
feature is computed only from information available on the forecast date
(1 August of the row's year):

* weather: NASA POWER daily values at the county's internal point, only for
  dates on or before 31 July of that year;
* history: the county's own published yields for the previous HISTORY_WINDOW
  years (never the current year, never other counties).

The target (yield_bu_ac) is left empty where NASS did not publish a county
value (suppressed, combined into "OTHER COUNTIES", or not yet released).
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from . import config


class PanelError(ValueError):
    """Raised when source data violate expectations."""


# --------------------------------------------------------------------- yields
def load_yields(path=None) -> pd.DataFrame:
    raw = pd.read_csv(path or config.NASS_FILE, dtype=str)
    expected = "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE"
    if set(raw["Data Item"]) != {expected} or set(raw["Geo Level"]) != {"COUNTY"}:
        raise PanelError("NASS file contains unexpected data items or geographic levels")
    # Combined / "other" counties have no county code: they cannot be matched to a place.
    named = raw[raw["County ANSI"].notna() & ~raw["County"].str.contains("OTHER", na=False)].copy()
    named["fips"] = config.STATE_FIPS + named["County ANSI"].str.zfill(3)
    named["year"] = named["Year"].astype(int)
    value = named["Value"].str.replace(",", "", regex=False).str.strip()
    withheld = value.str.startswith("(")          # e.g. (D) = withheld to avoid disclosure
    named[config.TARGET] = pd.to_numeric(value.where(~withheld), errors="raise")
    named["nass_cv_pct"] = pd.to_numeric(named["CV (%)"], errors="coerce")
    out = named.rename(columns={"County": "county", "Ag District": "district",
                                "Ag District Code": "district_code"})
    out = out[["fips", "county", "district_code", "district", "year", config.TARGET, "nass_cv_pct"]]
    out = out.dropna(subset=[config.TARGET])
    if out.duplicated(["fips", "year"]).any():
        raise PanelError("duplicate county-year yields in NASS file")
    if not out[config.TARGET].between(1, 350).all():
        raise PanelError("yield values outside plausible 1-350 bu/ac range")
    return out.sort_values(["fips", "year"]).reset_index(drop=True)


# -------------------------------------------------------------------- weather
def load_weather(fips: str) -> pd.DataFrame:
    payload = json.loads((config.POWER_DIR / f"{fips}.json").read_text())
    params = payload["properties"]["parameter"]
    fill = payload["header"].get("fill_value", -999.0)
    daily = pd.DataFrame({k: pd.Series(v) for k, v in params.items()})
    daily.index = pd.to_datetime(daily.index, format="%Y%m%d")
    daily = daily.replace(fill, np.nan).rename(columns={"T2M_MAX": "tmax", "T2M_MIN": "tmin",
                                                        "PRECTOTCORR": "prcp"})
    return daily.sort_index()


def gdd_corn(tmax: pd.Series, tmin: pd.Series) -> pd.Series:
    """Corn growing degree days (°C), 10/30 °C method (US 86/50 °F convention)."""
    hi = tmax.clip(upper=30.0)
    lo = tmin.clip(lower=10.0, upper=30.0)
    return ((hi + lo) / 2 - 10.0).clip(lower=0.0)


WEATHER_FEATURES = ["gdd_may_jul", "edd30_jun_jul", "prcp_apr_may", "prcp_jun", "prcp_jul", "tmax_jul"]


def weather_features(daily: pd.DataFrame, year: int) -> dict[str, float | str]:
    """Season-to-date weather for `year`, using only days up to the forecast cut-off."""
    cutoff = pd.Timestamp(date(year, *config.FORECAST_MONTH_DAY)) - pd.Timedelta(days=1)  # 31 July
    d = daily.loc[f"{year}-04-01":cutoff]
    if d.empty or d.index.max() > cutoff:
        raise PanelError(f"weather window for {year} is empty or extends past the forecast date")
    if d[["tmax", "tmin", "prcp"]].isna().mean().max() > 0.02:
        raise PanelError(f"more than 2% missing weather days in {year}")
    month = pd.DatetimeIndex(d.index).month
    gdd = gdd_corn(d.tmax, d.tmin)
    return {
        "gdd_may_jul": float(gdd[(month >= 5) & (month <= 7)].sum()),
        "edd30_jun_jul": float((d.tmax[(month >= 6) & (month <= 7)] - 30.0).clip(lower=0).sum()),
        "prcp_apr_may": float(d.prcp[month <= 5].sum()),
        "prcp_jun": float(d.prcp[month == 6].sum()),
        "prcp_jul": float(d.prcp[month == 7].sum()),
        "tmax_jul": float(d.tmax[month == 7].mean()),
        "weather_last_day": str(d.index.max().date()),
    }


# -------------------------------------------------------------------- history
HISTORY_FEATURES = ["trend_pred", "hist_mean", "prev_yield", "latest_yield", "prev_dev", "prev_missing"]


def history_features(series: pd.Series, year: int) -> dict[str, float]:
    """Features from one county's published yields strictly before `year`.

    series: yields indexed by year for a single county.
    """
    past = series[(series.index < year) & (series.index >= year - config.HISTORY_WINDOW)].dropna()
    if len(past) < config.MIN_HISTORY:
        return {k: np.nan for k in HISTORY_FEATURES}
    slope, intercept = np.polyfit(past.index.to_numpy(float), past.to_numpy(float), 1)
    trend = slope * year + intercept
    prev = series.get(year - 1, np.nan)
    prev_missing = bool(pd.isna(prev))
    prev_trend = slope * (year - 1) + intercept
    return {
        "trend_pred": float(trend),
        "hist_mean": float(past.mean()),
        "prev_yield": float(prev) if not prev_missing else np.nan,
        "latest_yield": float(past.iloc[-1]),
        "prev_dev": 0.0 if prev_missing else float(prev - prev_trend),
        "prev_missing": float(prev_missing),
    }


# ---------------------------------------------------------------------- panel
def build_panel() -> pd.DataFrame:
    yields = load_yields()
    centroids = pd.read_csv(config.GAZETTEER_FILE, dtype={"fips": str})
    names = yields.groupby("fips").agg(county=("county", "last"),
                                       district_code=("district_code", lambda s: s.mode().iat[0]),
                                       district=("district", lambda s: s.mode().iat[0]))
    rows: list[dict[str, object]] = []
    for fips, meta in names.iterrows():
        series = yields[yields.fips == fips].set_index("year")[config.TARGET]
        cvs = yields[yields.fips == fips].set_index("year")["nass_cv_pct"]
        daily = load_weather(str(fips))
        for year in range(config.FIRST_MODEL_YEAR, config.FORECAST_YEAR + 1):
            hist = history_features(series, year)
            if np.isnan(hist["trend_pred"]):
                continue
            rows.append({"fips": fips, **{str(k): v for k, v in meta.items()}, "year": year,
                         config.TARGET: series.get(year, np.nan), "nass_cv_pct": cvs.get(year, np.nan),
                         **hist, **weather_features(daily, year)})
    panel = pd.DataFrame(rows).merge(centroids[["fips", "lat", "lon"]], on="fips", how="left")
    if panel[["lat", "lon"]].isna().any().any():
        raise PanelError("county without Gazetteer coordinates")
    return panel.sort_values(["year", "fips"]).reset_index(drop=True)


def main() -> None:
    panel = build_panel()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.round(4).to_csv(config.PANEL_PATH, index=False)
    observed = panel[config.TARGET].notna()
    print(f"Wrote {config.PANEL_PATH.relative_to(config.PROJECT_ROOT)}: {len(panel)} county-years, "
          f"{observed.sum()} with published yield, {panel.fips.nunique()} counties, "
          f"years {panel.year.min()}-{panel.year.max()}")


if __name__ == "__main__":
    main()
