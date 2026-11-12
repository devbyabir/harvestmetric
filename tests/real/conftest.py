"""Tiny generated fixtures for Part B (no downloads needed)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from harvestmetric.real import config


def power_payload(lat: float, years=range(1981, 1987), seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    days = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D")
    doy = days.day_of_year.to_numpy()
    tmax = 15 + 15 * np.sin((doy - 105) / 365 * 2 * np.pi) + rng.normal(0, 2, len(days))
    tmin = tmax - 10
    prcp = rng.gamma(0.5, 5, len(days))
    fmt = days.strftime("%Y%m%d")
    return {"header": {"api": {"version": "test"}, "fill_value": -999.0},
            "properties": {"parameter": {"T2M_MAX": dict(zip(fmt, tmax.round(2))),
                                         "T2M_MIN": dict(zip(fmt, tmin.round(2))),
                                         "PRECTOTCORR": dict(zip(fmt, prcp.round(2)))}}}


def nass_rows(fips_suffix: str, county: str, years, base: float, district="CENTRAL", code="50"):
    return [{"Program": "SURVEY", "Year": str(y), "Period": "YEAR", "Week Ending": "", "Geo Level": "COUNTY",
             "State": "IOWA", "State ANSI": "19", "Ag District": district, "Ag District Code": code,
             "County": county, "County ANSI": fips_suffix, "Zip Code": "", "Region": "", "watershed_code": "0",
             "Watershed": "", "Commodity": "CORN", "Data Item": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
             "Domain": "TOTAL", "Domain Category": "NOT SPECIFIED", "Value": f"{base + 2 * (y - 1950):.1f}",
             "CV (%)": "3.0"} for y in years]


@pytest.fixture
def mini_sources(tmp_path, monkeypatch):
    """Two counties, yields 1950-1986, weather 1981-1986, plus an 'OTHER COUNTIES' row and a withheld value."""
    raw = tmp_path / "raw"
    (raw / "power").mkdir(parents=True)
    rows = nass_rows("015", "BOONE", range(1950, 1987), 60) + nass_rows("169", "STORY", range(1950, 1986), 62)
    rows.append({**rows[0], "County": "OTHER COUNTIES", "County ANSI": "", "Year": "1985"})
    rows.append({**rows[-2], "Year": "1986", "Value": "(D)"})   # STORY 1986 withheld
    pd.DataFrame(rows).to_csv(raw / "nass.csv", index=False)
    pd.DataFrame({"fips": ["19015", "19169"], "county_name": ["Boone County", "Story County"],
                  "lat": [42.03, 42.03], "lon": [-93.93, -93.46]}).to_csv(raw / "gaz.csv", index=False)
    for i, f in enumerate(["19015", "19169"]):
        (raw / "power" / f"{f}.json").write_text(json.dumps(power_payload(42.0, seed=i)))
    monkeypatch.setattr(config, "NASS_FILE", raw / "nass.csv")
    monkeypatch.setattr(config, "GAZETTEER_FILE", raw / "gaz.csv")
    monkeypatch.setattr(config, "POWER_DIR", raw / "power")
    monkeypatch.setattr(config, "FORECAST_YEAR", 1986)
    return raw


def synthetic_panel(n_counties: int = 12, years=range(1981, 2026), seed: int = 0) -> pd.DataFrame:
    """Panel-shaped frame where yield = trend + weather effect + noise (for model/split tests)."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_counties):
        district = f"D{c % 4}"
        for y in years:
            heat = rng.gamma(2, 10)
            trend = 100 + 1.8 * (y - 1981) + c
            rows.append({"fips": f"19{c:03d}", "county": f"C{c}", "district": district,
                         "district_code": str(c % 4), "year": y, "trend_pred": trend,
                         "hist_mean": trend - 15, "prev_yield": trend + rng.normal(0, 8),
                         "latest_yield": trend, "prev_dev": rng.normal(0, 8), "prev_missing": 0.0,
                         "gdd_may_jul": 1200 + rng.normal(0, 60), "edd30_jun_jul": heat,
                         "prcp_apr_may": 200 + rng.normal(0, 40), "prcp_jun": 120 + rng.normal(0, 30),
                         "prcp_jul": 100 + rng.normal(0, 30), "tmax_jul": 29 + rng.normal(0, 1.5),
                         config.TARGET: trend - 0.6 * heat + rng.normal(0, 5)})
    return pd.DataFrame(rows)


@pytest.fixture
def panel() -> pd.DataFrame:
    return synthetic_panel()
