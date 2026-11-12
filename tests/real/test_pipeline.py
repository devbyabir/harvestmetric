import numpy as np
import pandas as pd
import pytest

from harvestmetric.real import config
from harvestmetric.real.conformal import conformal_quantile, coverage, intervals
from harvestmetric.real.forecast import ForecastInputError, forecast_rows
from harvestmetric.real.models import (
    HistoricalMeanBaseline,
    PreviousYearBaseline,
    TrendBaseline,
    TrendDeviationRegressor,
    candidates,
)
from harvestmetric.real.prepare import (
    PanelError,
    build_panel,
    gdd_corn,
    history_features,
    load_weather,
    load_yields,
    weather_features,
)
from harvestmetric.real.splits import (
    SplitLeakageError,
    assert_time_order,
    district_folds,
    expanding_window_folds,
    time_split,
)
from harvestmetric.real.train import year_bootstrap_rmse_gain

# ------------------------------------------------------------------ yields / units / missing data


def test_yields_drop_combined_counties_and_withheld_values(mini_sources):
    y = load_yields()
    assert set(y.fips) == {"19015", "19169"}
    assert not ((y.fips == "19169") & (y.year == 1986)).any()   # (D) withheld -> no target
    assert y[config.TARGET].dtype == float


def test_yields_reject_wrong_unit(mini_sources):
    df = pd.read_csv(config.NASS_FILE, dtype=str)
    df.loc[0, "Data Item"] = "CORN, GRAIN - YIELD, MEASURED IN TONNE / HECTARE"
    df.to_csv(config.NASS_FILE, index=False)
    with pytest.raises(PanelError, match="unexpected data items"):
        load_yields()


def test_yields_reject_implausible_values(mini_sources):
    df = pd.read_csv(config.NASS_FILE, dtype=str)
    df.loc[3, "Value"] = "9999"
    df.to_csv(config.NASS_FILE, index=False)
    with pytest.raises(PanelError, match="plausible"):
        load_yields()


def test_weather_fill_values_become_missing(mini_sources):
    import json
    path = config.POWER_DIR / "19015.json"
    payload = json.loads(path.read_text())
    payload["properties"]["parameter"]["T2M_MAX"]["19810101"] = -999.0
    path.write_text(json.dumps(payload))
    assert np.isnan(load_weather("19015").loc["1981-01-01", "tmax"])


def test_gdd_known_values():
    tmax, tmin = pd.Series([35.0, 20.0, 8.0]), pd.Series([20.0, 5.0, 2.0])
    assert gdd_corn(tmax, tmin).tolist() == [15.0, 5.0, 0.0]


# ------------------------------------------------------------------ time leakage


def test_weather_features_ignore_days_after_forecast_date(mini_sources):
    daily = load_weather("19015")
    before = weather_features(daily, 1984)
    daily.loc["1984-08-01":"1984-12-31", ["tmax", "prcp"]] = 999.0   # absurd future values
    after = weather_features(daily, 1984)
    assert before == after
    assert after["weather_last_day"] == "1984-07-31"


def test_history_features_use_only_previous_years():
    s = pd.Series(np.linspace(80, 140, 40), index=range(1950, 1990))
    a = history_features(s, 1985)
    s2 = s.copy()
    s2.loc[1985:] = 0.0   # change current and future years
    assert history_features(s2, 1985) == a


def test_history_requires_minimum_years():
    s = pd.Series([100.0] * 10, index=range(1975, 1985))
    assert np.isnan(history_features(s, 1985)["trend_pred"])


def test_prev_year_missing_is_flagged():
    s = pd.Series(np.linspace(80, 140, 40), index=range(1950, 1990)).drop(1984)
    h = history_features(s, 1985)
    assert h["prev_missing"] == 1.0 and np.isnan(h["prev_yield"]) and h["prev_dev"] == 0.0


def test_panel_join_keys_and_coordinates(mini_sources):
    p = build_panel()
    assert not p.duplicated(["fips", "year"]).any()
    assert p.year.min() == 1981 and p.year.max() == 1986
    assert p.loc[p.fips == "19169", "lat"].iat[0] == pytest.approx(42.03)
    assert p.loc[(p.fips == "19169") & (p.year == 1986), config.TARGET].isna().all()  # withheld kept as NaN


def test_time_split_order(panel):
    train, test = time_split(panel)
    assert train.year.max() == config.TRAIN_END and test.year.min() == config.TEST_START
    with pytest.raises(SplitLeakageError):
        assert_time_order(test, train)


def test_expanding_folds_train_strictly_before_validation(panel):
    train, _ = time_split(panel)
    for tr, va in expanding_window_folds(train):
        assert train.year.iloc[tr].max() < train.year.iloc[va].min()


# ------------------------------------------------------------------ geographic split


def test_district_folds_hold_out_whole_districts(panel):
    folds = district_folds(panel)
    assert len(folds) == panel.district_code.nunique()
    for _, tr, te in folds:
        assert set(panel.fips.iloc[tr]).isdisjoint(panel.fips.iloc[te])
        assert panel.district_code.iloc[te].nunique() == 1


# ------------------------------------------------------------------ models / predictions


def test_baselines(panel):
    row = panel.head(3).copy()
    row.loc[row.index[0], "prev_yield"] = np.nan
    assert np.allclose(TrendBaseline().fit(row).predict(row), row.trend_pred)
    assert np.allclose(HistoricalMeanBaseline().fit(row).predict(row), row.hist_mean)
    prev = PreviousYearBaseline().fit(row).predict(row)
    assert prev[0] == row.latest_yield.iat[0] and prev[1] == row.prev_yield.iat[1]


def test_trend_deviation_model_learns_weather_signal(panel):
    train, test = time_split(panel)
    est, grid = candidates()["Trend + Ridge"]
    m = est.set_params(**grid[1]).fit(train, train[config.TARGET])
    err_model = np.sqrt(np.mean((m.predict(test) - test[config.TARGET]) ** 2))
    err_trend = np.sqrt(np.mean((test.trend_pred - test[config.TARGET]) ** 2))
    assert err_model < err_trend


def test_trend_deviation_adds_trend(panel):
    class Zero:
        def fit(self, X, y): return self
        def predict(self, X): return np.zeros(len(X))
        def get_params(self, deep=True): return {}
    m = TrendDeviationRegressor(Zero()).fit(panel, panel[config.TARGET])
    assert np.allclose(m.predict(panel), panel.trend_pred)


def test_forecast_rows_validates_and_gates(panel):
    train, test = time_split(panel)
    est, grid = candidates()["Trend + Ridge"]
    model = est.set_params(**grid[1]).fit(train, train[config.TARGET])
    art = {"model": model, "evaluated_model": model, "interval_half_widths": {0.8: 10.0, 0.9: 15.0},
           "prediction_supported": True}
    out = forecast_rows(test.head(5), art)
    assert np.allclose(out.forecast, model.predict(test.head(5)))
    assert np.allclose(out.hi90 - out.lo90, 30.0)
    bad = test.head(2).copy()
    bad.loc[bad.index[0], "edd30_jun_jul"] = np.nan
    with pytest.raises(ForecastInputError, match="missing or non-finite"):
        forecast_rows(bad, art)
    with pytest.raises(RuntimeError, match="disabled"):
        forecast_rows(test.head(2), {**art, "prediction_supported": False})


# ------------------------------------------------------------------ uncertainty / statistics


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    q = conformal_quantile(rng.normal(0, 1, 2000), 0.9)
    y = rng.normal(0, 1, 20000)
    lo, hi = intervals(np.zeros_like(y), q)
    assert 0.88 < coverage(y, lo, hi) < 0.92


def test_year_bootstrap_zero_gain_for_identical_predictions(panel):
    g = year_bootstrap_rmse_gain(panel, panel.trend_pred, panel.trend_pred, n_boot=200)
    assert g["rmse_gain"] == 0 and g["ci_low"] == 0 and g["ci_high"] == 0


def test_year_bootstrap_comparison_detects_heat_effect(panel):
    from harvestmetric.real.analysis import compare_by_year_bootstrap, with_deviation
    d = with_deviation(panel)
    hot = d.edd30_jun_jul > d.edd30_jun_jul.quantile(0.8)
    c = compare_by_year_bootstrap(d, hot, ~hot, value="dev_bu_ac", n_boot=500)
    assert c.diff < 0 and c.conclusive


def test_year_bootstrap_comparison_requires_several_years(panel):
    from harvestmetric.real.analysis import compare_by_year_bootstrap, with_deviation
    d = with_deviation(panel)
    with pytest.raises(ValueError, match="at least 3"):
        compare_by_year_bootstrap(d, d.year == 1990, d.year != 1990)
