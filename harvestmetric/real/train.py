"""Part B evaluation: Iowa county corn yield forecast issued on 1 August.

Run: python -m harvestmetric.real.train

Protocol
1. Training years 1981-2011; untouched test years 2012-2025.
2. Hyperparameters chosen by expanding-window CV inside the training years
   (each validation block uses only earlier years for fitting).
3. Model selection on CV RMSE only; the test years are scored once afterwards.
4. Significance vs baselines with a bootstrap that resamples whole years,
   because errors of counties in the same season are strongly correlated.
5. 80% / 90% empirical intervals from the expanding-window CV residuals of the
   selected model. These residuals were also used for model selection, so the
   intervals have no split-conformal coverage guarantee; coverage is checked
   empirically, overall and per test year.
6. Separate geographic check: leave-one-district-out inside the training years,
   with model choice and hyperparameters re-tuned inside every district fold
   (expanding-window CV on the other eight districts only).
7. Error analysis by year, district and weather-defined extreme groups.
8. Rolling-origin annual refit backtest, then a production refit for 2026.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

from ..metrics import regression_metrics
from . import config
from .intervals import coverage, empirical_quantile, intervals
from .models import DEVIATION_FEATURES, candidates, is_baseline
from .splits import district_folds, expanding_window_folds, time_split

LEVELS = (0.8, 0.9)


def load_panel(path=config.PANEL_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"fips": str, "district_code": str})


def cv_predict(estimator, frame: pd.DataFrame, folds) -> np.ndarray:
    """Out-of-fold predictions for validation rows (NaN elsewhere)."""
    oof = np.full(len(frame), np.nan)
    for tr, va in folds:
        model = clone(estimator).fit(frame.iloc[tr], frame[config.TARGET].iloc[tr])
        oof[va] = model.predict(frame.iloc[va])
    return oof


def year_bootstrap_rmse_gain(frame: pd.DataFrame, pred_a, pred_b, n_boot: int = 4000,
                             seed: int = config.RANDOM_STATE) -> dict[str, float]:
    """RMSE(b) - RMSE(a), resampling whole years (cluster bootstrap). Positive = a better."""
    y = frame[config.TARGET].to_numpy(float)
    ea2, eb2 = (y - np.asarray(pred_a)) ** 2, (y - np.asarray(pred_b)) ** 2
    years = frame.year.to_numpy()
    uniq = np.unique(years)
    idx_by_year = {yr: np.flatnonzero(years == yr) for yr in uniq}
    rng = np.random.default_rng(seed)
    gains = np.empty(n_boot)
    for i in range(n_boot):
        rows = np.concatenate([idx_by_year[yr] for yr in rng.choice(uniq, len(uniq))])
        gains[i] = np.sqrt(eb2[rows].mean()) - np.sqrt(ea2[rows].mean())
    return {"rmse_gain": float(np.sqrt(eb2.mean()) - np.sqrt(ea2.mean())),
            "ci_low": float(np.quantile(gains, 0.025)), "ci_high": float(np.quantile(gains, 0.975)),
            "share_resamples_better": float((gains > 0).mean()), "years": int(len(uniq))}


def grouped_metrics(frame: pd.DataFrame, by: str, cols: dict[str, str]) -> pd.DataFrame:
    rows = []
    for key, g in frame.groupby(by, observed=True):
        row = {by: key, "n": len(g)}
        for label, col in cols.items():
            m = regression_metrics(g[config.TARGET], g[col])
            row[f"{label}_mae"], row[f"{label}_rmse"] = m["mae"], m["rmse"]
            row[f"{label}_bias"] = float((g[col] - g[config.TARGET]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def tune(cands: dict, frame: pd.DataFrame, folds) -> dict[str, tuple[dict, dict, np.ndarray]]:
    """Best (params, CV metrics, out-of-fold predictions) per candidate, by pooled CV RMSE.

    Uses only `frame`; callers decide which rows it contains.
    """
    mask = np.zeros(len(frame), bool)
    for _, va in folds:
        mask[va] = True
    out = {}
    for name, (est, grid) in cands.items():
        best: tuple[dict, dict, np.ndarray] | None = None
        for params in grid:
            oof = cv_predict(clone(est).set_params(**params), frame, folds)
            m = regression_metrics(frame[config.TARGET][mask], oof[mask])
            if best is None or m["rmse"] < best[1]["rmse"]:
                best = (params, m, oof)
        assert best is not None
        out[name] = best
    return out


def geographic_check(train: pd.DataFrame, cands: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Leave-one-district-out with all model choices made inside each fold.

    For every held-out district, the learned model family and its hyperparameters are
    chosen by expanding-window CV on the other eight districts only, then refitted on
    those districts and scored once on the held-out one. No row of the held-out
    district influences any choice made for it.
    """
    learned = {n: c for n, c in cands.items() if not is_baseline(n)}
    rows, pred = [], pd.Series(np.nan, index=train.index)
    for district, tr, te in district_folds(train):
        inner = train.iloc[tr].reset_index(drop=True)
        tuned = tune(learned, inner, expanding_window_folds(inner))
        name = min(tuned, key=lambda n: tuned[n][1]["rmse"])
        params, inner_cv, _ = tuned[name]
        model = clone(learned[name][0]).set_params(**params).fit(inner, inner[config.TARGET])
        pred.iloc[te] = model.predict(train.iloc[te])
        m = regression_metrics(train[config.TARGET].iloc[te], pred.iloc[te])
        t = regression_metrics(train[config.TARGET].iloc[te], train.trend_pred.iloc[te])
        rows.append({"held_out_district": district, "n": m["n"], "model_rmse": m["rmse"],
                     "model_mae": m["mae"], "trend_rmse": t["rmse"], "trend_mae": t["mae"],
                     "model_chosen_in_fold": name,
                     "params_chosen_in_fold": json.dumps({k.split("__")[-1]: v for k, v in params.items()}),
                     "inner_cv_rmse": inner_cv["rmse"]})
    return pd.DataFrame(rows), pred


def run() -> dict:
    t0 = time.time()
    panel = load_panel()
    train, test = time_split(panel)
    folds = expanding_window_folds(train)
    cands = candidates()
    print(f"Train {train.year.min()}-{train.year.max()} ({len(train)} county-years), "
          f"test {test.year.min()}-{test.year.max()} ({len(test)} county-years)")

    # ---- 1. tuning + CV estimate (training years only)
    cv_mask = np.zeros(len(train), bool)
    for _, va in folds:
        cv_mask[va] = True
    results, fitted, oof_best = {}, {}, {}
    tuned = tune(cands, train, folds)
    for name, (est, _) in cands.items():
        params, cv_m, oof = tuned[name]
        model = clone(est).set_params(**params).fit(train, train[config.TARGET])
        pred = model.predict(test)
        fitted[name], oof_best[name] = model, oof
        test[f"pred_{name}"] = pred
        results[name] = {"params": {k.split("__")[-1]: v for k, v in params.items()},
                         "cv": cv_m, "test": regression_metrics(test[config.TARGET], pred)}
        print(f"  {name:26s} CV RMSE {cv_m['rmse']:6.2f} R2 {cv_m['r2']:+.3f} | "
              f"test RMSE {results[name]['test']['rmse']:6.2f} MAE {results[name]['test']['mae']:6.2f} "
              f"R2 {results[name]['test']['r2']:+.3f}")

    learned = [n for n in cands if not is_baseline(n)]
    baselines = [n for n in cands if is_baseline(n)]
    selected = min(learned, key=lambda n: results[n]["cv"]["rmse"])
    best_cv_baseline = min(baselines, key=lambda n: results[n]["cv"]["rmse"])
    strongest_test_baseline = min(baselines, key=lambda n: results[n]["test"]["rmse"])
    sel_col = f"pred_{selected}"

    # ---- 2. significance vs every baseline (year-cluster bootstrap)
    vs = {b: year_bootstrap_rmse_gain(test, test[sel_col], test[f"pred_{b}"]) for b in baselines}

    # ---- 3. empirical intervals from the selected model's time-ordered CV residuals.
    # These residuals also drove model selection, so there is no conformal guarantee.
    resid = (train[config.TARGET] - oof_best[selected])[cv_mask]
    qs = {lvl: empirical_quantile(resid, lvl) for lvl in LEVELS}
    for lvl, q in qs.items():
        lo, hi = intervals(test[sel_col], q)
        test[f"lo{int(lvl*100)}"], test[f"hi{int(lvl*100)}"] = lo, hi
    cov = {f"{int(lvl*100)}%": {"half_width_bu_ac": qs[lvl],
                                "test_coverage": coverage(test[config.TARGET], test[f"lo{int(lvl*100)}"],
                                                          test[f"hi{int(lvl*100)}"])} for lvl in LEVELS}
    year_cov = test.assign(in80=lambda d: (d[config.TARGET] >= d.lo80) & (d[config.TARGET] <= d.hi80),
                           in90=lambda d: (d[config.TARGET] >= d.lo90) & (d[config.TARGET] <= d.hi90)) \
        .groupby("year")[["in80", "in90"]].mean()

    gate = bool(results[selected]["cv"]["rmse"] < results[best_cv_baseline]["cv"]["rmse"]
                and vs[strongest_test_baseline]["ci_low"] > 0)

    # ---- 4. error analysis on the test years
    cols = {"model": sel_col, "trend": "pred_County trend"}
    by_year = grouped_metrics(test, "year", cols).merge(year_cov.reset_index(), on="year")
    by_district = grouped_metrics(test, "district", cols)
    p = {"edd90": train.edd30_jun_jul.quantile(0.9),
         "dry10": (train.prcp_jun + train.prcp_jul).quantile(0.1),
         "wet90": (train.prcp_jun + train.prcp_jul).quantile(0.9)}
    summer = test.prcp_jun + test.prcp_jul
    test["weather_group"] = np.select(
        [test.edd30_jun_jul > p["edd90"], summer < p["dry10"], summer > p["wet90"]],
        ["Extreme heat (Jun-Jul EDD>30°C above train P90)", "Dry summer (Jun-Jul rain below train P10)",
         "Wet summer (Jun-Jul rain above train P90)"], "Typical")
    by_weather = grouped_metrics(test, "weather_group", cols)
    test["shortfall"] = np.where(test[config.TARGET] < 0.85 * test.trend_pred,
                                 "Yield >15% below trend (outcome-defined)", "Within 15% of trend or above")
    by_shortfall = grouped_metrics(test, "shortfall", cols)

    # ---- 5. explanation, only for a validated model
    importance = None
    if gate:
        rng = np.random.default_rng(config.RANDOM_STATE)
        base_rmse = results[selected]["test"]["rmse"]
        rows = []
        for feat in DEVIATION_FEATURES:
            incs = []
            for _ in range(20):
                shuffled = test.copy()
                shuffled[feat] = rng.permutation(shuffled[feat].to_numpy())
                incs.append(regression_metrics(test[config.TARGET], fitted[selected].predict(shuffled))["rmse"]
                            - base_rmse)
            rows.append({"feature": feat, "rmse_increase_mean": float(np.mean(incs)),
                         "rmse_increase_std": float(np.std(incs))})
        importance = pd.DataFrame(rows).sort_values("rmse_increase_mean", ascending=False)

    # ---- 6. geographic generalisation (training years only; reported separately; tuned inside each fold)
    sel_est = clone(cands[selected][0]).set_params(**{f"{k}": v for k, v in
                                                     _full_params(cands[selected], results[selected]).items()})
    geo, geo_pred = geographic_check(train, cands)
    geo_overall = {"model": regression_metrics(train[config.TARGET], geo_pred),
                   "trend": regression_metrics(train[config.TARGET], train.trend_pred),
                   "districts_where_model_beats_trend": int((geo.model_rmse < geo.trend_rmse).sum())}

    # ---- 7. rolling-origin annual refit (same hyperparameters) and 2026 production forecast
    obs = panel[panel[config.TARGET].notna()]
    roll = []
    for yr in range(config.TEST_START, config.LAST_YIELD_YEAR + 1):
        hist, cur = obs[obs.year < yr], obs[obs.year == yr]
        model = clone(sel_est).fit(hist, hist[config.TARGET])
        roll.append(cur.assign(pred_roll=model.predict(cur)))
    roll_df = pd.concat(roll)
    roll_m = regression_metrics(roll_df[config.TARGET], roll_df.pred_roll)
    roll_trend = regression_metrics(roll_df[config.TARGET], roll_df.trend_pred)
    roll_vs_trend = year_bootstrap_rmse_gain(roll_df, roll_df.pred_roll, roll_df.trend_pred)
    prod_q = {lvl: empirical_quantile(roll_df[config.TARGET] - roll_df.pred_roll, lvl) for lvl in LEVELS}
    production, future = None, None
    if gate:  # forecasts for the unreleased season are produced only for a validated model
        production = clone(sel_est).fit(obs, obs[config.TARGET])
        future = panel[(panel.year == config.FORECAST_YEAR)].copy()
        future["forecast"] = production.predict(future)
        for lvl, q in prod_q.items():
            future[f"lo{int(lvl*100)}"], future[f"hi{int(lvl*100)}"] = future.forecast - q, future.forecast + q

    out = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "random_state": config.RANDOM_STATE,
        "task": {"unit": "Iowa county x year", "target": "NASS corn-for-grain yield, bu/acre",
                 "forecast_date": "1 August of the target year",
                 "information": "county's published yields up to year t-1; NASA POWER weather 1 Apr-31 Jul of year t"},
        "data": {"train_years": [int(train.year.min()), int(train.year.max())], "train_rows": int(len(train)),
                 "test_years": [int(test.year.min()), int(test.year.max())], "test_rows": int(len(test)),
                 "counties": int(panel.fips.nunique()), "cv_blocks": config.CV_BLOCKS},
        "results": results,
        "selection": {"selected_model": selected, "best_cv_baseline": best_cv_baseline,
                      "strongest_test_baseline": strongest_test_baseline,
                      "rule": "lowest expanding-window CV RMSE among learned models; test used once afterwards",
                      "rule_status": ("acceptance rule written into this script before the test scores were "
                                      "computed (author's account); not an independently dated preregistration"),
                      "vs_baselines_year_bootstrap": vs, "prediction_supported": gate,
                      "gate": ("selected model beats best CV baseline in CV AND its test RMSE gain over the "
                               "strongest test baseline has a year-cluster bootstrap 95% CI above 0")},
        "intervals": {"method": ("empirical intervals: +/- upper quantile of |expanding-window CV residuals| "
                                 "of the selected model (training years)"),
                      "guarantee": ("none: the same CV residuals were used for model selection, and residuals are "
                                    "correlated within years and shift over time, so split-conformal coverage "
                                    "does not apply; coverage is only checked empirically on the test years"),
                      "calibration_rows": int(cv_mask.sum()), **cov},
        "geographic_check": {"design": ("leave-one-NASS-district-out, training years 1981-2011 only; model "
                                        "family and hyperparameters re-selected inside each fold by "
                                        "expanding-window CV on the other eight districts"),
                             "caveat": ("same seasons on both sides: measures transfer across space, not skill "
                                        "on new seasons"),
                             **geo_overall},
        "weather_thresholds_from_train": {k: float(v) for k, v in p.items()},
        "rolling_origin_refit": {"years": [config.TEST_START, config.LAST_YIELD_YEAR],
                                 "design": "refit every year on all earlier years, same hyperparameters",
                                 "model": roll_m, "trend": roll_trend, "vs_trend_year_bootstrap": roll_vs_trend},
        "forecast_2026": ({"counties": int(len(future)) if future is not None else 0,
                           "interval_half_widths": {f"{int(k*100)}%": v for k, v in prod_q.items()},
                           "calibration": ("empirical, rolling-origin residuals 2012-2025; "
                                           "no coverage guarantee")} if gate else
                          {"produced": False, "reason": "prediction gate not passed"}),
        "runtime_seconds": round(time.time() - t0, 1),
    }

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["fips", "county", "district", "year", "lat", "lon", config.TARGET, "trend_pred",
            *[c for c in test.columns if c.startswith("pred_")], "lo80", "hi80", "lo90", "hi90",
            "weather_group", "shortfall", "edd30_jun_jul", "prcp_jun", "prcp_jul"]
    test[keep].round(2).to_csv(config.TABLES_DIR / "test_predictions.csv", index=False)
    by_year.round(3).to_csv(config.TABLES_DIR / "test_by_year.csv", index=False)
    by_district.round(3).to_csv(config.TABLES_DIR / "test_by_district.csv", index=False)
    by_weather.round(3).to_csv(config.TABLES_DIR / "test_by_weather_group.csv", index=False)
    by_shortfall.round(3).to_csv(config.TABLES_DIR / "test_by_shortfall.csv", index=False)
    geo.round(3).to_csv(config.TABLES_DIR / "geographic_check.csv", index=False)
    roll_df[["fips", "county", "district", "year", config.TARGET, "trend_pred", "pred_roll"]].round(2) \
        .to_csv(config.TABLES_DIR / "rolling_origin_predictions.csv", index=False)
    forecast_file = config.TABLES_DIR / "forecast_2026.csv"
    if future is not None:
        future[["fips", "county", "district", "lat", "lon", "year", "trend_pred", "forecast", "lo80", "hi80",
                "lo90", "hi90", "edd30_jun_jul", "prcp_jun", "prcp_jul", "weather_last_day"]].round(2) \
            .to_csv(forecast_file, index=False)
    elif forecast_file.exists():
        forecast_file.unlink()
    if importance is not None:
        importance.round(3).to_csv(config.TABLES_DIR / "permutation_importance.csv", index=False)
    pd.DataFrame([{"model": n, **{f"cv_{k}": v for k, v in r["cv"].items()},
                   **{f"test_{k}": v for k, v in r["test"].items()}} for n, r in results.items()]) \
        .round(4).to_csv(config.TABLES_DIR / "model_comparison.csv", index=False)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": production if production is not None else fitted[selected],
                 "evaluated_model": fitted[selected], "model_name": selected,
                 "interval_half_widths": prod_q, "prediction_supported": gate,
                 "sklearn_version": sklearn.__version__}, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")

    print(f"\nSelected: {selected}. Gain vs {strongest_test_baseline}: "
          f"{vs[strongest_test_baseline]['rmse_gain']:+.2f} bu/ac "
          f"(95% CI {vs[strongest_test_baseline]['ci_low']:+.2f}..{vs[strongest_test_baseline]['ci_high']:+.2f})")
    print(f"Intervals: {cov}")
    print(f"Geographic: {geo_overall}")
    print(f"Rolling refit: model {roll_m['rmse']:.2f} vs trend {roll_trend['rmse']:.2f}; "
          f"gain CI {roll_vs_trend['ci_low']:+.2f}..{roll_vs_trend['ci_high']:+.2f}")
    print(f"Prediction supported: {gate}. Runtime {out['runtime_seconds']} s")
    return out


def _full_params(cand, result) -> dict:
    """Map short parameter names in the report back to full estimator keys."""
    _, grid = cand
    keys = grid[0].keys() if grid and grid[0] else []
    return {k: result["params"][k.split("__")[-1]] for k in keys}


if __name__ == "__main__":
    run()
