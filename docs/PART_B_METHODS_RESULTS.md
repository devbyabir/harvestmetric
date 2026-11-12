# Part B: Iowa county corn yield forecast (methods and results)

All numbers come from [`reports/real/metrics.json`](../reports/real/metrics.json), produced by
`python -m harvestmetric.real.train` (seed 42, scikit-learn 1.9.1, about 9 s).

## 1. Question and task

**Question:** does season-to-date weather improve a 1 August forecast of county corn yield beyond
baselines built from the county's own yield history, and does any improvement hold across years and regions?

| Item | Definition |
|---|---|
| Unit of prediction | Iowa county × year (99 counties) |
| Target | NASS corn-for-grain yield, bu/acre (final published estimate) |
| Forecast date | 1 August of year *t* |
| Information available | The county's published yields for *t*−30 … *t*−1; NASA POWER weather 1 April – 31 July of *t* |
| Training years | 1981–2011 (3,069 county-years) |
| Untouched test years | 2012–2025 (1,282 county-years with a published yield) |

### Features (all computed in `harvestmetric/real/prepare.py`)
| Feature | Meaning |
|---|---|
| `trend_pred` | Linear trend fitted to the county's previous 30 years, extrapolated to *t* (needs ≥ 15 years) |
| `prev_dev`, `prev_missing` | Last year's deviation from that trend; 0 plus a flag when *t*−1 was not published |
| `gdd_may_jul` | Corn growing degree-days, May–July, 10/30 °C method |
| `edd30_jun_jul` | Σ max(T<sub>max</sub> − 30 °C, 0), June–July: extreme heat |
| `prcp_apr_may`, `prcp_jun`, `prcp_jul` | Precipitation totals (mm) |
| `tmax_jul` | Mean July daily maximum temperature (°C) |

### Leakage controls (each has a test in `tests/real/test_pipeline.py`)
- Weather features ignore every day after 31 July, even if absurd values are injected there.
- History features use only years before *t*, so changing year *t* or later leaves them identical.
- No feature uses another county's same-year yield, and identifiers are never inputs.
- Training years end before test years begin, and each expanding-window fold trains only on earlier years.
- Geographic folds hold out whole districts, so no county appears on both sides, and every model choice
  for a fold is made without the held-out district (tested).

## 2. Validation design

1. **Chronological test.** Fit on 1981–2011, score 2012–2025 once, after model selection.
2. **Tuning inside training years.** Expanding-window CV with validation blocks 1997–99, 2000–02, 2003–05,
   2006–08 and 2009–11; each block is fitted only on years before it.
3. **Selection rule.** The learned model with the lowest CV RMSE.
4. **Baselines.**
   - Historical mean: the county's mean over 30 years.
   - Previous year: last year's yield, or the latest published yield if last year was not published.
   - County trend: `trend_pred`.
5. **Models.**
   - Ridge: linear weather effects, stable with about 3,000 rows.
   - Histogram gradient boosting: allows thresholds such as heat above 30 °C and drought, plus interactions.
   - Both predict the deviation from `trend_pred`. Final forecast = trend + predicted deviation.
6. **Uncertainty.**
   - Empirical 80% and 90% intervals: ± an upper quantile of the selected model's 1,485 time-ordered CV
     residuals. The same residuals were used to choose the model, and residuals are correlated within a
     season and shift over time, so these are **not** split-conformal intervals and have **no** coverage
     guarantee. Coverage is checked empirically on the test years, overall and per year.
   - The test gain over each baseline uses a bootstrap that **resamples whole years**, because counties in
     one season share its weather shock.
7. **Acceptance gate** (written into the training script before the test scores were computed, by the author's account; this is not an independently dated preregistration): forecasts are published only if the
   model beats the best baseline in CV **and** its test RMSE gain over the strongest test baseline has a
   95% interval above 0.
8. **Geographic check (separate).** Leave one of NASS's nine crop-reporting districts out, within 1981–2011.
   Inside each fold, the model family and hyperparameters are re-selected by the same expanding-window CV,
   run on the other eight districts only; the chosen model is refitted on them and scored once on the
   held-out district.
9. **Rolling-origin check (secondary).** Refit every test year on all earlier years, with the same hyperparameters.

## 3. Results

### Time-based test (bu/acre)
| Model | CV RMSE | CV R² | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|
| Historical mean | 39.75 | −2.24 | 38.19 | 41.64 | −1.31 |
| Previous year | 19.41 | +0.23 | 19.67 | 25.38 | +0.14 |
| County trend | **18.82** | +0.27 | 16.51 | 21.81 | +0.37 |
| Trend + Ridge (α = 100) | 23.37 | −0.12 | 14.17 | 18.95 | +0.52 |
| **Trend + gradient boosting** (selected; lr 0.03, depth 2, 200 iter) | 19.52 | +0.22 | **13.55** | **17.61** | **+0.59** |

**Gain of the selected model on test years (year-cluster bootstrap, 4,000 resamples, 14 years):**
| Baseline | RMSE gain | 95% CI | Resamples where model better |
|---|---:|---|---:|
| Historical mean | +24.03 | [+16.52, +30.92] | 100% |
| Previous year | +7.77 | [+4.33, +11.32] | 100% |
| County trend | +4.20 | **[−0.00, +9.46]** | 97.5% |

**Gate: not passed.**
- In CV the model was worse than the county trend (19.52 vs 18.82).
- On test, its gain over the trend has an interval that reaches zero.
- So no live forecasts are published. The app shows evaluation results and a backtest viewer only.

**Why CV and test disagree (diagnosed, not tuned away).**
- The first validation block (1997–99) was fitted on 1981–96 only. That period contains three disaster
  years (1983, 1988, 1993) and little else extreme.
- The models then predicted heat losses in 1997, 1999 and 2002, but those seasons ended above trend.
- The weather–yield relationship at the POWER point is not stable across eras. Possible reasons include
  hybrids, management and point-weather error. None was established.
- After seeing the test results I did not change the protocol, features or gate.

**Revision before merge (review of PR #1).** Three corrections were made after the first results:
- "Pre-registered" was removed: the gate has no independently dated preregistration.
- The intervals were relabelled as empirical CV-residual intervals with no conformal guarantee. Their
  numbers are unchanged.
- The geographic check previously reused hyperparameters chosen with all districts, including the held-out
  one. It now re-tunes inside each fold; RMSE moved from 16.83 to 16.68, still 9 of 9 districts.
The chronological test, its baselines, the selected model and the gate are unchanged.

### Rolling-origin annual refit (2012–2025)
RMSE 17.88 vs trend 21.81. Gain 95% CI [−0.39, +9.34]. The same conclusion: probably helpful, not established.

### Prediction intervals (selected model, test years)
| Nominal | Half-width | Coverage on test |
|---|---:|---:|
| 80% | ±24.8 bu/ac | 86.2% |
| 90% | ±32.7 bu/ac | 94.1% |

Per-year coverage of the 80% interval ranges from 68% (2020) and 69% (2012) to 97% (2015). For the 90%
interval it ranges from 80% to 100%. Overall coverage above nominal is not evidence of a guarantee: the
calibration residuals were also used for model selection, errors within a shock year move together, and
the error distribution changes between eras. Calibrating on residuals from
the more volatile 1997–2011 validation blocks probably also makes typical years conservative.

### Geographic check (leave-one-district-out, 1981–2011; reported separately)
Model RMSE 16.68 vs trend 22.80. The model was better in **9 of 9** held-out districts. In-fold tuning chose
gradient boosting in every fold (depth 2, except depth 3 when East Central was held out). The same seasons
appear on both sides, so this measures transfer across space, not skill on new seasons.

## 4. Error analysis (test years)

| Group | Definition | n | Model RMSE | Trend RMSE | Model bias | Trend bias |
|---|---|---:|---:|---:|---:|---:|
| Extreme heat | Jun–Jul EDD above training P90 (119.6) | 200 | 22.2 | 36.5 | +7.5 | +22.3 |
| Dry summer | Jun–Jul rain below training P10 (129 mm) | 90 | 19.9 | 21.9 | −2.7 | +9.5 |
| Wet summer | Jun–Jul rain above training P90 (369 mm) | 82 | 11.9 | 11.5 | 0.0 | −6.0 |
| Typical | none of the above | 910 | 16.7 | 17.8 | −1.1 | −2.9 |
| Shortfall >15% below trend *(outcome-defined, descriptive only)* | | 141 | 31.5 | 48.6 | +24.4 | +46.4 |

- **2012 drought:** bias fell from +45.7 (trend) to +14.7 bu/ac, but the forecast was still too high.
- **2020:** the 10 August derecho hit after the forecast date, giving a bias of +20.2 bu/ac. This error is
  not attributable to the method.
- **Districts:** the largest errors were in South Central (RMSE 22.4) and West Central (19.3).
- **Explanations:** none produced. SHAP or permutation importance would only be appropriate for a validated
  model, and associations here are not causal effects.

## 5. What would change the conclusion
- More test years, or a second state, to tighten the year-level interval.
- Weather weighted by cropland (e.g. USDA Cropland Data Layer) instead of a single point.
- Inputs available by 1 August that are not used here: planting progress, crop-condition ratings, satellite NDVI.
