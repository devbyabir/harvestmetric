# Model card: HarvestMetric Iowa county corn forecast (Part B)

**Status: evaluated, not validated for use.** The model did not pass its pre-registered gate, so no forecasts
are published. This card documents it so the result can be checked and reproduced.

## Model
- **Type:** county linear trend (from the previous 30 years of published yields) plus a histogram
  gradient-boosting regressor on the deviation from that trend (scikit-learn 1.9.1;
  `learning_rate=0.03`, `max_depth=2`, `max_iter=200`, `min_samples_leaf=40`, `random_state=42`).
- **Selected by:** the lowest expanding-window CV RMSE among the two learned candidates (Ridge, gradient boosting).
- **Code:** `harvestmetric/real/models.py` (`TrendDeviationRegressor`). Training and evaluation are in
  `harvestmetric/real/train.py`, inference in `harvestmetric/real/forecast.py`.
- **Artifact:** `artifacts/real/forecast_model.joblib` (not committed; re-created by training).

## Intended use
- A research and teaching example of a leakage-safe, time-separated evaluation of a county-level
  in-season yield forecast issued on 1 August.
- **Not intended** for farm, insurance, trading or policy decisions, nor for any region other than Iowa.

## Inputs (available on 1 August of year *t*)
- The county's published NASS yields for years *t*−30 … *t*−1.
- NASA POWER daily max/min temperature and precipitation from 1 April to 31 July of *t*, at the county's
  Census internal point.

## Training data
- 99 Iowa counties, 1981–2011, giving 3,069 county-years.
- Sources: USDA NASS Quick Stats (CC0), NASA POWER, Census Gazetteer. Details in [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

## Evaluation (untouched test years 2012–2025, 1,282 county-years)
| Metric | Model | County-trend baseline |
|---|---:|---:|
| RMSE (bu/ac) | 17.61 | 21.81 |
| MAE (bu/ac) | 13.55 | 16.51 |
| R² | 0.59 | 0.37 |

- **Gain over the trend:** +4.20 bu/ac RMSE, with a year-cluster bootstrap 95% CI of [−0.00, +9.46].
- **Training-period CV:** RMSE 19.52, against 18.82 for the trend baseline. The model was worse in CV.
- **80% / 90% conformal intervals:** test coverage 86.2% / 94.1%; per-year coverage as low as 68% / 80%.
- **Geographic check:** leave-one-district-out within 1981–2011, RMSE 16.83 vs 22.80 (better in 9 of 9 districts).
- **Pre-registered gate:** not passed.

## Known failure modes
- **Drought years:** the forecast is still too high (2012 bias +14.7 bu/ac).
- **Damage after 1 August:** events such as the 2020 derecho (bias +20.2 bu/ac) are invisible to the model.
- **Large shortfalls** (more than 15% below trend): under-forecast by about 24 bu/ac on average.
- **Unstable relationship:** the weather–yield relationship shifts across eras, which is why CV and test disagree.

## Limitations and ethical considerations
- Point weather from a coarse grid; current (possibly revised) NASS estimates; counties missing after 2014,
  not at random.
- Only 14 test years, so year-level uncertainty dominates.
- Associations learned here are not causal effects of weather or management.
- Mis-used as a confident forecast, it could mislead decisions. That is why forecasts are disabled.

## Reproduce
```bash
python -m harvestmetric.real.acquire && python -m harvestmetric.real.prepare && python -m harvestmetric.real.train
```
