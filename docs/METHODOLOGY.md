# Methodology and results

All numbers below come from `reports/metrics.json`, produced by
`python -m harvestmetric.train` (scikit-learn 1.9.1, seed 42).

## 1. Prediction task

**When:** on the sowing date, before any in-season measurement exists.

**Inputs allowed (at-sowing set):** region, crop type, irrigation type, fertiliser type,
soil pH, latitude, longitude, sowing month.

**Excluded, with reasons:**

| Column(s) | Reason |
|---|---|
| `harvest_date`, `total_days` | Known only at harvest |
| `rainfall_mm`, `temperature_C`, `humidity_%`, `sunlight_hours` | Totals or means "during the crop cycle", complete only at harvest |
| `NDVI_index`, `crop_disease_status`, `soil_moisture_%`, `pesticide_usage_ml` | In-season observations (the snapshot `timestamp` is random within the season) |
| `timestamp` | Random in-season snapshot date |
| `farm_id`, `sensor_id` | Identifiers |

A second **full-season** feature set adds the in-season and season-aggregate columns.
It is a *diagnostic upper bound*, not a forecast: if even this set has no signal, the
data cannot support yield prediction. The forbidden-column list is enforced in code
(`harvestmetric/features.py`) and by tests.

## 2. Evaluation protocol

1. Drop exact duplicate rows (5,000 → 500, one row per farm). Sort by `farm_id`.
2. **Untouched test set:** 100 farms (20%), chosen with `StratifiedGroupKFold`
   (grouped by farm, stratified by crop). Used once, after model selection.
3. **Tuning and selection on the 400 training farms only.** Each model's generalisation is
   estimated by nested CV: 5 outer farm-grouped folds, with grid-search tuning inside
   each outer training fold.
4. **Models:** mean baseline, crop-mean baseline, Ridge, random forest, histogram
   gradient boosting. Small grids (see `harvestmetric/models.py`).
5. **Selection rule:** lowest nested-CV RMSE among learned models on the at-sowing task.
6. **Is it better than chance?** A label-permutation test (200 permutations, grouped CV
   on training farms), a bootstrap 95% CI for test R², and a paired bootstrap of the
   RMSE reduction versus the best baseline.
7. **Prediction gate:** the app offers predictions only if permutation p < 0.05 **and**
   the test-R² CI lower bound > 0 **and** the RMSE-reduction CI lower bound > 0.

Preprocessing (one-hot with fixed category lists, standard scaling) lives inside the
sklearn `Pipeline`, so CV, final training and inference apply identical transformations.

## 3. Results

### At-sowing task (the forecast)

| Model | Nested CV R² | CV RMSE | Test R² | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|---:|
| Mean baseline | −0.010 | 1,175 | −0.001 | 1,013 | 1,175 |
| Crop-mean baseline | −0.017 | 1,178 | −0.011 | 1,011 | 1,181 |
| **Ridge (selected, α = 1000)** | −0.018 | 1,179 | **−0.006** | 1,013 | 1,178 |
| Random forest | −0.063 | 1,204 | −0.031 | 1,021 | 1,192 |
| Gradient boosting | −0.057 | 1,201 | −0.015 | 1,008 | 1,183 |

Units: kg/ha. Test set: 100 farms.

- Test R² 95% CI for Ridge: **−0.060 to +0.001**.
- Permutation test: CV R² −0.016; the 95th percentile of the shuffled-label R² is −0.003,
  so **p = 0.55**.
- RMSE change vs mean baseline: −3.1 kg/ha (95% CI −7.5 to +1.4), i.e. not better.
- **Gate: not passed. The app shows no prediction form.**

### Full-season diagnostic

The best nested-CV R² is −0.021 (Ridge), and every learned model scores below the mean
baseline in CV. On the test set, gradient boosting reaches R² +0.017 and random forest
+0.004. Both have negative CV scores, so these are within noise. Even with in-season
information there is no usable signal.

### Crop-wise test errors (selected model)

| Crop | n | MAE | RMSE | R² | Bias (pred − actual) |
|---|---:|---:|---:|---:|---:|
| Cotton | 21 | 1,256 | 1,397 | −0.04 | +226 |
| Maize | 22 | 929 | 1,106 | −0.12 | −362 |
| Rice | 17 | 1,136 | 1,270 | −0.02 | +121 |
| Soybean | 21 | 937 | 1,071 | −0.03 | −182 |
| Wheat | 19 | 818 | 1,008 | −0.01 | +113 |

### Failure cases

Predictions span only 3,970-4,087 kg/ha (sd 23) while actual yields span 2,000-6,000
(sd 1,174). The model predicts the average for everyone:

| Actual yield band | n | MAE | Bias |
|---|---:|---:|---:|
| 2,000-3,000 | 23 | 1,555 | +1,555 (over-predicted) |
| 3,000-4,000 | 24 | 563 | +563 |
| 4,000-5,000 | 25 | 401 | −401 |
| 5,000-6,000 | 28 | 1,502 | −1,502 (under-predicted) |

The farms the model gets most wrong are exactly the low- and high-yield farms a
forecasting tool would exist to flag.

## 4. Why the original R² was invalid

Same model type (random forest, full-season features), same 5,000-row file, two splits:

| Split | Mean 5-fold R² |
|---|---:|
| Row-level `KFold` (as in the original notebook) | **0.99997** |
| Grouped by farm | **−0.206** |

The only difference is whether copies of the same farm can appear on both sides of the
split.

## 5. Conclusion

The data cannot support a defensible yield forecast. The project is therefore delivered
as a **crop-yield analytics** project: a data audit, a corrected evaluation, and an
interactive demo that explains these results. The inference code, input validation and
prediction gate are in place, so a dataset with real signal could be dropped in and
judged by the same protocol.
