# Changelog

## 1.0.0 (2026-11-14)
First complete release.

### Part A: synthetic leakage case study
- Showed that the internship notebook's R² ≈ 1.00 came from duplicated rows leaking across a row-level
  split. Split by farm, the same model scores R² −0.21, and no model beats the mean.
- Rebuilt from the official Kaggle file, verified record-identical to the original working copy.

### Part B: Iowa county corn yield forecast (real data)
- A 1 August forecast from USDA NASS county yields and NASA POWER weather, trained on 1981–2011 and tested
  once on 2012–2025, against three baselines.
- Gradient boosting lowered test RMSE from 21.8 to 17.6 bu/acre against the county trend. The year-level
  95% CI of the gain, [−0.00, +9.46], reaches zero, and the model lost in training-period CV, so the
  acceptance gate did not pass and no live forecasts are published.
- Geographic check with tuning inside each held-out district fold: RMSE 16.7 vs 22.8, better in 9 of 9
  districts.
- Empirical CV-residual prediction intervals, with no coverage guarantee; coverage checked per test year.

### Methodology corrections made before release
- Removed the "pre-registered" claim: the acceptance rule has no independently dated preregistration.
- Relabelled the intervals as empirical rather than split-conformal.
- Re-tuned the geographic check inside each district fold instead of reusing globally tuned settings.

### Engineering
- Python package with 88 tests, ruff, mypy and CI on Python 3.12 and 3.13. CI also re-trains Part B,
  checks the metrics match, audits dependencies and checks the package build.
- Streamlit app deployed at https://harvestmetric.streamlit.app.
- 4-page technical report, model card, data provenance, reproducibility and architecture notes.
