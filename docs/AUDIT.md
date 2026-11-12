# Starting-state audit (2026-11-12)

This records what the project folder held before this rebuild and the problems found.
The original files were not modified. They stay in the folder and are excluded from
version control by `.gitignore`.

## Files found

| File | Size | SHA-256 (first 16) | What it is |
|---|---|---|---|
| `final_dataset_crop_yield.csv` | 837,390 B | `d3fbc283f6a9d3eb` | 5,000 rows × 22 columns |
| `Smart_Crop_Yield_Analysis_and_Modeling.ipynb` | 2,284,046 B | `501b955ac81292fc` | 61-cell analysis notebook with saved outputs |
| `1.jpg` | 201,490 B | `3ecdb77dff7fc60b` | Screenshot of the notebook's "Numerical feature distributions" figure |
| `2.jpg` | 118,311 B | `2b558f498b2f5b64` | Screenshot of "Yield by categorical feature" figure |
| `3.jpg` | 50,499 B | `1b46d2e6738e9f0f` | Screenshot of "Training fit versus test performance" figure |
| `4.jpg` | 76,394 B | `115c03895698ce88` | Screenshot of "SHAP summary (XGBoost)" figure |

The folder had no Git history, README, licence, dependency file, tests or app. The
images carry no EXIF or author metadata. A search for secrets (API keys, tokens,
passwords) and personal data found none. The word "hospital" appears in none of the files.

## Notebook methodology: problems

1. **Row-level split on duplicated data (fatal).** `train_test_split` and `KFold(shuffle=True)`
   ran on all 5,000 rows, and each distinct row appears 10 times. Almost every test row
   had identical copies in training, so the reported R² ≈ 1.000 (Extra Trees, HGB,
   LightGBM, RF) measures memorisation. The notebook's own audit cell printed
   "Exact duplicate rows: 4500" but did not act on it.
2. **Leakage and timing.** Features included `harvest_month`, `total_days`,
   `growing_days_calc` and `duration_gap`, which are only known at harvest, plus season
   totals and means (rainfall, temperature, humidity, sunlight) and in-season NDVI and
   disease status. The notebook never stated when a prediction would be made.
3. **Contradictory evidence ignored.** On the 500 unique farms, the notebook's own
   correlation and ANOVA tables show no relationship between any feature and yield
   (max |r| = 0.077, all ANOVA p ≥ 0.18). The KS test it ran found yield consistent with
   Uniform(2000, 6000). These results contradict an R² of 1.0.
4. **Tuning on the full data.** `RandomizedSearchCV` was fitted on all rows, test included.
5. **Mislabelled outputs.** The permutation-importance plot is titled "held-out farms",
   but the test rows were not held-out farms.
6. **Stale or inconsistent outputs.** Cell 6 prints `Unique farm_id: 5`, but cell 9
   reports 500. Saved file names in the outputs (`14_feature_importance.csv`) differ from
   the code (`feature_importance`). The kernel metadata says Python 3.12.15, but the
   versions table says 3.13.18. The outputs were therefore not all produced by the code
   as saved.
7. **Environment.** It ran on Google Colab (`/content` paths). It downloaded the CSV from
   a Google Drive file ID and installed `catboost` inline. There was no dependency file.

## Dataset: problems

See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) for source and rights, and
`reports/data_audit.json` for the computed checks.

- 500 distinct records, each repeated exactly 10 times in the same order (rows
  1-500 = rows 501-1000 = …).
- `farm_id` and `sensor_id` share the same number (FARM0001 ↔ SENS0001).
- Coordinates contradict region labels. Every region spans lat 10-35, lon 70-90 E (South
  Asia). No "Central USA", "South USA" or "East Africa" farm lies in its real region.
- All sowing dates fall between 1 Jan and 28 Mar 2024, the same window for every crop and region.
- 10 of 11 numeric columns are statistically consistent with uniform distributions.
  Rainfall is the exception (KS p = 0.006).
- Yield ≈ Uniform(2000, 6000) kg/ha for every crop, cotton included.
- Units are under-specified: pesticide is "ml per day" with no area basis.
- No missing values. `total_days` always equals harvest − sowing, and the timestamp always
  falls within the season.
