# Ownership, attribution and what is excluded

## Three phases of work, kept distinct

| Phase | What | Who | In this repository? |
|---|---|---|---|
| 1. Original internship notebook | `Smart_Crop_Yield_Analysis_and_Modeling.ipynb` and four JPG screenshots of its figures | The repository owner **and a friend**; their split of work is undocumented | **No.** Ownership is unclear, and its R² results are invalid (see Part A) |
| 2. Rebuild, Part A | Leakage audit and corrected evaluation of the synthetic dataset | The repository owner, with an AI coding assistant | Yes |
| 3. Rebuild, Part B | Real-data pipeline, forecasting evaluation, app, report | The repository owner, with an AI coding assistant | Yes |

## Material in the original folder

| Item | Origin | Evidence | Published? |
|---|---|---|---|
| `Smart_Crop_Yield_Analysis_and_Modeling.ipynb` | Phase 1: owner and friend | No author metadata. Colab paths; downloads data from an unrecorded Google Drive file ID | **No** |
| `1.jpg`–`4.jpg` | Screenshots of that notebook's figures (cells 15, 24, 43, 57) | Visual match | **No** |
| `final_dataset_crop_yield.csv` | Derived from the Kaggle dataset by Atharva Soundankar (Apache-2.0): dates reformatted, each record ×10 | Record-level verification, 0 differing cells ([DATA_PROVENANCE.md](DATA_PROVENANCE.md)) | **No.** Modified copy, not needed |

These files stay in the local folder, untouched, and are listed in `.gitignore`.

## Material created during the rebuild
Everything in the repository was written during the rebuild sessions of 12 November 2026:
- the `harvestmetric/` package, `streamlit_app.py`, `app_pages/` and `tests/`;
- `docs/`, `reports/` and `data/processed/`;
- the screenshots (captured from the new app) and the technical report.

- No code was copied from the original notebook.
- Ideas shared with that notebook are stated openly: the synthetic dataset, standard scikit-learn building
  blocks, and two checks it also ran (counting duplicates, a KS test against a uniform distribution) but did
  not act on.
- The work was produced with an AI coding assistant directed and reviewed by the repository owner. See
  [CV.md](CV.md) for honest wording.

## Third-party material and credits

| Material | Terms | Required action |
|---|---|---|
| USDA NASS Quick Stats data (in `data/processed/` and `reports/real/`) | CC0 1.0 | Non-endorsement line, as in the README, app and NOTICE (courtesy; CC0 needs none) |
| NASA POWER data (derived features) | Citation required | Acknowledgement sentence in README, NOTICE, app and report |
| US Census 2024 Gazetteer (coordinates) | US Gov. work | Credit given |
| Kaggle dataset (Part A) | Apache-2.0 | Not redistributed; credited in NOTICE and README |
| Python libraries | BSD / Apache-2.0 / MIT | Installed from PyPI, not vendored |

## To include the original notebook later
Include it only with your friend's written agreement on authorship and on publishing it under this
repository's licence. Credit them by name, and keep a note that its R² ≈ 1.00 is invalid.
