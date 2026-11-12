# Ownership, attribution and what is excluded

## Material in the original folder

| Item | Who made it | Evidence | In the repository? |
|---|---|---|---|
| `Smart_Crop_Yield_Analysis_and_Modeling.ipynb` | **Unknown**: the owner and a friend worked on it, and their exact contributions are undocumented | No author metadata. Colab notebook (`/content` paths, empty `colab.provenance`). Downloads data from an unrecorded Google Drive file ID. | **No.** Excluded, ownership unclear |
| `1.jpg`–`4.jpg` | Screenshots of that notebook's output figures | Content matches notebook cells 15, 24, 43, 57 | **No.** Derived from the notebook above |
| `final_dataset_crop_yield.csv` | Very likely a 10× duplicated copy of a third-party Kaggle dataset (Atharva Soundankar, Apache-2.0) | See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) | **No.** Unverified, modified copy |

These files remain in the local folder, untouched, and are listed in `.gitignore`.

## Material created during this rebuild (2026-11-12)

Everything else in the repository was written new during this session: the
`harvestmetric/` package, `streamlit_app.py`, `tests/`, `docs/`, `reports/`, and the
screenshots in `docs/screenshots/` (captured from the new app).

- No code was copied from the original notebook. The new code was written from scratch
  against the dataset's documented schema.
- **Ideas shared with the original notebook,** stated openly: the dataset choice; standard
  scikit-learn building blocks (pipelines, one-hot encoding, scaling, random forest and
  gradient boosting); and two checks the notebook also ran, counting duplicate rows and
  a KS test of yield against a uniform distribution. The notebook ran these checks but
  did not act on them. The rebuild acts on them, and its methodology differs throughout
  (de-duplication, defined prediction point, grouped and nested evaluation,
  permutation test, prediction gate).
- The work was carried out with an AI coding assistant at the repository owner's
  direction. See [CV.md](CV.md) for how to describe this honestly.

## Third-party material and credits

| Material | Licence | What is required |
|---|---|---|
| Smart Farming Sensor Data for Yield Prediction (Atharva Soundankar, Kaggle) | Apache 2.0 | Not redistributed by default. If you commit the CSV, keep `NOTICE` and the attribution in the README. |
| Python libraries (numpy, pandas, scipy, scikit-learn, joblib, plotly, streamlit, pytest) | BSD / Apache 2.0 / MIT | Installed from PyPI, not vendored; nothing to bundle |

## To include the original notebook later

Include it only if you and your friend agree in writing (an email is enough) on who wrote
what and on publishing it under this repository's licence. Then credit your friend by name
in the README. Even then, publish it only with a note that its R² results are invalid.
