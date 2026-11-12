# Reproducibility

## Environment
- Developed and verified on **Python 3.14.7, Windows 11**. CI also runs Python 3.12 and 3.13 on Ubuntu.
- `requirements.txt` pins the runtime (numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, scikit-learn 1.9.1,
  joblib 1.6.0, plotly 7.1.0, streamlit 1.67.0).
- `requirements-dev.txt` adds pytest, ruff, mypy and pandas-stubs; `requirements-lock.txt` freezes the full
  environment. `requirements-report.txt` adds matplotlib and playwright, needed only to rebuild the PDF.
- Random seed: 42 everywhere (`RANDOM_STATE` in each part's `config.py`).

## Commands and typical runtime (laptop)
| Step | Command | Output | Time |
|---|---|---|---|
| Install | `pip install -r requirements-dev.txt` | | 1–3 min |
| Part A data | `python -m harvestmetric.synthetic.acquire` | `data/raw/synthetic/` (git-ignored) | seconds |
| Part A audit | `python -m harvestmetric.synthetic.audit` | `reports/synthetic/data_audit.json` | ~2 s |
| Part A evaluation | `python -m harvestmetric.synthetic.train` | `reports/synthetic/` | ~40 s |
| Part B data | `python -m harvestmetric.real.acquire` | `data/raw/real/` (git-ignored), `data/processed/manifest.json` | ~3 min (99 POWER requests, cached) |
| Part B panel | `python -m harvestmetric.real.prepare` | `data/processed/iowa_corn_county_panel.csv` | ~15 s |
| Part B evaluation | `python -m harvestmetric.real.train` | `reports/real/`, `artifacts/real/` | ~10 s |
| Checks | `python -m ruff check . && python -m mypy && python -m pytest` | | ~1 min |
| Report | `python docs/report/build_report.py` | `docs/report/*.pdf` | ~10 s |
| App | `streamlit run streamlit_app.py` | http://localhost:8501 | |

## Data versions
- NASS: retrieved 2026-11-12. Query UUID, URL and SHA-256 are in `data/processed/manifest.json`. NASS
  revises estimates, so a later download can differ; the manifest checksum identifies the version used.
- NASA POWER: API v2.10.3, retrieved 2026-11-12; a per-county SHA-256 is in the manifest.
- Census Gazetteer 2024 counties: zip SHA-256 in the manifest.
- Kaggle (Part A): `Smart_Farming_Crop_Yield_2024.csv`, SHA-256 `49b1f245…7940d`, checked by the acquire script.

## What CI verifies
On every pull request (`.github/workflows/tests.yml`), in a clean environment:
- ruff, mypy, and the full test suite;
- the app tests, covering every page with the committed data, with a small fixture panel, and with no data;
- that `python -m harvestmetric.real.train` reproduces the committed Part B metrics from the committed panel
  (within 0.05 bu/ac).

## Known non-determinism
Random-forest results in Part A can differ in the last floating-point digits between runs, because of
parallel tree building. Reported values are unaffected at the precision shown.
