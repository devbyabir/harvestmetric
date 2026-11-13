<div align="center">

# 🌾 HarvestMetric

**Leakage-safe evaluation of crop-yield models: exposing an R² of 1.00, then testing a real forecast honestly**

<a href="https://harvestmetric.streamlit.app"><img src="https://img.shields.io/badge/%E2%96%B6%20Open%20live%20demo-harvestmetric.streamlit.app-2f7d4f?style=for-the-badge&logo=streamlit&logoColor=white" alt="Open the live demo"></a>
&nbsp;
<a href="docs/report/HarvestMetric_technical_report.pdf"><img src="https://img.shields.io/badge/Technical%20report-PDF%2C%204%20pages-555555?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Read the technical report"></a>

[![Tests](https://github.com/devbyabir/harvestmetric/actions/workflows/tests.yml/badge.svg)](https://github.com/devbyabir/harvestmetric/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/github/license/devbyabir/harvestmetric)](LICENSE)

<sub>The demo sleeps when nobody has used it for a while. If you see a "wake up" button, click it and give it about a minute.</sub>

</div>

---

## In 30 seconds

| | Question | Answer |
|---|---|---|
| **Part A** · synthetic data | An internship notebook reported **R² ≈ 1.00**. Was it real? | **No.** Every record appeared 10 times, so test rows had copies in training. Split by farm, the same model scores **R² −0.21**, and no model beats the mean. |
| **Part B** · real data | Can season-to-date weather improve a **1 August** forecast of Iowa county corn yield? | **Probably, but not proven.** On untouched years 2012–2025, RMSE fell from 21.8 to **17.6 bu/acre** (−19%) against a county-trend baseline. The year-level 95% CI of the gain, [−0.00, +9.46], reaches zero, so no live forecasts are published. |

The point of the project is the evaluation: defining what is known on the forecast date, holding out
whole farms, years and districts, comparing against strong baselines, and reporting a result that did not
clear its own bar.

<p align="center"><img src="docs/screenshots/01_overview.png" alt="HarvestMetric app overview" width="900"></p>

## Results

| | Model | Strongest baseline | Verdict |
|---|---|---|---|
| **A** Same model, two splits | Row-level KFold R² **1.000** | Grouped by farm R² **−0.206** | The original score was leakage |
| **A** 100 held-out farms | Ridge, R² −0.006 (95% CI −0.060 … +0.001) | Mean, R² −0.001 | No skill (permutation p = 0.55) |
| **B** Test years 2012–2025 | Trend + gradient boosting, RMSE **17.6** | County trend, RMSE 21.8 | Gain +4.2 bu/ac, 95% CI [−0.00, +9.46]: **not established** |
| **B** Held-out districts, 1981–2011 | RMSE **16.7** (tuned inside each fold) | County trend, RMSE 22.8 | Better in **9 of 9** districts |
| **B** 80% / 90% prediction intervals | Coverage 86% / 94% | — | Empirical, no guarantee; 68–69% at 80% in 2012 and 2020 |

Full numbers by year, district and weather group: [Part B methods and results](docs/PART_B_METHODS_RESULTS.md).

## How the evaluation guards against fooling itself

| Safeguard | Part A | Part B |
|---|---|---|
| **Forecast moment** | At sowing | 1 August: weather up to 31 July, the county's yields up to last year |
| **Held-out data** | Whole farms | Years 2012–2025, scored once after model selection |
| **Tuning** | Grouped CV on training farms | Expanding-window CV inside 1981–2011 |
| **Baselines** | Mean, crop mean | Historical mean, previous year, county trend |
| **Uncertainty** | Bootstrap CIs, permutation test | Bootstrap that resamples whole *years*; empirical intervals checked per year |
| **Spatial check** | — | Leave one of 9 districts out, with tuning redone inside each fold |
| **Leakage tests** | Duplicates cannot cross the split | Injected future weather or yields cannot reach a feature |

**Acceptance rule.** Forecasts are shown only if the model beats the best baseline in CV *and* its test
gain has a 95% CI above zero. Part B's model passed neither, and the app says so. The rule was written
into the code before the test scores were computed, by the author's account; it is not an independently
dated preregistration.

## Explore the app

| Leakage case study | Forecast evaluation |
|---|---|
| ![Leakage case study](docs/screenshots/02_case_study.png) | ![Forecast evaluation](docs/screenshots/05_evaluation.png) |
| **Explore Iowa data** | **Error analysis and backtest** |
| ![Explore Iowa data](docs/screenshots/04_explore.png) | ![Error analysis](docs/screenshots/06_errors.png) |

The app has eight pages: an overview, the Part A case study and a live leakage simulator, and four Part B
pages (data explorer, evaluation, error analysis, scenarios), plus data and limitations.

## Data

All sources are public. Details, queries and checksums: [data provenance](docs/DATA_PROVENANCE.md).

| Part | Source | Terms | In this repository |
|---|---|---|---|
| A | *Smart Farming Sensor Data for Yield Prediction*, A. Soundankar (Kaggle). **Synthetic.** | Apache-2.0 | No: aggregate results only |
| B | USDA NASS Quick Stats: Iowa county corn-for-grain yield, bu/acre | CC0 1.0 | Processed panel |
| B | NASA POWER daily point data v2.10.3 (T2M_MAX, T2M_MIN, PRECTOTCORR) | Citation required | Derived features |
| B | US Census Bureau 2024 Gazetteer county points | US government work | Coordinates |

<sub>This product uses NASS data but is not endorsed or certified by NASS. These data were obtained from
the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/Applied
Science Program.</sub>

## Run it locally

Requires Python 3.11 or newer (CI runs 3.12 and 3.13).

```bash
git clone https://github.com/devbyabir/harvestmetric.git
```

```bash
cd harvestmetric
```

```bash
python -m venv .venv
```

```bash
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
```

```bash
streamlit run streamlit_app.py
```

The app needs nothing outside the repository: the processed panel and all reports are committed.

### Reproduce every number

| Step | Command |
|---|---|
| Part A: download and verify the official file | `python -m harvestmetric.synthetic.acquire` |
| Part A: audit and evaluation | `python -m harvestmetric.synthetic.audit` then `python -m harvestmetric.synthetic.train` |
| Part B: download NASS, Census and NASA POWER | `python -m harvestmetric.real.acquire` |
| Part B: build the panel and evaluate (~1 min) | `python -m harvestmetric.real.prepare` then `python -m harvestmetric.real.train` |
| Checks | `python -m ruff check .` · `python -m mypy` · `python -m pytest` |
| Technical report | `pip install -r requirements-report.txt` then `python docs/report/build_report.py` |

CI runs lint, type checks, 88 tests, a dependency audit and a package-build check on every push. It also
re-trains Part B from the committed panel and fails if the metrics drift. Versions, seeds and checksums:
[reproducibility](docs/REPRODUCIBILITY.md).

## Project structure

```
harvestmetric/
├── synthetic/      Part A: acquire, audit, provenance, splits, models, train, inference, simulator
├── real/           Part B: acquire, prepare, splits, models, train, intervals, forecast, analysis
├── metrics.py      shared metrics and bootstrap
├── app_data.py     the app's only file access (cached)
└── ui.py           one function per app page
streamlit_app.py, app_pages/   app entry point and routing
tests/              leakage, splits, joins, units, inference, every app page
data/processed/     Part B panel and provenance manifest
reports/            Part A and Part B metrics and tables
docs/               methods, model card, provenance, architecture, report
```

How the pieces fit together: [architecture](docs/ARCHITECTURE.md).

## Documentation

| Document | What it covers |
|---|---|
| [Technical report (PDF)](docs/report/HarvestMetric_technical_report.pdf) | Both parts in four pages |
| [Part A: leakage case study](docs/PART_A_CASE_STUDY.md) | Why R² ≈ 1.00 was invalid, and the corrected evaluation |
| [Part B: methods and results](docs/PART_B_METHODS_RESULTS.md) | Task, validation design, results, error analysis |
| [Model card](docs/MODEL_CARD.md) | Intended use, performance, failure modes |
| [Data provenance](docs/DATA_PROVENANCE.md) | Sources, licences, queries, checksums |
| [Architecture](docs/ARCHITECTURE.md) | Pipelines, modules, design decisions |
| [Reproducibility](docs/REPRODUCIBILITY.md) · [Deployment](docs/DEPLOYMENT.md) | Rebuilding results; hosting the app |
| [Starting-state audit](docs/AUDIT.md) · [Attribution](docs/ATTRIBUTION.md) | What the original notebook did; what is and is not included |
| [Security](SECURITY.md) · [Changelog](CHANGELOG.md) | Security design; release notes |

## Limitations

- Weather comes from one grid point per county, about 0.5° × 0.625°; neighbouring counties often share it.
- NASS values are current estimates and may differ from what was published in real time.
- Counties NASS stopped publishing after 2014 are missing, not at random.
- Fourteen test years give wide year-level uncertainty.
- One state and one crop.
- Part A is synthetic and says nothing about real crops.

## Citation, licence and credits

- **Cite:** see [CITATION.cff](CITATION.cff), or use GitHub's "Cite this repository" button.
- **Licence:** code and documentation are MIT, © 2026 Abir Ahmmed ([LICENSE](LICENSE)). Third-party data
  terms: [NOTICE](NOTICE).
- **Origins:** a rebuild of an exploratory notebook developed during an internship with a colleague. That
  notebook is not included ([attribution](docs/ATTRIBUTION.md)).
