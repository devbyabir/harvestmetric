# Deploying to Streamlit Community Cloud

**Status:** not deployed. No live URL exists yet.

## What is already set up

- Entry point: `streamlit_app.py` at the repository root
- Dependencies: `requirements.txt`, pinned versions; needs Python ≥ 3.11
- Theme and server settings: `.streamlit/config.toml`
- Evaluation and audit reports are committed in `reports/`, so the Overview, Data audit,
  Model evaluation and Limitations tabs work without the raw CSV.
- Smoke test: `tests/test_app.py` runs the app with and without the dataset.

## Steps (need your accounts)

1. Push the repository to GitHub (see the README, "Publish to GitHub").
2. *Optional, enables the Explore and Scenarios tabs:* complete the steps in
   [DATA_PROVENANCE.md](DATA_PROVENANCE.md) "Steps to clear the data for publication"
   and commit `data/raw/Smart_Farming_Crop_Yield_2024.csv`.
3. Go to https://share.streamlit.io, sign in with GitHub, and click **Create app**.
4. Repository `devbyabir/harvestmetric-crop-yield`, branch `main`, main file
   `streamlit_app.py`. Under **Advanced settings**, choose **Python 3.12** (or 3.13).
5. Deploy. When the app loads, check that every tab opens without an error. Then add the
   URL Streamlit gives you to the README and to [CV.md](CV.md).

Without step 2, the deployed app shows a notice that the raw dataset is absent and
disables the two data-driven tabs. The test-set scatter plot also stays hidden, because
`reports/tables/test_predictions.csv` is not committed.
