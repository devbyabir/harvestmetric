# Deploying to Streamlit Community Cloud

**Status: deployed** at https://harvestmetric.streamlit.app from `main` (Python 3.12, `streamlit_app.py`).
Every push to `main` redeploys it. All 8 pages were checked on the live app with no errors.

## Already in place
- **Entry point:** `streamlit_app.py` at the repository root, with pages in `app_pages/` and their code in
  `harvestmetric/ui.py`.
- **Dependencies:** `requirements.txt`, pinned; needs Python ≥ 3.11.
- **Settings:** `.streamlit/config.toml` (theme, headless).
- **Data:** everything the app shows is committed. Part B is the public-domain panel
  (`data/processed/iowa_corn_county_panel.csv`) plus `reports/real/`. Part A uses `reports/synthetic/` and a
  simulator that generates its own data. **No local or git-ignored file is needed.**
- **Tests:** the app tests run every page with the committed data, with a small fixture and with no data.
  CI runs them in a clean environment.

## Steps (your accounts)
1. Review and merge the pull request into `main` on https://github.com/devbyabir/harvestmetric.
2. The repository is public, so Streamlit needs access to public repositories only.
3. Go to https://share.streamlit.io, sign in with GitHub, and click **Create app**.
4. Repository `devbyabir/harvestmetric`, branch `main`, main file `streamlit_app.py`. Under **Advanced
   settings**, choose **Python 3.12** (or 3.13).
5. Deploy. Open every page in the sidebar and check that none shows an error.
6. Then add the real URL to the README.

## Expected behaviour
- The forecast page is **not** shown, because Part B's gate did not pass.
- The Part B pages show evaluation results and a backtest viewer, clearly labelled as not live forecasts.
