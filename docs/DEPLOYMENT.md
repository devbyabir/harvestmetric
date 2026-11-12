# Deploying to Streamlit Community Cloud

**Status: not deployed.** No live URL exists. Deploying needs your Streamlit account and your decision, and
should happen only after you have reviewed and merged the pull request.

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
2. The repository is private. Streamlit Community Cloud can deploy private repositories once you authorise
   its GitHub app for them; making the repository public is a separate decision.
3. Go to https://share.streamlit.io, sign in with GitHub, and click **Create app**.
4. Repository `devbyabir/harvestmetric`, branch `main`, main file `streamlit_app.py`. Under **Advanced
   settings**, choose **Python 3.12** (or 3.13).
5. Deploy. Open every page in the sidebar and check that none shows an error.
6. Then add the real URL to the README and to [CV.md](CV.md).

## Expected behaviour
- The forecast page is **not** shown, because Part B's gate did not pass.
- The Part B pages show evaluation results and a backtest viewer, clearly labelled as not live forecasts.
