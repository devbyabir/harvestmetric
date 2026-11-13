# Security policy

## Reporting a vulnerability
Please use GitHub's private vulnerability reporting for this repository (**Security → Report a
vulnerability**). Do not open a public issue for a security problem. You can expect a first reply within
7 days.

## Scope and design
HarvestMetric is a research and demonstration project. The public demo is read-only.

| Area | What is in place |
|---|---|
| Secrets | The app needs no API keys or credentials, and none are stored in the repository. Data downloads use keyless public endpoints. |
| User input | The app takes only slider, select and radio values. It never runs user-supplied code, uploads or queries. |
| HTML | `unsafe_allow_html` is used only for fixed CSS and fixed badge labels, never for data or user input. |
| Network | Only the acquisition scripts (`*/acquire.py`) make outbound requests, to fixed HTTPS hosts (USDA NASS, US Census, NASA POWER, Kaggle), with timeouts. The app makes none. |
| Model files | `joblib` files in `artifacts/` are produced locally by the training scripts and are git-ignored. `joblib.load` can execute code, so **never load a model file you did not create yourself**. |
| Dependencies | Runtime versions are pinned in `requirements.txt` (full set in `requirements-lock.txt`). CI audits them with `pip-audit` on every push and pull request. |
| CI permissions | The workflow runs with read-only repository permissions (`contents: read`). |

## Supported versions
Only the latest commit on `main` is supported.
