"""File access for the Streamlit app.

Everything the app needs is committed to the repository: Part A aggregate
reports, and Part B's processed public-domain panel plus reports. Paths can be
overridden with environment variables (used by tests to run the app against
small fixtures or against a missing-data configuration).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .paths import DATA_DIR, REPORTS_DIR


def reports_root() -> Path:
    return Path(os.environ.get("HARVESTMETRIC_REPORTS", REPORTS_DIR))


def panel_path() -> Path:
    return Path(os.environ.get("HARVESTMETRIC_PANEL", DATA_DIR / "processed" / "iowa_corn_county_panel.csv"))


def read_json(part: str, name: str) -> dict | None:
    path = reports_root() / part / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_table(part: str, name: str) -> pd.DataFrame | None:
    path = reports_root() / part / "tables" / name
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"fips": str, "district_code": str})


def read_panel() -> pd.DataFrame | None:
    path = panel_path()
    return pd.read_csv(path, dtype={"fips": str, "district_code": str}) if path.exists() else None
