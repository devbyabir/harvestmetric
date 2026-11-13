"""File access for the Streamlit app.

Everything the app needs is committed to the repository: Part A aggregate
reports, and Part B's processed public-domain panel plus reports. Paths can be
overridden with environment variables (used by tests to run the app against
small fixtures or against a missing-data configuration).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .paths import DATA_DIR, REPORTS_DIR


def reports_root() -> Path:
    return Path(os.environ.get("HARVESTMETRIC_REPORTS", REPORTS_DIR))


def panel_path() -> Path:
    return Path(os.environ.get("HARVESTMETRIC_PANEL", DATA_DIR / "processed" / "iowa_corn_county_panel.csv"))


# Committed files are read once per (path, modification time) and then served from
# memory, so a page rerun does not re-parse the 680 KB panel. Callers get copies.
@lru_cache(maxsize=64)
def _csv(path: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"fips": str, "district_code": str})


@lru_cache(maxsize=16)
def _json(path: str, mtime_ns: int) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cached_csv(path: Path) -> pd.DataFrame | None:
    return _csv(str(path), path.stat().st_mtime_ns).copy() if path.exists() else None


def read_json(part: str, name: str) -> dict | None:
    path = reports_root() / part / name
    return json.loads(json.dumps(_json(str(path), path.stat().st_mtime_ns))) if path.exists() else None


def read_table(part: str, name: str) -> pd.DataFrame | None:
    return _cached_csv(reports_root() / part / "tables" / name)


def read_panel() -> pd.DataFrame | None:
    return _cached_csv(panel_path())
