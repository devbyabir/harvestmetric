"""Streamlit app tests (AppTest harness): every page, three data configurations.

1. committed data: what a clean Streamlit Community Cloud deployment sees;
2. a small fixture panel (public-domain subset written by the test) - covers the
   data path independently of the full committed panel;
3. no data at all: every page must degrade to a notice, never an exception.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from harvestmetric.paths import DATA_DIR, REPORTS_DIR

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
PAGES = ["app_pages/case_study.py", "app_pages/simulator.py", "app_pages/explore.py", "app_pages/evaluation.py",
         "app_pages/errors.py", "app_pages/scenarios.py", "app_pages/about.py"]
PANEL = DATA_DIR / "processed" / "iowa_corn_county_panel.csv"


def run_page(page: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120).run()
    if page:
        at.switch_page(page).run()
    return at


@pytest.mark.parametrize("page", [None, *PAGES])
def test_every_page_runs_with_committed_data(page):
    if not PANEL.exists():
        pytest.skip("committed panel missing")
    at = run_page(page)
    assert not at.exception, at.exception
    assert not at.warning, [w.value for w in at.warning]


def test_forecast_page_only_when_gate_passed():
    metrics = json.loads((REPORTS_DIR / "real" / "metrics.json").read_text())
    at = run_page()
    supported = metrics["selection"]["prediction_supported"]
    if not supported:
        assert not (REPORTS_DIR / "real" / "tables" / "forecast_2026.csv").exists()
        with pytest.raises(Exception):
            at.switch_page("app_pages/forecasts.py").run()


@pytest.fixture
def fixture_env(tmp_path, monkeypatch):
    """Small permissible fixture: 6 counties x all years of the public-domain panel + copied reports."""
    panel = pd.read_csv(PANEL, dtype={"fips": str, "district_code": str})
    small = panel[panel.fips.isin(sorted(panel.fips.unique())[:6])]
    path = tmp_path / "mini_panel.csv"
    small.to_csv(path, index=False)
    shutil.copytree(REPORTS_DIR, tmp_path / "reports")
    monkeypatch.setenv("HARVESTMETRIC_PANEL", str(path))
    monkeypatch.setenv("HARVESTMETRIC_REPORTS", str(tmp_path / "reports"))
    return small


@pytest.mark.parametrize("page", ["app_pages/explore.py", "app_pages/scenarios.py", "app_pages/errors.py"])
def test_data_pages_run_with_small_fixture(fixture_env, page):
    at = run_page(page)
    assert not at.exception, at.exception


def test_scenarios_report_too_few_seasons_clearly(fixture_env):
    at = run_page("app_pages/scenarios.py")
    at.slider[0].set_value(0.95).run()          # extreme threshold on a tiny fixture
    at.radio[0].set_value("2012–2025").run()
    assert not at.exception
    assert at.error or at.metric  # either a clear message or a valid comparison, never a crash


@pytest.mark.parametrize("page", [None, *PAGES])
def test_every_page_degrades_without_data(tmp_path, monkeypatch, page):
    monkeypatch.setenv("HARVESTMETRIC_PANEL", str(tmp_path / "missing.csv"))
    monkeypatch.setenv("HARVESTMETRIC_REPORTS", str(tmp_path / "no_reports"))
    at = run_page(page)
    assert not at.exception, at.exception
    if page not in ("app_pages/simulator.py", "app_pages/about.py"):
        assert at.warning, "expected a 'not available' notice"


def test_simulator_shows_leakage_by_default():
    at = run_page("app_pages/simulator.py")
    assert any("Leakage" in e.value for e in at.error)
