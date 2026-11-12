"""Smoke tests for the Streamlit app using Streamlit's built-in AppTest harness."""

from pathlib import Path

import pytest

from harvestmetric import config

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def test_app_runs_without_raw_data(monkeypatch, tmp_path):
    monkeypatch.setenv("HARVESTMETRIC_DATA", str(tmp_path / "missing.csv"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Raw dataset not found" in w.value for w in at.warning)


@pytest.mark.skipif(not config.default_data_path().exists(), reason="dataset not available locally")
def test_app_runs_with_data_and_shows_no_prediction_form_when_gate_fails():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    if config.METRICS_PATH.exists():
        import json
        supported = json.loads(config.METRICS_PATH.read_text())["selection"]["prediction_supported"]
        assert (len(at.get("form")) > 0) == supported
