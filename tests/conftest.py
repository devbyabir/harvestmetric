"""Shared fixtures. Tests use a small generated frame so they run without the real dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harvestmetric import config


def make_frame(n_farms: int = 60, copies: int = 1, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, n_farms + 1):
        sow = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 80)))
        days = int(rng.integers(90, 151))
        harvest = sow + pd.Timedelta(days=days)
        stamp = sow + pd.Timedelta(days=int(rng.integers(0, days)))
        row = {"farm_id": f"FARM{i:04d}", "sensor_id": f"SENS{i:04d}"}
        for col, levels in config.CATEGORIES.items():
            row[col] = levels[i % len(levels)] if col == "crop_type" else rng.choice(levels)
        for col, (lo, hi) in config.NUMERIC_RANGES.items():
            if col != "total_days":
                row[col] = round(float(rng.uniform(lo, hi)), 2)
        row.update(total_days=days, sowing_date=sow.strftime("%d/%m/%Y"),
                   harvest_date=harvest.strftime("%d/%m/%Y"), timestamp=stamp.strftime("%d/%m/%Y"))
        rows.append(row)
    df = pd.DataFrame(rows)[config.EXPECTED_COLUMNS]
    return pd.concat([df] * copies, ignore_index=True)


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture
def csv_with_duplicates(tmp_path):
    path = tmp_path / "dup.csv"
    make_frame(n_farms=50, copies=10).to_csv(path, index=False)
    return path


@pytest.fixture
def valid_record() -> dict:
    return {"region": "North India", "crop_type": "Wheat", "irrigation_type": "Drip",
            "fertilizer_type": "Organic", "soil_pH": 6.5, "latitude": 25.0,
            "longitude": 80.0, "sowing_month": 2}
