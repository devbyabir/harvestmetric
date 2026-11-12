"""Project-wide constants: paths, column schema, documented value ranges."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METRICS_PATH = REPORTS_DIR / "metrics.json"
AUDIT_PATH = REPORTS_DIR / "data_audit.json"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"

# Name of the file as published on Kaggle (see docs/DATA_PROVENANCE.md).
KAGGLE_FILENAME = "Smart_Farming_Crop_Yield_2024.csv"
# File that existed in the original working folder (10x duplicated copy).
LEGACY_FILENAME = "final_dataset_crop_yield.csv"

RANDOM_STATE = 42
TARGET = "yield_kg_per_hectare"
GROUP_COL = "farm_id"

CATEGORIES: dict[str, list[str]] = {
    "region": ["Central USA", "East Africa", "North India", "South India", "South USA"],
    "crop_type": ["Cotton", "Maize", "Rice", "Soybean", "Wheat"],
    "irrigation_type": ["Drip", "Manual", "None", "Sprinkler"],
    "fertilizer_type": ["Inorganic", "Mixed", "Organic"],
    "crop_disease_status": ["Mild", "Moderate", "None", "Severe"],
}

# Allowed ranges. They follow the dataset documentation where it states one
# (soil pH, lat/lon, NDVI) and otherwise the observed generator bounds.
# Values outside these ranges are outside the data the model has seen.
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "soil_moisture_%": (10.0, 45.0),
    "soil_pH": (5.5, 7.5),
    "temperature_C": (15.0, 35.0),
    "rainfall_mm": (50.0, 300.0),
    "humidity_%": (40.0, 90.0),
    "sunlight_hours": (4.0, 10.0),
    "pesticide_usage_ml": (5.0, 50.0),
    "total_days": (90, 150),
    "latitude": (10.0, 35.0),
    "longitude": (70.0, 90.0),
    "NDVI_index": (0.3, 0.9),
    TARGET: (2000.0, 6000.0),
}

UNITS: dict[str, str] = {
    "soil_moisture_%": "% volumetric",
    "soil_pH": "pH",
    "temperature_C": "°C (season mean)",
    "rainfall_mm": "mm (season total)",
    "humidity_%": "% (season mean)",
    "sunlight_hours": "h/day (season mean)",
    "pesticide_usage_ml": "ml/day (area not stated)",
    "total_days": "days",
    "latitude": "° N",
    "longitude": "° E",
    "NDVI_index": "index",
    "sowing_month": "month (1-12)",
    TARGET: "kg/ha",
}

DATE_COLS = ["sowing_date", "harvest_date", "timestamp"]
ID_COLS = ["farm_id", "sensor_id"]

EXPECTED_COLUMNS = [
    "farm_id", "region", "crop_type", "soil_moisture_%", "soil_pH", "temperature_C",
    "rainfall_mm", "humidity_%", "sunlight_hours", "irrigation_type", "fertilizer_type",
    "pesticide_usage_ml", "sowing_date", "harvest_date", "total_days", TARGET,
    "sensor_id", "timestamp", "latitude", "longitude", "NDVI_index", "crop_disease_status",
]


def default_data_path() -> Path:
    """Resolve the dataset path: $HARVESTMETRIC_DATA, then Kaggle file, then legacy copy."""
    env = os.environ.get("HARVESTMETRIC_DATA")
    if env:
        return Path(env)
    for candidate in (DATA_DIR / KAGGLE_FILENAME, PROJECT_ROOT / LEGACY_FILENAME):
        if candidate.exists():
            return candidate
    return DATA_DIR / KAGGLE_FILENAME
