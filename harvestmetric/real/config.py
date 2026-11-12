"""Part B constants: scope, forecast date, data sources and splits."""

from __future__ import annotations

from ..paths import ARTIFACTS_DIR as _ARTIFACTS
from ..paths import DATA_DIR as _DATA
from ..paths import PROJECT_ROOT
from ..paths import REPORTS_DIR as _REPORTS

__all__ = ["PROJECT_ROOT"]

RAW_DIR = _DATA / "raw" / "real"
POWER_DIR = RAW_DIR / "power"
PROCESSED_DIR = _DATA / "processed"
PANEL_PATH = PROCESSED_DIR / "iowa_corn_county_panel.csv"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"
REPORTS_DIR = _REPORTS / "real"
TABLES_DIR = REPORTS_DIR / "tables"
METRICS_PATH = REPORTS_DIR / "metrics.json"
ARTIFACTS_DIR = _ARTIFACTS / "real"
MODEL_PATH = ARTIFACTS_DIR / "forecast_model.joblib"

RANDOM_STATE = 42

# ---- Scope -----------------------------------------------------------------
STATE_NAME = "IOWA"
STATE_FIPS = "19"
TARGET = "yield_bu_ac"          # corn for grain, bushels per harvested acre (NASS)
BU_AC_TO_T_HA = 0.0627          # 56 lb/bu at 15.5% moisture -> t/ha
FIRST_MODEL_YEAR = 1981         # NASA POWER daily data start
LAST_YIELD_YEAR = 2025          # latest NASS county estimates at retrieval
FORECAST_YEAR = 2026            # season with weather to 31 Jul but no county yields yet
HISTORY_WINDOW = 30             # years of a county's own past yields used for trend/baselines
MIN_HISTORY = 15                # minimum past observations required

# Forecast is issued on 1 August of year t: weather up to 31 July of t and
# published yields up to year t-1 are available. Nothing later.
FORECAST_MONTH_DAY = (8, 1)

# ---- Splits ----------------------------------------------------------------
TRAIN_END = 2011                # training years: FIRST_MODEL_YEAR .. TRAIN_END
TEST_START = 2012               # untouched test years: TEST_START .. LAST_YIELD_YEAR
# Expanding-window validation blocks inside the training period (tuning + empirical interval residuals)
CV_BLOCKS = [(1997, 1999), (2000, 2002), (2003, 2005), (2006, 2008), (2009, 2011)]

# ---- Sources ---------------------------------------------------------------
NASS_QUERY = {
    "source_desc": "SURVEY",
    "sector_desc": "CROPS",
    "group_desc": "FIELD CROPS",
    "commodity_desc": "CORN",
    "statisticcat_desc": "YIELD",
    "short_desc": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
    "agg_level_desc": "COUNTY",
    "state_name": STATE_NAME,
}
NASS_FILE = RAW_DIR / "nass_iowa_corn_grain_yield_county.csv"
GAZETTEER_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_counties_national.zip"
GAZETTEER_FILE = RAW_DIR / "gazetteer_2024_iowa_counties.csv"
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMETERS = ["T2M_MAX", "T2M_MIN", "PRECTOTCORR"]
POWER_START = f"{FIRST_MODEL_YEAR}0101"
POWER_END = f"{FORECAST_YEAR}0731"
