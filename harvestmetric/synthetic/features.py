"""Feature sets tied to the moment a prediction would be made, and the shared preprocessor.

Prediction point (primary task, "at_sowing"): on the sowing date, before any
in-season measurement exists. Only information known or decided at that
moment is allowed.

Column timing, based on the dataset documentation:

* known at sowing  -> region, crop_type, irrigation_type, fertilizer_type,
                      soil_pH, latitude, longitude, sowing_month
* in-season        -> soil_moisture_%, NDVI_index, crop_disease_status,
                      pesticide_usage_ml, timestamp (a random in-cycle snapshot)
* season aggregate -> temperature_C, rainfall_mm, humidity_%, sunlight_hours
                      (means/totals "during crop cycle": only known at harvest)
* post-harvest     -> harvest_date, total_days, the target itself
* identifiers      -> farm_id, sensor_id (never features)

The "full_season" set is a diagnostic upper bound, not a forecast: it uses
everything except identifiers, raw dates and post-harvest duration, so it
answers "is there any signal at all in these columns?".
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


@dataclass(frozen=True)
class FeatureSet:
    name: str
    description: str
    categorical: tuple[str, ...]
    numeric: tuple[str, ...]

    @property
    def columns(self) -> list[str]:
        return [*self.categorical, *self.numeric]


AT_SOWING = FeatureSet(
    name="at_sowing",
    description="Forecast made on the sowing date: site, crop and management choices only.",
    categorical=("region", "crop_type", "irrigation_type", "fertilizer_type"),
    numeric=("soil_pH", "latitude", "longitude", "sowing_month"),
)

FULL_SEASON = FeatureSet(
    name="full_season",
    description=(
        "Diagnostic only (not a forecast): adds in-season sensor readings and "
        "season aggregates that are only complete at harvest."
    ),
    categorical=(*AT_SOWING.categorical, "crop_disease_status"),
    numeric=(
        *AT_SOWING.numeric, "soil_moisture_%", "temperature_C", "rainfall_mm",
        "humidity_%", "sunlight_hours", "pesticide_usage_ml", "NDVI_index",
    ),
)

FEATURE_SETS = {fs.name: fs for fs in (AT_SOWING, FULL_SEASON)}

# Never allowed as model inputs for any task.
FORBIDDEN = {config.TARGET, "harvest_date", "total_days", "timestamp", *config.ID_COLS}


def build_preprocessor(fs: FeatureSet) -> ColumnTransformer:
    """One-hot for categoricals (fixed category lists), scaling for numerics.

    Fixing the category lists means training and inference always produce the
    same columns regardless of which categories appear in a given batch.
    """
    leaked = FORBIDDEN.intersection(fs.columns)
    if leaked:
        raise ValueError(f"Feature set {fs.name} contains forbidden columns: {sorted(leaked)}")
    encoder = OneHotEncoder(
        categories=[config.CATEGORIES[c] for c in fs.categorical],
        handle_unknown="error",
        sparse_output=False,
    )
    return ColumnTransformer(
        [("cat", encoder, list(fs.categorical)), ("num", StandardScaler(), list(fs.numeric))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
