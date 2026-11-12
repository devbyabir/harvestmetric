import numpy as np
import pytest

from harvestmetric.synthetic import config
from harvestmetric.synthetic.data import load_dataset, prepare, read_raw
from harvestmetric.synthetic.features import AT_SOWING, FEATURE_SETS, FORBIDDEN, FeatureSet, build_preprocessor
from harvestmetric.synthetic.split import LeakageError, assert_no_leakage, group_holdout_split
from harvestmetric.synthetic.train import cv_splits


def test_duplicates_are_removed_on_load(csv_with_duplicates):
    df, report = load_dataset(csv_with_duplicates)
    assert report.raw_rows == 500 and report.unique_rows == 50
    assert df[config.GROUP_COL].is_unique


def test_holdout_split_has_no_shared_farms_even_with_duplicate_rows(csv_with_duplicates):
    raw = prepare(read_raw(csv_with_duplicates))  # still 10 copies per farm
    tr, te = group_holdout_split(raw)
    assert set(raw.iloc[tr].farm_id).isdisjoint(raw.iloc[te].farm_id)
    assert 0.1 < len(te) / len(raw) < 0.3


def test_holdout_split_is_deterministic(frame):
    df = prepare(frame)
    a, b = group_holdout_split(df), group_holdout_split(df)
    assert np.array_equal(a[1], b[1])


def test_leakage_check_detects_shared_farm(frame):
    df = prepare(frame)
    with pytest.raises(LeakageError):
        assert_no_leakage(df.iloc[:40], df.iloc[30:])


def test_leakage_check_detects_identical_record_under_new_id(frame):
    df = prepare(frame)
    test = df.iloc[[0]].copy()
    test["farm_id"], test["sensor_id"] = "FARM9999", "SENS9999"
    with pytest.raises(LeakageError, match="identical records"):
        assert_no_leakage(df.iloc[:40], test)


def test_cv_folds_are_grouped(frame):
    df = prepare(frame)
    for tr, va in cv_splits(df, seed=1):
        assert set(df.iloc[tr].farm_id).isdisjoint(df.iloc[va].farm_id)


@pytest.mark.parametrize("name", list(FEATURE_SETS))
def test_feature_sets_exclude_post_harvest_and_ids(name):
    assert not FORBIDDEN.intersection(FEATURE_SETS[name].columns)


def test_at_sowing_excludes_in_season_measurements():
    in_season = {"NDVI_index", "crop_disease_status", "soil_moisture_%", "rainfall_mm",
                 "temperature_C", "humidity_%", "sunlight_hours", "pesticide_usage_ml"}
    assert not in_season.intersection(AT_SOWING.columns)


def test_preprocessor_rejects_forbidden_feature():
    bad = FeatureSet("bad", "", ("crop_type",), ("total_days",))
    with pytest.raises(ValueError, match="forbidden"):
        build_preprocessor(bad)


def test_preprocessor_output_width_is_fixed(frame):
    df = prepare(frame)
    prep = build_preprocessor(AT_SOWING).fit(df[AT_SOWING.columns])
    one_crop = df[df.crop_type == "Wheat"][AT_SOWING.columns]
    assert prep.transform(one_crop).shape[1] == prep.transform(df[AT_SOWING.columns]).shape[1]
