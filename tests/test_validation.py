import pandas as pd
import pytest

from harvestmetric.data import DataValidationError, load_dataset, parse_dates, prepare, validate
from harvestmetric.inference import InputError, validate_input


def test_valid_frame_has_no_issues(frame):
    assert validate(frame) == []


def test_missing_column_reported(frame):
    assert "missing columns" in validate(frame.drop(columns=["soil_pH"]))[0]


def test_unknown_category_reported(frame):
    frame.loc[0, "crop_type"] = "Banana"
    assert any("crop_type" in i and "Banana" in i for i in validate(frame))


def test_out_of_range_value_reported(frame):
    frame.loc[0, "soil_pH"] = 9.9
    assert any(i.startswith("soil_pH") for i in validate(frame))


def test_conflicting_records_for_same_farm_reported(frame):
    frame.loc[1, "farm_id"] = frame.loc[0, "farm_id"]
    assert any("conflicting" in i for i in validate(frame))


def test_strict_load_raises(tmp_path, frame):
    frame.loc[0, "region"] = "Mars"
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(DataValidationError):
        load_dataset(path)


def test_none_category_is_not_treated_as_missing(tmp_path, frame):
    frame.loc[:, "irrigation_type"] = "None"
    path = tmp_path / "none.csv"
    frame.to_csv(path, index=False)
    df, _ = load_dataset(path)
    assert (df.irrigation_type == "None").all()


def test_parse_dates_accepts_both_formats():
    import pandas as pd
    s = pd.Series(["08/01/2024", "2024-01-08"], name="d")
    out = parse_dates(s)
    assert out.iloc[0] == out.iloc[1] == pd.Timestamp("2024-01-08")


def test_parse_dates_rejects_garbage():
    import pandas as pd
    with pytest.raises(DataValidationError):
        parse_dates(pd.Series(["not a date"], name="d"))


def test_inconsistent_total_days_rejected(frame):
    frame.loc[0, "total_days"] = frame.loc[0, "total_days"] + 1
    with pytest.raises(DataValidationError, match="total_days"):
        prepare(frame)


def test_valid_input_passes(valid_record):
    assert validate_input(valid_record)["soil_pH"] == 6.5


@pytest.mark.parametrize("field,value,message", [
    ("soil_pH", 8.2, "soil_pH must be between 5.5 and 7.5"),
    ("latitude", "abc", "latitude must be a number"),
    ("crop_type", "Banana", "crop_type must be one of"),
    ("sowing_month", 13, "sowing_month must be between 1 and 12"),
    ("sowing_month", 2.5, "whole month"),
])
def test_invalid_input_gives_clear_message(valid_record, field, value, message):
    valid_record[field] = value
    with pytest.raises(InputError, match=message):
        validate_input(valid_record)


def test_missing_input_field(valid_record):
    del valid_record["region"]
    with pytest.raises(InputError, match="Missing required field"):
        validate_input(valid_record)


def test_all_errors_reported_together(valid_record):
    valid_record.update(soil_pH=9, crop_type="Banana")
    with pytest.raises(InputError) as exc:
        validate_input(valid_record)
    assert "soil_pH" in str(exc.value) and "crop_type" in str(exc.value)


def test_provenance_compare_matches_duplicated_copy(tmp_path):
    from harvestmetric.provenance import compare
    from tests.conftest import make_frame
    base = make_frame(n_farms=20)
    base.to_csv(tmp_path / "a.csv", index=False)
    dup = make_frame(n_farms=20, copies=10)
    dup["sowing_date"] = pd.to_datetime(dup["sowing_date"], format="%d/%m/%Y").dt.strftime("%Y-%m-%d")
    dup.to_csv(tmp_path / "b.csv", index=False)
    assert compare(tmp_path / "a.csv", tmp_path / "b.csv")["identical_records"]
    dup.loc[0:9, "soil_pH"] = 7.0  # one copy of farms 1-10 altered -> records differ
    dup.to_csv(tmp_path / "c.csv", index=False)
    assert not compare(tmp_path / "a.csv", tmp_path / "c.csv")["identical_records"]
