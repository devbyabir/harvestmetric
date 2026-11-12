import joblib
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from harvestmetric.synthetic import config
from harvestmetric.synthetic.data import prepare
from harvestmetric.synthetic.features import AT_SOWING, build_preprocessor
from harvestmetric.synthetic.inference import InputError, load_artifact, predict
from harvestmetric.synthetic.models import GroupMeanRegressor


@pytest.fixture
def artifact(frame):
    df = prepare(frame)
    model = Pipeline([("prep", build_preprocessor(AT_SOWING)), ("model", Ridge())])
    model.fit(df[AT_SOWING.columns], df[config.TARGET])
    return {"model": model, "feature_set": AT_SOWING.name, "prediction_supported": True,
            "_train": df}


def test_inference_matches_training_pipeline(artifact, valid_record):
    """The inference path must give exactly what the fitted pipeline gives on the same row."""
    import pandas as pd
    direct = artifact["model"].predict(pd.DataFrame([valid_record])[AT_SOWING.columns])[0]
    assert predict(valid_record, artifact)[0] == pytest.approx(direct)


def test_batch_prediction(artifact, valid_record):
    other = dict(valid_record, crop_type="Rice", soil_pH=7.0)
    preds = predict([valid_record, other], artifact)
    assert len(preds) == 2 and all(np.isfinite(preds))


def test_numeric_strings_are_accepted(artifact, valid_record):
    as_text = {k: str(v) for k, v in valid_record.items()}
    assert predict(as_text, artifact) == pytest.approx(predict(valid_record, artifact))


def test_invalid_input_raises_before_model(artifact, valid_record):
    valid_record["soil_pH"] = 3
    with pytest.raises(InputError):
        predict(valid_record, artifact)


def test_refuses_when_evaluation_did_not_support_prediction(artifact, valid_record):
    artifact["prediction_supported"] = False
    with pytest.raises(RuntimeError, match="did not support"):
        predict(valid_record, artifact)
    assert len(predict(valid_record, artifact, allow_unsupported=True)) == 1


def test_artifact_roundtrip(tmp_path, artifact, valid_record):
    path = tmp_path / "m.joblib"
    joblib.dump({k: v for k, v in artifact.items() if k != "_train"}, path)
    loaded = load_artifact(path)
    assert predict(valid_record, loaded) == pytest.approx(predict(valid_record, artifact))


def test_missing_artifact_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="harvestmetric.train"):
        load_artifact(tmp_path / "none.joblib")


def test_group_mean_baseline(frame):
    df = prepare(frame)
    m = GroupMeanRegressor("crop_type").fit(df, df[config.TARGET])
    expected = df.groupby("crop_type")[config.TARGET].mean()
    pred = m.predict(df.head(5))
    assert np.allclose(pred, df.head(5).crop_type.map(expected))


def test_compare_groups_detects_clear_difference():
    from harvestmetric.synthetic.analytics import compare_groups
    rng = np.random.default_rng(0)
    c = compare_groups(rng.normal(5000, 100, 50), rng.normal(3000, 100, 50))
    assert c.conclusive and 1900 < c.diff < 2100 and c.ci_low > 0


def test_compare_groups_identical_distributions_inconclusive():
    from harvestmetric.synthetic.analytics import compare_groups
    rng = np.random.default_rng(1)
    x = rng.uniform(2000, 6000, 200)
    assert not compare_groups(x[:100], x[100:]).conclusive


def test_compare_groups_needs_data():
    from harvestmetric.synthetic.analytics import compare_groups
    with pytest.raises(ValueError, match="at least 2"):
        compare_groups([1.0], [1.0, 2.0])


def test_simulator_reproduces_leakage_with_pure_noise():
    from harvestmetric.synthetic.simulator import simulate
    r = simulate(n_farms=120, copies=10, signal=0.0, n_estimators=30)
    assert r.row_level_r2 > 0.9          # memorisation through duplicates
    assert r.grouped_r2 < 0.1            # honest estimate: no skill
    assert r.rows == 1200


def test_simulator_without_duplicates_has_no_gap():
    from harvestmetric.synthetic.simulator import simulate
    r = simulate(n_farms=200, copies=1, signal=0.6, n_estimators=30)
    assert abs(r.row_level_r2 - r.grouped_r2) < 0.1 and r.grouped_r2 > 0.3


def test_simulator_rejects_bad_signal():
    from harvestmetric.synthetic.simulator import make_dataset
    with pytest.raises(ValueError, match="signal"):
        make_dataset(50, 2, 1.5)
