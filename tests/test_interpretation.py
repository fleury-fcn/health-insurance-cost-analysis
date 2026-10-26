import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from insurance_costs.features import preprocessor
from insurance_costs.interpretation import decode, encode, explain_profile, subgroup_errors
from tests.test_modeling import synthetic


@pytest.fixture(scope="module")
def fitted():
    df = synthetic(300)
    X, y = df.drop(columns="charges"), df["charges"]
    model = Pipeline([("prep", preprocessor()), ("model", Ridge())]).fit(X, y)
    return model, X, y


def test_encode_decode_roundtrip(fitted):
    _, X, _ = fitted
    back = decode(encode(X.head(20)))
    pd.testing.assert_frame_equal(back, X.head(20).reset_index(drop=True), check_dtype=False)


def test_shap_contributions_add_up_to_prediction(fitted):
    model, X, _ = fitted
    out = explain_profile(model, X.head(1), X)
    total = out["base_value"] + sum(out["contributions"].values())
    assert total == pytest.approx(out["prediction"], rel=1e-6)
    assert abs(out["contributions"]["smoker"]) > abs(out["contributions"]["region"])


def test_subgroup_errors_cover_every_group_with_ci(fitted):
    model, X, y = fitted
    result = subgroup_errors(y, model.predict(X), X)
    assert set(result["sex"]["groups"]) == {"female", "male"}
    ci = result["smoker"]["groups"]["yes"]["calibration_ci95"]
    assert ci[0] <= result["smoker"]["groups"]["yes"]["calibration_ratio"] <= ci[1]
    assert np.isfinite(result["overall_calibration_ratio"])
