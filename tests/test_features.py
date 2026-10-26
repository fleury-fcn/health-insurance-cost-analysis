import numpy as np
import pandas as pd

from insurance_costs.features import InsuranceFeatures, SmearedLogRegressor, preprocessor


def _profiles() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [30, 50], "sex": ["female", "male"], "bmi": [32.0, 24.0], "children": [1, 0],
        "smoker": ["yes", "no"], "region": ["southwest", "northeast"],
    })


def test_interactions_are_built_from_domain_rules():
    out = InsuranceFeatures().fit_transform(_profiles())
    assert out.loc[0, ["obese", "smoker_obese"]].tolist() == [1.0, 1.0]
    assert out.loc[0, "bmi_smoker"] == 32.0
    assert out.loc[0, "age_smoker"] == 30
    assert out.loc[1, ["smoker_obese", "bmi_smoker", "age_smoker"]].sum() == 0
    assert out.loc[0, "age_sq"] == out.loc[1, "age_sq"] == 100  # centred on mean age 40


def test_preprocessor_output_is_numeric_and_named():
    prep = preprocessor().fit(_profiles())
    Xt = prep.transform(_profiles())
    assert Xt.shape == (2, len(prep.get_feature_names_out()))
    assert np.isfinite(Xt).all()


def test_tree_preprocessor_handles_unseen_region():
    prep = preprocessor(kind="tree").fit(_profiles())
    unseen = _profiles().assign(region=["mars", "mars"])
    assert (prep.transform(unseen)[:, -1] == -1).all()


def test_smearing_restores_mean_of_lognormal_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 1))
    y = np.exp(1 + 0.5 * X[:, 0] + rng.normal(0, 0.8, 5000))
    model = SmearedLogRegressor().fit(X, y)
    assert model.smearing_ > 1.2  # exp(0.8^2 / 2) ~ 1.38
    assert abs(model.predict(X).mean() / y.mean() - 1) < 0.05
