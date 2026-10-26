import pandas as pd

from insurance_costs.modeling import feature_frame


def test_feature_engineering_has_interactions():
    df = pd.DataFrame({
        "age": [30], "sex": ["female"], "bmi": [25.0], "children": [1],
        "smoker": ["yes"], "region": ["southwest"], "charges": [5000],
    })
    features = feature_frame(df)
    assert features.loc[0, "bmi_smoker"] == 25
    assert "charges" not in features
