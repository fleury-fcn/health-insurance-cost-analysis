import pandas as pd

from insurance_costs.data import audit, clean_data


def test_cleaning_removes_duplicates_and_invalid_rows():
    df = pd.DataFrame({
        "age": [20, 20, -1], "sex": ["male"] * 3, "bmi": [22, 22, 0],
        "children": [0] * 3, "smoker": ["no"] * 3, "region": ["north"] * 3,
        "charges": [1000, 1000, -5],
    })
    cleaned = clean_data(df)
    assert len(cleaned) == 1
    assert audit(cleaned)["missing_values"] == 0
