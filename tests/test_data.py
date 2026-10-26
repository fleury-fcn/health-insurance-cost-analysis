import numpy as np
import pandas as pd
import pytest

from insurance_costs.data import audit, clean_data


def _raw() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [20, 20, -1, 45, 33],
        "sex": ["male", "male", "male", " Female", "female"],
        "bmi": [22.0, 22.0, 0.0, 31.2, np.nan],
        "children": [0, 0, 0, 2, 1],
        "smoker": ["no", "no", "no", "YES", "no"],
        "region": ["northeast"] * 5,
        "charges": [1000.0, 1000.0, -5.0, 40000.0, 3000.0],
    })


def test_audit_detects_quality_issues():
    report = audit(_raw())
    assert report["duplicates"] == 1
    assert report["missing_by_column"]["bmi"] == 1
    assert report["out_of_range"]["age"] == 1
    assert report["out_of_range"]["charges"] == 1


def test_cleaning_removes_duplicates_missing_and_invalid_rows(tmp_path):
    cleaned, log = clean_data(_raw(), tmp_path / "clean.csv")
    assert len(cleaned) == 2
    assert set(cleaned["sex"]) == {"male", "female"}
    assert set(cleaned["smoker"]) == {"no", "yes"}
    assert cleaned.isna().sum().sum() == 0
    assert (tmp_path / "clean.csv").exists()
    assert sum(entry["rows_removed"] for entry in log) == 3


def test_cleaning_keeps_extreme_but_valid_costs(tmp_path):
    raw = _raw().iloc[[0, 3]].assign(charges=[1000.0, 63770.0])
    cleaned, _ = clean_data(raw, None)
    assert cleaned["charges"].max() == 63770.0


def test_cleaning_rejects_missing_columns():
    with pytest.raises(ValueError):
        clean_data(_raw().drop(columns=["region"]), None)
