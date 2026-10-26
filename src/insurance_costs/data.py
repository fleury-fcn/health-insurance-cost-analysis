"""Data acquisition, quality audit and cleaning."""

from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from .config import DATA_URL, PROCESSED_DATA, RAW_DATA

EXPECTED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]


def load_raw(path: Path = RAW_DATA) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(DATA_URL, timeout=30) as response:
            path.write_bytes(response.read())
    return pd.read_csv(path)


def audit(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "dtypes": {key: str(value) for key, value in df.dtypes.items()},
        "negative_charges": int((df["charges"] < 0).sum()),
        "invalid_bmi": int((df["bmi"] <= 0).sum()),
        "invalid_age": int((df["age"] <= 0).sum()),
        "outlier_charges_iqr": int(
            (df["charges"] > df["charges"].quantile(0.75)
             + 1.5 * (df["charges"].quantile(0.75) - df["charges"].quantile(0.25))).sum()
        ),
    }


def clean_data(df: pd.DataFrame, output_path: Path = PROCESSED_DATA) -> pd.DataFrame:
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Colonnes manquantes: {sorted(missing_columns)}")
    clean = df[EXPECTED_COLUMNS].drop_duplicates().copy()
    clean = clean[(clean["age"] > 0) & (clean["bmi"] > 0) & (clean["charges"] >= 0)]
    clean["sex"] = clean["sex"].astype("category")
    clean["smoker"] = clean["smoker"].astype("category")
    clean["region"] = clean["region"].astype("category")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)
    return clean


def load_clean() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA)
