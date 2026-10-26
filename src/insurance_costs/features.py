"""Feature engineering as scikit-learn components (no leakage, identical at serving time)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import (
    ARTIFACT_DIR,
    CATEGORY_LEVELS,
    CV_FOLDS,
    FEATURES,
    OBESITY_THRESHOLD,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from .viz import dump

INTERACTIONS = ["obese", "smoker_obese", "bmi_smoker", "age_smoker", "age_sq"]


class InsuranceFeatures(BaseEstimator, TransformerMixin):
    """Add domain features to the raw policyholder table.

    - smoker_flag: binary smoking indicator (needed to build interactions)
    - obese: BMI >= 30 (WHO threshold where the EDA shows a cost break for smokers)
    - smoker_obese: smoker x obese, the dominant cost segment
    - bmi_smoker: BMI slope that only applies to smokers
    - age_smoker: age slope specific to smokers. Costs rise by the same dollar amount per
      year in both groups (parallel bands), so on a log scale (log target, log-link GLM)
      smokers need a flatter *relative* age slope.
    - age_sq: curvature of the age effect (centred to limit collinearity)
    """

    def __init__(self, interactions: bool = True):
        self.interactions = interactions

    def fit(self, X: pd.DataFrame, y=None):
        missing = set(FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"Variables manquantes: {sorted(missing)}")
        self.age_mean_ = float(X["age"].mean())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[FEATURES].copy()
        smoker = out["smoker"].eq("yes").astype(float)
        if self.interactions:
            obese = (out["bmi"] >= OBESITY_THRESHOLD).astype(float)
            out["obese"] = obese
            out["smoker_obese"] = smoker * obese
            out["bmi_smoker"] = out["bmi"] * smoker
            out["age_smoker"] = out["age"] * smoker
            out["age_sq"] = (out["age"] - self.age_mean_) ** 2
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURES + (INTERACTIONS if self.interactions else []), dtype=object)


def numeric_columns(interactions: bool) -> list[str]:
    return ["age", "bmi", "children"] + (INTERACTIONS if interactions else [])


def preprocessor(interactions: bool = True, kind: str = "linear") -> Pipeline:
    """Feature pipeline.

    linear: one-hot encoding + standardisation (penalised/GLM models are scale-sensitive).
    tree: ordinal encoding, no scaling (trees are invariant to monotone transforms).
    """
    categories = [CATEGORY_LEVELS[c] for c in ["sex", "smoker", "region"]]
    if kind == "linear":
        encoder = ColumnTransformer([
            ("num", StandardScaler(), numeric_columns(interactions)),
            ("cat", OneHotEncoder(categories=categories, drop="if_binary",
                                  handle_unknown="ignore", sparse_output=False),
             ["sex", "smoker", "region"]),
        ], verbose_feature_names_out=False)
    else:
        encoder = ColumnTransformer([
            ("num", "passthrough", numeric_columns(interactions)),
            ("cat", OrdinalEncoder(categories=categories, handle_unknown="use_encoded_value",
                                   unknown_value=-1), ["sex", "smoker", "region"]),
        ], verbose_feature_names_out=False)
    return Pipeline([("features", InsuranceFeatures(interactions)), ("encode", encoder)])


class SmearedLogRegressor(RegressorMixin, BaseEstimator):
    """Fit a regressor on log(y) and back-transform with Duan's smearing estimator.

    exp(E[log y | x]) estimates the conditional *median*, which underestimates the mean
    of a right-skewed cost. Duan's factor mean(exp(residuals)) restores an unbiased mean,
    which a premium needs (the portfolio must balance).
    """

    def __init__(self, regressor=None):
        self.regressor = regressor

    def fit(self, X, y):
        self.regressor_ = clone(self.regressor if self.regressor is not None else Ridge())
        log_y = np.log(np.asarray(y, dtype=float))
        self.regressor_.fit(X, log_y)
        residuals = log_y - self.regressor_.predict(X)
        self.smearing_ = float(np.mean(np.exp(residuals)))
        return self

    def predict(self, X):
        return np.exp(self.regressor_.predict(X)) * self.smearing_


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Fixed, stratified-by-smoking hold-out split shared by every phase."""
    return train_test_split(df[FEATURES], df[TARGET], test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=df["smoker"])


def feature_ablation(df: pd.DataFrame) -> dict:
    """Cross-validated value of each engineered feature (training data only)."""
    X_train, _, y_train, _ = split(df)
    cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    def score(name: str, model) -> None:
        s = cross_validate(model, X_train, y_train, cv=cv,
                           scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"})
        results[name] = {"cv_rmse": float(-s["test_rmse"].mean()),
                         "cv_r2": float(s["test_r2"].mean())}

    score("Ridge - variables brutes", Pipeline([("prep", preprocessor(False)),
                                                ("model", Ridge(1.0))]))
    score("Ridge - avec interactions", Pipeline([("prep", preprocessor(True)),
                                                 ("model", Ridge(1.0))]))
    score("Ridge log (smearing) - variables brutes",
          Pipeline([("prep", preprocessor(False)),
                    ("model", SmearedLogRegressor(Ridge(1.0)))]))
    score("Ridge log (smearing) - avec interactions",
          Pipeline([("prep", preprocessor(True)),
                    ("model", SmearedLogRegressor(Ridge(1.0)))]))
    prep = preprocessor(True)
    prep.steps[1][1].transformers[0] = (
        "num", StandardScaler(), [c for c in numeric_columns(True) if c != "age_smoker"])
    score("Ridge log (smearing) - interactions sans age_smoker",
          Pipeline([("prep", prep), ("model", SmearedLogRegressor(Ridge(1.0)))]))
    for drop in INTERACTIONS:
        prep = preprocessor(True)
        cols = [c for c in numeric_columns(True) if c != drop]
        prep.steps[1][1].transformers[0] = ("num", StandardScaler(), cols)
        score(f"Ridge - interactions sans {drop}", Pipeline([("prep", prep),
                                                             ("model", Ridge(1.0))]))
    base = results["Ridge - avec interactions"]["cv_rmse"]
    for v in results.values():
        v["delta_rmse_vs_full"] = v["cv_rmse"] - base
    dump(results, ARTIFACT_DIR / "feature_ablation.json")
    return results
