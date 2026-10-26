"""Feature engineering and model benchmarking."""

import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import RANDOM_STATE
from .config import ARTIFACT_DIR, MODEL_DIR

TARGET = "charges"
NUMERIC = ["age", "bmi", "children"]
CATEGORICAL = ["sex", "smoker", "region"]


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=[TARGET]).copy()
    out["bmi_smoker"] = out["bmi"] * (out["smoker"] == "yes").astype(int)
    out["age_smoker"] = out["age"] * (out["smoker"] == "yes").astype(int)
    return out


def preprocessor() -> ColumnTransformer:
    numeric = NUMERIC + ["bmi_smoker", "age_smoker"]
    return ColumnTransformer(
        [("num", StandardScaler(), numeric),
         ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)]
    )


def model_pipelines() -> dict[str, Pipeline]:
    return {
        "Ridge": Pipeline([("prep", preprocessor()), ("model", Ridge(alpha=10))]),
        "Lasso": Pipeline([("prep", preprocessor()), ("model", Lasso(alpha=100, max_iter=10000))]),
        "GLM Gamma log": Pipeline([
            ("prep", preprocessor()),
            ("model", TweedieRegressor(power=2, link="log", alpha=0.1, max_iter=2000)),
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor()),
            ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                            random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", preprocessor()),
            ("model", GradientBoostingRegressor(n_estimators=250, max_depth=2,
                                                learning_rate=0.04, loss="huber",
                                                random_state=RANDOM_STATE)),
        ]),
    }


def train_and_evaluate(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, Pipeline, pd.DataFrame]:
    X, y = feature_frame(df), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    metrics, predictions = {}, []
    fitted = {}
    for name, pipe in model_pipelines().items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv,
                                scoring={"rmse": "neg_root_mean_squared_error",
                                         "mae": "neg_mean_absolute_error", "r2": "r2"})
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics[name] = {
            "cv_rmse": float(-scores["test_rmse"].mean()),
            "cv_mae": float(-scores["test_mae"].mean()),
            "cv_r2": float(scores["test_r2"].mean()),
            "test_rmse": float(mean_squared_error(y_test, pred) ** 0.5),
            "test_mae": float(mean_absolute_error(y_test, pred)),
            "test_r2": float(r2_score(y_test, pred)),
        }
        fitted[name] = pipe
        predictions.append(pd.DataFrame({"actual": y_test, "predicted": pred, "model": name}))
    best_name = min(metrics, key=lambda x: metrics[x]["test_rmse"])
    MODEL_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[best_name], MODEL_DIR / "best_model.joblib")
    result = {"best_model": best_name, "models": metrics}
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(result, indent=2))
    return result, pd.concat(predictions), fitted[best_name], pd.DataFrame({
        "feature": X_test.index, "actual": y_test, "predicted": fitted[best_name].predict(X_test)
    })
