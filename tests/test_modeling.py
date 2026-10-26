import numpy as np
import pandas as pd
import pytest

from insurance_costs.modeling import metrics, normalized_gini, paired_bootstrap, train_and_evaluate


def synthetic(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 65, n), "sex": rng.choice(["female", "male"], n),
        "bmi": rng.normal(30, 6, n).clip(16, 50), "children": rng.integers(0, 4, n),
        "smoker": rng.choice(["no", "yes"], n, p=[0.8, 0.2]),
        "region": rng.choice(["northeast", "northwest", "southeast", "southwest"], n),
    })
    smoker = df.smoker.eq("yes")
    df["charges"] = (2000 + 260 * df.age + smoker * 15000
                     + smoker * (df.bmi >= 30) * 20000 + rng.gamma(2, 1000, n))
    return df


def test_gini_is_one_for_perfect_ranking_and_zero_for_constant():
    y = np.array([1.0, 5, 3, 10])
    assert normalized_gini(y, y) == pytest.approx(1.0)
    assert normalized_gini(y, np.ones(4)) == 0.0


def test_metrics_report_calibration_ratio():
    y = np.array([100.0, 200, 300])
    m = metrics(y, y * 1.1)
    assert m["calibration_ratio"] == pytest.approx(1.1)
    assert m["r2"] < 1


def test_paired_bootstrap_detects_clearly_better_model():
    rng = np.random.default_rng(0)
    y = rng.normal(100, 10, 500)
    result = paired_bootstrap(y, y + rng.normal(0, 1, 500), y + rng.normal(0, 10, 500))
    assert result["ci95"][1] < 0
    assert result["prob_a_better"] > 0.99


def test_training_selects_on_cv_and_beats_baseline():
    result = train_and_evaluate(synthetic(), n_iter=2,
                                only=["Baseline (moyenne)", "Ridge", "LightGBM"])
    assert result["best_model"] in {"Ridge", "LightGBM"}
    best = result["models"][result["best_model"]]
    assert best["test"]["r2"] > 0.8
    assert best["cv_rmse"] < result["models"]["Baseline (moyenne)"]["cv_rmse"]
    assert len(best["cv_fold_rmse"]) == 5
