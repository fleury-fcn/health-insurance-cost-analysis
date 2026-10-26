import numpy as np
import pandas as pd
import pytest

from insurance_costs.inference import (
    bootstrap_ci,
    check_assumptions,
    hedges_g,
    holm,
    k_group_comparison,
    rank_biserial,
    two_group_comparison,
)


def test_bootstrap_ci_contains_true_difference():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 1, 400), rng.normal(8, 1, 400)
    lo, hi = bootstrap_ci(a, b, n_boot=2000)
    assert lo < 2 < hi


def test_holm_is_monotone_and_bounded():
    adjusted = holm([0.01, 0.04, 0.03, 0.5])
    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert all(0 <= p <= 1 for p in adjusted)


def test_effect_sizes_have_expected_sign_and_range():
    a, b = np.array([5.0, 6, 7, 8]), np.array([1.0, 2, 3, 4])
    assert hedges_g(a, b) > 0
    assert rank_biserial(16, 4, 4) == 1.0


def test_assumption_check_flags_skewed_data():
    rng = np.random.default_rng(1)
    result = check_assumptions([rng.lognormal(0, 1, 300), rng.lognormal(0, 2, 300)])
    assert not result["normality_ok"]
    assert not result["equal_variance_ok"]


def _toy() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = 300
    smoker = rng.choice(["yes", "no"], n)
    return pd.DataFrame({
        "smoker": smoker, "region": rng.choice(["a", "b", "c"], n),
        "charges": np.where(smoker == "yes", 30000, 8000) * rng.lognormal(0, 0.3, n),
    })


def test_two_group_detects_large_effect():
    r = two_group_comparison(_toy(), "smoker", "yes", "no", "?")
    assert r["mann_whitney"]["p_value"] < 1e-10
    assert r["hedges_g_magnitude"] == "fort"


def test_k_group_returns_posthoc_for_each_pair():
    r = k_group_comparison(_toy(), "region", "?")
    assert len(r["posthoc_mann_whitney_holm"]) == 3
    assert 0 <= r["anova"]["eta2"] <= 1
