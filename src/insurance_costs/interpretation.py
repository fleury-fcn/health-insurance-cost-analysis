"""Model interpretation and subgroup error checks."""

import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay

from .config import ARTIFACT_DIR
from .modeling import feature_frame


def interpret(model, df: pd.DataFrame) -> dict:
    X = feature_frame(df)
    predictions = model.predict(X)
    errors = pd.DataFrame({
        "actual": df["charges"], "predicted": predictions,
        "abs_error": (df["charges"] - predictions).abs(),
        "age_band": pd.cut(df["age"], [0, 30, 45, 60, 100],
                           labels=["18-30", "31-45", "46-60", "61+"]),
        "sex": df["sex"], "region": df["region"],
    })
    fairness = {
        column: errors.groupby(column, observed=True)["abs_error"].mean().round(2).to_dict()
        for column in ["age_band", "sex", "region"]
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    (ARTIFACT_DIR / "subgroup_errors.json").write_text(json.dumps(fairness, indent=2, default=str))
    try:
        import shap
        sample = X.sample(min(250, len(X)), random_state=42)
        explainer = shap.Explainer(model.predict, sample)
        values = explainer(sample)
        shap.summary_plot(values, sample, show=False)
        plt.tight_layout()
        plt.savefig(ARTIFACT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
    except (ImportError, ValueError, RuntimeError) as exc:
        (ARTIFACT_DIR / "shap_status.txt").write_text(f"SHAP indisponible: {exc}")
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        PartialDependenceDisplay.from_estimator(model, X, ["age", "bmi"], ax=ax)
        fig.tight_layout()
        fig.savefig(ARTIFACT_DIR / "partial_dependence.png", dpi=150)
        plt.close(fig)
    except (ValueError, RuntimeError) as exc:
        (ARTIFACT_DIR / "partial_dependence_status.txt").write_text(str(exc))
    return fairness
