"""Experiment tracking with MLflow (local file store, one child run per candidate)."""

from __future__ import annotations

import logging
import os

from .config import ARTIFACT_DIR, BEST_MODEL, FIGURE_DIR, MLFLOW_DIR, MODEL_DIR, RANDOM_STATE

EXPERIMENT = "health-insurance-costs"


def _flatten(prefix: str, d: dict) -> dict:
    return {f"{prefix}{k}": v for k, v in d.items() if isinstance(v, int | float)}


def log_experiment(modeling: dict, summary: dict) -> str:
    """Log the benchmark to MLflow and return the parent run id."""
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    import mlflow

    logging.getLogger("mlflow").setLevel(logging.ERROR)
    mlflow.set_tracking_uri(MLFLOW_DIR.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    best = modeling["best_model"]
    with mlflow.start_run(run_name=f"benchmark-{summary['git_commit'][:7]}") as parent:
        mlflow.set_tags({"git_commit": summary["git_commit"], "data_sha256": summary["data_sha256"],
                         "best_model": best})
        mlflow.log_params({"random_state": RANDOM_STATE, "n_rows": summary["rows"],
                           "n_iter": summary["n_iter"], "selection_metric": "cv_rmse",
                           "best_model": best})
        b = modeling["models"][best]
        mlflow.log_metrics({"best_cv_rmse": b["cv_rmse"],
                            **_flatten("best_test_", b["test"])})
        for name, res in modeling["models"].items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tags({"family": res["family"], "is_best": str(name == best)})
                mlflow.log_params({k: str(v)[:250] for k, v in res["best_params"].items()})
                mlflow.log_metrics({"cv_rmse": res["cv_rmse"], "cv_rmse_std": res["cv_rmse_std"],
                                    "cv_mae": res["cv_mae"], "cv_r2": res["cv_r2"],
                                    "fit_seconds": res["fit_seconds"],
                                    **_flatten("test_", res["test"])})
        for path in [ARTIFACT_DIR / "metrics.json", ARTIFACT_DIR / "interpretation.json",
                     MODEL_DIR / "model_card.json", BEST_MODEL]:
            if path.exists():
                mlflow.log_artifact(str(path))
        for png in FIGURE_DIR.glob("*.png"):
            mlflow.log_artifact(str(png), artifact_path="figures")
        return parent.info.run_id
