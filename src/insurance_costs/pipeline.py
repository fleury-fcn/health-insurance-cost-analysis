"""End-to-end pipeline: data, quality, EDA, inference, models, interpretation, report."""

from __future__ import annotations

import platform
import subprocess
import time
import warnings
from collections.abc import Callable
from datetime import UTC, datetime

from . import __version__
from .config import ARTIFACT_DIR, RAW_DATA, ROOT
from .data import load_clean, load_raw, run_quality, sha256
from .eda import run_eda
from .features import feature_ablation
from .inference import run_inference
from .interpretation import run_interpretation
from .modeling import run_modeling
from .report import build_report
from .viz import dump


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(n_iter: int = 30, mlflow: bool = True,
        log: Callable[[str], None] = print) -> dict:
    """Run every phase in order and return a summary of the execution."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    timings: dict[str, float] = {}

    def step(name: str, func, *args, **kwargs):
        start = time.perf_counter()
        log(f"▶ {name}…")
        out = func(*args, **kwargs)
        timings[name] = round(time.perf_counter() - start, 2)
        log(f"  ✓ {name} ({timings[name]} s)")
        return out

    raw = step("Chargement et vérification des données", load_raw)
    clean, quality = step("Audit qualité et nettoyage", run_quality, raw)
    step("Analyse exploratoire", run_eda, clean)
    step("Statistiques inférentielles", run_inference, clean)
    step("Ablation du feature engineering", feature_ablation, clean)
    modeling = step("Entraînement et évaluation des modèles", run_modeling, clean, n_iter)
    step("Interprétation et équité", run_interpretation, modeling)

    best = modeling["best_model"]
    summary = {
        "package_version": __version__,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(), "data_sha256": sha256(RAW_DATA),
        "python": platform.python_version(), "platform": platform.platform(),
        "rows": quality["rows_after_cleaning"], "n_iter": n_iter, "best_model": best,
        "best_test_metrics": modeling["models"][best]["test"], "timings_s": timings,
    }
    if mlflow:
        from .tracking import log_experiment
        summary["mlflow_run_id"] = step("Suivi MLflow", log_experiment, modeling, summary)
    dump(summary, ARTIFACT_DIR / "run_summary.json")
    summary["report"] = str(step("Génération du rapport", build_report))
    dump(summary, ARTIFACT_DIR / "run_summary.json")
    return summary


def retrain(n_iter: int = 30) -> dict:
    """Re-run modelling and interpretation only, from the cleaned dataset."""
    modeling = run_modeling(load_clean(), n_iter)
    run_interpretation(modeling)
    return modeling


if __name__ == "__main__":
    run()
