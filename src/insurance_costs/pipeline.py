"""Single-command end-to-end pipeline."""

import json
from pathlib import Path

from .analysis import run_analysis
from .data import audit, clean_data, load_raw
from .interpretation import interpret
from .modeling import train_and_evaluate
from .report import generate_report


def main() -> None:
    raw = load_raw()
    quality = audit(raw)
    clean = clean_data(raw)
    statistics = run_analysis(clean)
    metrics, _, best_model, _ = train_and_evaluate(clean)
    interpret(best_model, clean)
    best_name = metrics["best_model"]
    import mlflow
    mlflow.set_tracking_uri(str(Path("artifacts/mlruns").resolve()))
    mlflow.set_experiment("health-insurance-costs")
    with mlflow.start_run():
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_rows", len(clean))
        mlflow.log_param("best_model", best_name)
        mlflow.log_metrics({
            "test_rmse": metrics["models"][best_name]["test_rmse"],
            "test_mae": metrics["models"][best_name]["test_mae"],
            "test_r2": metrics["models"][best_name]["test_r2"],
        })
    report = generate_report(quality, statistics, metrics)
    print(json.dumps({
        "rows_after_cleaning": len(clean),
        "best_model": best_name,
        "test_rmse": metrics["models"][best_name]["test_rmse"],
        "report": str(report),
    }, indent=2))


if __name__ == "__main__":
    main()
