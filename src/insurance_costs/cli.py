"""Command line interface: `insurance-costs --help`."""

from __future__ import annotations

import json

import typer

from . import pipeline
from .config import ARTIFACT_DIR

app = typer.Typer(help="Analyse et modélisation des coûts d'assurance maladie.",
                  no_args_is_help=True, add_completion=False)


@app.command()
def run(n_iter: int = typer.Option(30, help="Tirages de recherche aléatoire par modèle à "
                                             "arbres."),
        mlflow: bool = typer.Option(True, help="Tracer l'expérience dans MLflow.")) -> None:
    """Exécute tout le pipeline et génère le rapport."""
    summary = pipeline.run(n_iter=n_iter, mlflow=mlflow, log=typer.echo)
    m = summary["best_test_metrics"]
    typer.secho(f"\nMeilleur modèle : {summary['best_model']} | RMSE test {m['rmse']:,.0f} $ | "
                f"MAE {m['mae']:,.0f} $ | R² {m['r2']:.3f}", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"Rapport : {summary['report']}")


@app.command()
def quality() -> None:
    """Télécharge, audite et nettoie les données."""
    from .data import load_raw, run_quality
    _, report = run_quality(load_raw())
    typer.echo(json.dumps({k: report[k] for k in ["rows", "duplicates", "missing_values",
                                                   "rows_after_cleaning"]}, indent=2))


@app.command()
def analyze() -> None:
    """Analyse exploratoire et tests statistiques sur les données nettoyées."""
    from .data import load_clean
    from .eda import run_eda
    from .inference import run_inference
    df = load_clean()
    run_eda(df)
    for line in run_inference(df)["conclusions"]:
        typer.echo(f"• {line}")


@app.command()
def train(n_iter: int = typer.Option(30, help="Tirages de recherche aléatoire.")) -> None:
    """Réentraîne les modèles et recalcule l'interprétation."""
    modeling = pipeline.retrain(n_iter)
    for name in modeling["ranking"]:
        r = modeling["models"][name]
        typer.echo(f"{name:22s} CV RMSE {r['cv_rmse']:8,.0f}  test RMSE {r['test']['rmse']:8,.0f}")


@app.command()
def report() -> None:
    """Régénère le rapport HTML à partir des artefacts existants."""
    from .report import build_report
    typer.echo(f"Rapport : {build_report()}")


@app.command()
def summary() -> None:
    """Affiche le résumé de la dernière exécution."""
    path = ARTIFACT_DIR / "run_summary.json"
    if not path.exists():
        raise typer.Exit("Aucune exécution trouvée : lancez `insurance-costs run`.")
    typer.echo(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()
