"""HTML report built from the persisted artifacts (minimal version)."""

from __future__ import annotations

import json
from pathlib import Path

from .config import ARTIFACT_DIR, REPORT_DIR


def build_report(output: Path | None = None) -> Path:
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
    rows = "".join(
        f"<tr><td>{n}</td><td>{m['cv_rmse']:,.0f}</td><td>{m['test']['rmse']:,.0f}</td>"
        f"<td>{m['test']['r2']:.3f}</td></tr>" for n, m in metrics["models"].items())
    html = (f"<!doctype html><meta charset='utf-8'><title>Coûts d'assurance maladie</title>"
            f"<h1>Coûts d'assurance maladie</h1><p>Meilleur modèle : {metrics['best_model']}</p>"
            f"<table><tr><th>Modèle</th><th>RMSE CV</th><th>RMSE test</th><th>R²</th></tr>"
            f"{rows}</table>")
    target = output or REPORT_DIR / "report.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return target
