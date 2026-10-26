"""Self-contained HTML (and optional PDF) report rendered from persisted artifacts.

The report only reads `artifacts/*.json` and figures, so it can be regenerated without
retraining and can never contradict the numbers of the last pipeline run.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import plotly.io as pio
from jinja2 import Environment, PackageLoader, select_autoescape
from markupsafe import Markup, escape

from .config import ARTIFACT_DIR, DATA_URL, FIGURE_DIR, MODEL_DIR, REPORT_DIR
from .viz import load_plotly

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"  # plotly.js bundled with plotly 5.24.1
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
]


def _load(name: str) -> dict:
    return json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))


def fr_num(value: float, digits: int = 0) -> str:
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def fr_money(value: float) -> str:
    return f"{fr_num(value)} $"


def fr_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f} %".replace(".", ",")


def fr_p(value: float) -> str:
    return "< 0,001" if value < 1e-3 else fr_num(value, 3)


def md(text: str) -> Markup:
    """Escape text, then render **bold** markers produced by the analysis modules."""
    text = re.sub(r"(?<=\d)\.(?=\d)", ",", str(text))  # French decimal separator
    return Markup(re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", str(escape(text))))


def figure_html(name: str) -> str:
    if not (FIGURE_DIR / f"{name}.json").exists():
        return ""
    return Markup(pio.to_html(load_plotly(name), full_html=False, include_plotlyjs=False,
                              config={"displaylogo": False, "responsive": True},
                              default_width="100%"))


def context() -> dict:
    quality, eda, inf = _load("data_quality.json"), _load("eda.json"), _load("inference.json")
    metrics, interp = _load("metrics.json"), _load("interpretation.json")
    ablation, summary = _load("feature_ablation.json"), _load("run_summary.json")
    card = json.loads((MODEL_DIR / "model_card.json").read_text(encoding="utf-8"))
    best = metrics["best_model"]
    bm = metrics["models"][best]
    groups = eda["findings"]["risk_groups"]
    ranking = metrics["ranking"] + [m for m in metrics["models"] if m not in metrics["ranking"]]
    relativities = None
    if metrics.get("glm_coefficients"):
        rel = metrics["glm_coefficients"]["relativities"]
        relativities = sorted(rel.items(), key=lambda kv: abs(kv[1] - 1), reverse=True)
    smoker_obese = groups["Fumeur, IMC ≥ 30"]["mean"]
    non_smoker = groups["Non-fumeur"]["mean"]
    return {
        "summary": summary, "quality": quality, "eda": eda, "inf": inf, "metrics": metrics,
        "interp": interp, "ablation": ablation, "card": card, "best": best, "bm": bm,
        "ranking": ranking, "relativities": relativities, "groups": groups,
        "data_url": DATA_URL, "plotly_cdn": PLOTLY_CDN,
        "fig": {name: figure_html(name) for name in [
            "eda_target_distribution", "eda_numeric_features", "eda_costs_by_category",
            "eda_costs_vs_numeric", "eda_correlations", "eda_risk_segments",
            "eda_heatmap_bmi_age", "inference_group_ci", "model_comparison", "model_cv_folds",
            "residuals", "calibration_lorenz", "importance", "shap_beeswarm",
            "partial_dependence", "fairness"]},
        "kpi": {
            "r2": bm["test"]["r2"], "mae": bm["test"]["mae"], "rmse": bm["test"]["rmse"],
            "baseline_rmse": metrics["models"]["Baseline (moyenne)"]["test"]["rmse"],
            "smoker_ratio": eda["findings"]["smoker_ratio"],
            "smoker_obese": smoker_obese, "non_smoker": non_smoker,
            "gap_smoker_obese": smoker_obese - non_smoker,
            "gini": bm["test"]["gini"],
        },
    }


def build_report(output: Path | None = None) -> Path:
    env = Environment(loader=PackageLoader("insurance_costs", "templates"),
                      autoescape=select_autoescape(["html", "j2"]), trim_blocks=True,
                      lstrip_blocks=True)
    env.filters.update(num=fr_num, money=fr_money, pct=fr_pct, p=fr_p, md=md)
    html = env.get_template("report.html.j2").render(**context())
    target = output or REPORT_DIR / "report.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return target


def export_pdf(html: Path, pdf: Path | None = None) -> Path | None:
    """Print the report to PDF with headless Chrome/Chromium when available."""
    chrome = next((c for c in CHROME_CANDIDATES if Path(c).exists() or shutil.which(c)), None)
    if chrome is None:
        return None
    pdf = pdf or html.with_suffix(".pdf")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=20000", "--run-all-compositor-stages-before-draw",
                    f"--print-to-pdf={pdf}", html.resolve().as_uri()],
                   check=True, capture_output=True, timeout=180)
    return pdf
