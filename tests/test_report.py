import pytest

from insurance_costs.config import ARTIFACT_DIR
from insurance_costs.report import build_report, fr_money, fr_p, md

pytestmark_artifacts = pytest.mark.skipif(
    not (ARTIFACT_DIR / "run_summary.json").exists(), reason="pipeline not run yet")


def test_french_formatting():
    assert fr_money(13279.12) == "13 279 $"
    assert fr_p(0.0001) == "< 0,001"
    assert fr_p(0.0345) == "0,035"


def test_md_escapes_html_and_localises_decimals():
    out = str(md("**gras** <script> r = 0.79"))
    assert "<strong>gras</strong>" in out
    assert "<script>" not in out
    assert "0,79" in out


@pytestmark_artifacts
def test_report_renders_every_section(tmp_path):
    html = build_report(tmp_path / "report.html").read_text(encoding="utf-8")
    for anchor in ["resume", "donnees", "eda", "inference", "features", "modeles",
                   "interpretation", "limites", "recommandations", "annexe"]:
        assert f'id="{anchor}"' in html
    assert "Plotly.newPlot" in html
