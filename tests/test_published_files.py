"""Guard against leaking local machine paths into versioned, public outputs."""

from pathlib import Path

import pytest

from insurance_costs.config import ROOT

PUBLISHED = [*ROOT.glob("artifacts/*.json"), *ROOT.glob("models/*.json"),
             *ROOT.glob("notebooks/*.ipynb"), ROOT / "reports/generated/report.html"]


@pytest.mark.parametrize("path", [p for p in PUBLISHED if p.exists()],
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_no_absolute_local_path(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    assert str(ROOT) not in text
    assert str(Path.home()) not in text
