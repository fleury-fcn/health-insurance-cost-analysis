import re

from typer.testing import CliRunner

from insurance_costs.cli import app

runner = CliRunner(env={"NO_COLOR": "1", "COLUMNS": "120", "TERM": "dumb"})
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """Rich forces colours on CI runners; strip escape codes before matching."""
    return ANSI.sub("", text)


def test_cli_lists_every_pipeline_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ["run", "quality", "analyze", "train", "report", "summary"]:
        assert command in plain(result.output)


def test_run_command_exposes_reproducibility_options():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    out = plain(result.output)
    assert "--n-iter" in out and "--no-mlflow" in out
