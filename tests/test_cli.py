from typer.testing import CliRunner

from insurance_costs.cli import app

runner = CliRunner()


def test_cli_lists_every_pipeline_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ["run", "quality", "analyze", "train", "report", "summary"]:
        assert command in result.output


def test_run_command_exposes_reproducibility_options():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--n-iter" in result.output and "--no-mlflow" in result.output
