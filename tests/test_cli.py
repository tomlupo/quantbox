"""Tests for quantbox CLI (quantbox.cli:app)."""

from __future__ import annotations

import json

import yaml
from typer.testing import CliRunner

from quantbox.cli import app

runner = CliRunner()


def _minimal_valid_config() -> dict:
    """Return a minimal config that passes validate_config without errors."""
    return {
        "run": {
            "mode": "backtest",
            "asof": "2026-01-31",
        },
        "artifacts": {
            "root": "./artifacts",
        },
        "plugins": {
            "pipeline": {
                "name": "fund_selection.simple.v1",
                "params": {"top_n": 5},
            },
            "data": {
                "name": "local_file_data",
                "params_init": {"prices_path": "./fake.parquet"},
            },
        },
    }


class TestCLI:
    """CLI test suite using typer.testing.CliRunner."""

    # ------------------------------------------------------------------
    # 1. plugins list exits 0 and outputs plugin names
    # ------------------------------------------------------------------
    def test_plugins_list(self):
        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0, result.output
        # Should contain section headers and at least one known plugin
        assert "Pipelines:" in result.output
        assert "Strategies:" in result.output
        assert "Data:" in result.output
        assert "fund_selection.simple.v1" in result.output

    # ------------------------------------------------------------------
    # 2. plugins list --json outputs valid JSON
    # ------------------------------------------------------------------
    def test_plugins_list_json(self):
        result = runner.invoke(app, ["plugins", "list", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert isinstance(payload, dict)
        assert "pipelines" in payload
        assert "strategies" in payload
        assert "data" in payload
        assert "brokers" in payload
        # Check that at least one known plugin is present
        assert "fund_selection.simple.v1" in payload["pipelines"]
        assert "local_file_data" in payload["data"]

    # ------------------------------------------------------------------
    # 3. plugins info --name <valid> outputs plugin details
    # ------------------------------------------------------------------
    def test_plugins_info_valid(self):
        result = runner.invoke(app, ["plugins", "info", "--name", "strategy.crypto_trend.v1"])
        assert result.exit_code == 0, result.output
        assert "strategy.crypto_trend.v1" in result.output
        assert "strategy" in result.output

    # ------------------------------------------------------------------
    # 4. plugins info --name <nonexistent> exits non-zero or shows error
    # ------------------------------------------------------------------
    def test_plugins_info_nonexistent(self):
        result = runner.invoke(app, ["plugins", "info", "--name", "nonexistent.plugin.v99"])
        # PluginNotFoundError is raised inside the command, which typer
        # surfaces as a non-zero exit code (uncaught exception -> exit 1).
        assert result.exit_code != 0
        assert "plugin_not_found" in result.output or result.exception is not None

    # ------------------------------------------------------------------
    # 5. validate -c <valid_config> exits 0
    # ------------------------------------------------------------------
    def test_validate_valid_config(self, tmp_path):
        cfg = _minimal_valid_config()
        cfg_path = tmp_path / "valid.yaml"
        cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

        result = runner.invoke(app, ["validate", "-c", str(cfg_path)])
        assert result.exit_code == 0, result.output

    # ------------------------------------------------------------------
    # 6. validate -c <invalid_path> exits non-zero
    # ------------------------------------------------------------------
    def test_validate_invalid_path(self, tmp_path):
        missing = tmp_path / "does_not_exist.yaml"
        result = runner.invoke(app, ["validate", "-c", str(missing)])
        # Should fail because the file doesn't exist (FileNotFoundError).
        assert result.exit_code != 0

    # ------------------------------------------------------------------
    # 7. run --dry-run -c <config> exits 0
    # ------------------------------------------------------------------
    def test_run_dry_run(self, tmp_path):
        cfg = _minimal_valid_config()
        cfg_path = tmp_path / "dry_run.yaml"
        cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

        result = runner.invoke(app, ["run", "--dry-run", "-c", str(cfg_path)])
        assert result.exit_code == 0, result.output
        # dry-run prints a JSON plan with pipeline, data, mode, asof
        plan = json.loads(result.output)
        assert plan["mode"] == "backtest"
        assert plan["asof"] == "2026-01-31"
        assert plan["pipeline"] == "fund_selection.simple.v1"
        assert plan["data"] == "local_file_data"

    # ------------------------------------------------------------------
    # 8. --help exits 0
    # ------------------------------------------------------------------
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0, result.output
        assert "quantbox" in result.output.lower() or "Usage" in result.output
