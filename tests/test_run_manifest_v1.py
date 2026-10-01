"""Run manifest ``quantbox/run@1`` and ``quantbox run --json`` (TOM-1348).

The manifest is the machine-readable record of a run. These tests are its
contract: a backtest on EACH engine writes a manifest that validates against
the schema shipped in the package, every file it lists exists, and the CLI
emits exactly that manifest on stdout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from quantbox.cli import app
from quantbox.registry import PluginRegistry
from quantbox.run_manifest import SCHEMA_ID, load_run_schema, run_files, validate_run_manifest
from quantbox.runner import run_from_config

GOLDEN = Path(__file__).parent / "fixtures" / "golden_run"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    rng = np.random.default_rng(7)
    a = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, len(idx)))
    prices = pd.DataFrame({"A": a, "USD": 100.0}, index=idx)
    long = prices.rename_axis("date").reset_index().melt("date", var_name="symbol", value_name="close")
    prices_path = tmp_path / "prices.parquet"
    long.to_parquet(prices_path, index=False)
    funding = pd.DataFrame({"A": 0.0001, "USD": 0.0}, index=idx)
    flong = funding.rename_axis("date").reset_index().melt("date", var_name="symbol", value_name="rate")
    funding_path = tmp_path / "funding.parquet"
    flong.to_parquet(funding_path, index=False)
    return prices_path, funding_path


def _config(tmp_path: Path, engine: str, *, n_trials: int | None = None, hold: str = "A") -> dict:
    prices_path, funding_path = _write_inputs(tmp_path)
    cfg = yaml.safe_load(f"""
run: {{mode: backtest, asof: "2024-02-09", pipeline: backtest.pipeline.v1}}
artifacts: {{root: "{tmp_path / "artifacts"}"}}
plugins:
  pipeline:
    name: backtest.pipeline.v1
    params:
      engine: {engine}
      fees: 0.0
      venue: {{allow_shorts: false}}
      risk: {{max_leverage: 1.5}}
      universe: {{symbols: [A, USD]}}
  strategies:
    - name: strategy.static_weights.v1
      weight: 1.0
      params_init: {{weights: {{{hold}: 1.0}}}}
  data:
    name: local_file_data
    params_init: {{prices_path: "{prices_path}", funding_rates_path: "{funding_path}"}}
""")
    if n_trials is not None:
        cfg["run"]["n_trials"] = n_trials
    return cfg


def _run(tmp_path: Path, engine: str, **kw) -> tuple[dict, Path]:
    cfg = _config(tmp_path, engine, **kw)
    result = run_from_config(cfg, PluginRegistry.discover())
    run_dir = tmp_path / "artifacts" / result.run_id
    return json.loads((run_dir / "run_manifest.json").read_text()), run_dir


@pytest.mark.parametrize("engine", ["vectorbt", "rsims"])
def test_backtest_manifest_validates_and_every_listed_file_exists(tmp_path, engine):
    manifest, run_dir = _run(tmp_path, engine)

    assert validate_run_manifest(manifest) == []
    assert manifest["schema"] == SCHEMA_ID == "quantbox/run@1"

    files = run_files(manifest)
    assert set(files) == {"returns", "traded_weights", "metrics"}
    for logical, rel in files.items():
        assert rel is not None, f"{logical} not listed"
        assert not Path(rel).is_absolute(), f"{logical} path must be relative to the run dir: {rel}"
        assert (run_dir / rel).is_file(), f"{logical} listed but missing: {rel}"


@pytest.mark.parametrize("engine", ["vectorbt", "rsims"])
def test_manifest_records_engine_dataset_funding_execution_venue(tmp_path, engine):
    manifest, _ = _run(tmp_path, engine)

    assert manifest["run_id"]
    assert len(manifest["config"]["sha256"]) == 64
    assert "commit" in manifest["git"]

    assert manifest["engine"]["name"] == engine
    assert manifest["engine"]["version"]

    prices_path = tmp_path / "prices.parquet"
    assert manifest["dataset"]["source"] == "inline"
    assert manifest["dataset"]["sha256"] == _sha256(prices_path)
    assert manifest["dataset"]["name"]

    funding_path = tmp_path / "funding.parquet"
    assert manifest["funding"]["modelled"] is (engine == "rsims")
    assert manifest["funding"]["source_path"] == str(funding_path)
    assert manifest["funding"]["sha256"] == _sha256(funding_path)

    assert manifest["execution"]["lag_bars"] == 1
    assert manifest["execution"]["description"]
    assert manifest["venue"]["allow_shorts"] is False
    assert manifest["venue"]["max_leverage"] == 1.5

    assert manifest["n_trials"] is None
    assert isinstance(manifest["metrics"], dict) and manifest["metrics"]


def _strict_loads(text: str):
    def refuse(token):
        raise ValueError(f"non-standard JSON token {token!r}")

    return json.loads(text, parse_constant=refuse)


@pytest.mark.parametrize("engine", ["vectorbt", "rsims"])
def test_manifest_is_strict_json_non_finite_metrics_become_null(tmp_path, engine):
    """jq / JS parsers refuse NaN and Infinity; an undefined metric is written as null.

    Holding only the constant `USD` gives a book with no losing bar, whose
    profit factor is undefined (Python computes +Infinity)."""
    _, run_dir = _run(tmp_path, engine, hold="USD")
    manifest = _strict_loads((run_dir / "run_manifest.json").read_text())
    assert manifest["metrics"]["profit_factor"] is None
    assert validate_run_manifest(manifest) == []


def test_json_safe_maps_non_finite_floats_to_null():
    from quantbox.run_manifest import json_safe

    assert json_safe({"a": float("inf"), "b": [float("nan"), 1.0], "c": "x"}) == {"a": None, "b": [None, 1.0], "c": "x"}


def test_n_trials_is_recorded_when_the_config_states_it(tmp_path):
    manifest, _ = _run(tmp_path, "vectorbt", n_trials=12)
    assert manifest["n_trials"] == 12
    assert validate_run_manifest(manifest) == []


@pytest.mark.parametrize("bad", [0, -3, "many", True, 2.5])
def test_a_malformed_n_trials_is_refused_before_the_run(tmp_path, bad):
    cfg = _config(tmp_path, "vectorbt", n_trials=1)
    cfg["run"]["n_trials"] = bad
    with pytest.raises(ValueError, match="n_trials"):
        run_from_config(cfg, PluginRegistry.discover())
    assert not (tmp_path / "artifacts").exists() or not any((tmp_path / "artifacts").rglob("run_manifest.json"))


def test_a_run_that_simulates_nothing_still_writes_a_valid_manifest(tmp_path):
    cfg = yaml.safe_load(Path("cookbook/configs/run_fund_selection.yaml").read_text(encoding="utf-8"))
    cfg["artifacts"]["root"] = str(tmp_path)
    result = run_from_config(cfg, PluginRegistry.discover())
    manifest = json.loads((tmp_path / result.run_id / "run_manifest.json").read_text())
    assert manifest["engine"] is None
    assert validate_run_manifest(manifest) == []


def test_schema_ships_with_the_package_and_rejects_a_manifest_without_an_engine():
    schema = load_run_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    golden = json.loads((GOLDEN / "run_manifest.json").read_text())
    broken = {k: v for k, v in golden.items() if k != "engine"}
    assert validate_run_manifest(broken), "a manifest without `engine` must not validate"
    wrong_major = {**golden, "schema": "quantbox/run@2"}
    assert validate_run_manifest(wrong_major), "a run@2 manifest must not validate as run@1"


def test_a_backtest_manifest_must_list_its_files():
    golden = json.loads((GOLDEN / "run_manifest.json").read_text())
    unlisted = {**golden, "files": {**golden["files"], "traded_weights": None}}
    assert validate_run_manifest(unlisted), "a backtest that does not point at traded weights must not validate"


def test_golden_run_directory_validates_and_its_files_exist():
    manifest = _strict_loads((GOLDEN / "run_manifest.json").read_text())
    assert validate_run_manifest(manifest) == []
    for logical, rel in run_files(manifest).items():
        assert (GOLDEN / rel).is_file(), f"golden {logical} missing: {rel}"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _cli_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_config(tmp_path, "vectorbt")))
    return cfg_path


def test_run_json_prints_only_the_manifest_on_stdout(tmp_path):
    cfg_path = _cli_config(tmp_path)
    result = CliRunner().invoke(app, ["run", "-c", str(cfg_path), "--json"])
    assert result.exit_code == 0, result.stderr
    printed = json.loads(result.stdout)  # the WHOLE of stdout is one JSON document
    on_disk = json.loads((tmp_path / "artifacts" / printed["run_id"] / "run_manifest.json").read_text())
    assert printed == on_disk
    assert validate_run_manifest(printed) == []


def test_run_summary_out_writes_the_manifest(tmp_path):
    cfg_path = _cli_config(tmp_path)
    out = tmp_path / "summary.json"
    result = CliRunner().invoke(app, ["run", "-c", str(cfg_path), "--summary-out", str(out)])
    assert result.exit_code == 0, result.stderr
    summary = json.loads(out.read_text())
    assert summary["schema"] == SCHEMA_ID
    assert "RUN_ID:" in result.stdout  # the human output is unchanged without --json


def test_run_json_fails_non_zero_with_nothing_on_stdout(tmp_path):
    cfg = _config(tmp_path, "vectorbt")
    cfg["plugins"]["pipeline"]["params"]["engine"] = "no-such-engine"
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    result = CliRunner().invoke(app, ["run", "-c", str(cfg_path), "--json"])
    assert result.exit_code != 0
    assert result.stdout == ""
