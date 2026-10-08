"""The funding guard (TOM-1609): a funding series on an engine that does not charge it is refused.

vectorbt charges no funding (``charges_funding`` false, docs/adr/0008). A perps config
on it with a funding series ran silently and overstated the book. ``quantbox validate``,
``quantbox config explain`` and the run now refuse it with ONE check, unless the config
declares ``funding: {ignore: true, reason: ...}``; the run manifest records the reason.

One test per path (validate, explain, run with a planned file, run with a data plugin
that hands back funding without a file), and the declared escape on each.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from test_run_manifest_v1 import _config
from typer.testing import CliRunner

from quantbox.cli import app
from quantbox.exceptions import ConfigValidationError
from quantbox.explain import explain_config, validate_explain
from quantbox.plugins.pipeline.backtest_pipeline import BacktestPipeline
from quantbox.registry import PluginRegistry
from quantbox.run_manifest import validate_run_manifest
from quantbox.runner import run_from_config
from quantbox.store import FileArtifactStore
from quantbox.validate import validate_config

REFUSAL = "funding_not_charged"
REASON = "spot proxy: the perps funding is studied separately in TOM-0000"


def _ignore(cfg: dict, reason: str | None = REASON) -> dict:
    block: dict[str, Any] = {"ignore": True}
    if reason is not None:
        block["reason"] = reason
    cfg["plugins"]["pipeline"]["params"]["funding"] = block
    return cfg


def _write(tmp_path: Path, cfg: dict) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


# --------------------------------------------------------------------------- validate


def test_validate_refuses_a_funding_file_on_vectorbt(tmp_path):
    cfg_path = _write(tmp_path, _config(tmp_path, "vectorbt", ignore_funding=False))
    result = CliRunner().invoke(app, ["validate", "-c", str(cfg_path)])

    assert result.exit_code == 2, result.output
    assert REFUSAL in result.output
    assert "rsims" in result.output  # the message names the engine that charges funding


def test_validate_passes_the_same_config_on_rsims(tmp_path):
    findings = validate_config(_config(tmp_path, "rsims"))
    assert not [f for f in findings if f.level == "error"], [f.message for f in findings]


def test_validate_passes_vectorbt_with_a_declared_ignore(tmp_path):
    findings = validate_config(_ignore(_config(tmp_path, "vectorbt", ignore_funding=False)))
    assert not [f for f in findings if f.level == "error"], [f.message for f in findings]


def test_validate_refuses_an_ignore_without_a_reason(tmp_path):
    findings = validate_config(_ignore(_config(tmp_path, "vectorbt", ignore_funding=False), reason=None))
    errors = [f.message for f in findings if f.level == "error"]
    assert any("funding.reason" in m for m in errors), errors


def test_validate_refuses_an_ignore_on_an_engine_that_charges_funding(tmp_path):
    findings = validate_config(_ignore(_config(tmp_path, "rsims")))
    errors = [f.message for f in findings if f.level == "error"]
    assert any("charges funding" in m for m in errors), errors


# --------------------------------------------------------------------------- explain


def test_explain_refuses_a_funding_file_on_vectorbt(tmp_path):
    cfg = _config(tmp_path, "vectorbt", ignore_funding=False)
    doc = explain_config(cfg, PluginRegistry.discover())

    assert doc["ok"] is False
    assert any(REFUSAL in e for e in doc["errors"]), doc["errors"]
    assert validate_explain(doc) == []


def test_explain_plans_the_declared_ignore(tmp_path):
    doc = explain_config(_ignore(_config(tmp_path, "vectorbt", ignore_funding=False)), PluginRegistry.discover())

    assert doc["ok"] is True, doc["errors"]
    assert doc["funding"]["modelled"] is False
    assert doc["funding"]["ignored_reason"] == REASON
    assert validate_explain(doc) == []


# --------------------------------------------------------------------------- run


def test_run_refuses_a_funding_file_on_vectorbt_before_any_data_is_read(tmp_path):
    cfg = _config(tmp_path, "vectorbt", ignore_funding=False)
    with pytest.raises(ConfigValidationError, match=REFUSAL) as exc:
        run_from_config(cfg, PluginRegistry.discover())
    assert [f.code for f in exc.value.findings] == [REFUSAL]
    # Refused before the pipeline wrote anything.
    assert not list((tmp_path / "artifacts").rglob("returns*.parquet"))


def test_run_records_the_declared_ignore(tmp_path):
    cfg = _ignore(_config(tmp_path, "vectorbt", ignore_funding=False))
    result = run_from_config(copy.deepcopy(cfg), PluginRegistry.discover())
    manifest = json.loads((tmp_path / "artifacts" / result.run_id / "run_manifest.json").read_text())

    assert manifest["funding"]["modelled"] is False
    assert manifest["funding"]["ignored_reason"] == REASON
    assert validate_run_manifest(manifest) == []


def test_run_on_rsims_charges_funding_and_records_no_ignore(tmp_path):
    result = run_from_config(_config(tmp_path, "rsims"), PluginRegistry.discover())
    manifest = json.loads((tmp_path / "artifacts" / result.run_id / "run_manifest.json").read_text())

    assert manifest["funding"]["modelled"] is True
    assert "ignored_reason" not in manifest["funding"]


# A data plugin with no planned file (an API plugin: binance futures, hyperliquid) is seen
# only once it hands the frames back; the run refuses it there, before any strategy runs.


def _prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    a = 100.0 * np.cumprod(1 + np.random.default_rng(3).normal(0, 0.01, len(idx)))
    return pd.DataFrame({"A": a, "USD": 100.0}, index=idx)


class _ApiData:
    meta = type("M", (), {"name": "fake.api_funding.v1"})()

    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame({"symbol": ["A", "USD"]})

    def load_market_data(self, universe: Any, asof: str, params: dict[str, Any]) -> dict[str, pd.DataFrame]:
        prices = _prices()
        return {"prices": prices, "funding_rates": pd.DataFrame({"A": 0.0001}, index=prices.index)}


class _Hold:
    meta = type("M", (), {"name": "strategy.hold.v1"})()
    ran = False

    def run(self, data: Any, params: Any = None) -> dict[str, Any]:
        _Hold.ran = True
        return {"weights": pd.DataFrame({"A": 1.0, "USD": 0.0}, index=_prices().index)}


def _run_api(tmp_path, **params):
    _Hold.ran = False
    return BacktestPipeline().run(
        mode="backtest",
        asof="2024-02-09",
        params={"fees": 0.0, "strategies": [{"name": "strategy.hold.v1", "weight": 1.0}], **params},
        data=_ApiData(),
        store=FileArtifactStore(str(tmp_path), "run"),
        broker=None,
        risk=[],
        strategies=[_Hold()],
    )


def test_run_refuses_funding_a_data_plugin_hands_back_on_vectorbt(tmp_path):
    with pytest.raises(ConfigValidationError, match=REFUSAL) as exc:
        _run_api(tmp_path, engine="vectorbt")
    assert "fake.api_funding.v1" in str(exc.value)
    assert _Hold.ran is False  # refused before any strategy ran


def test_a_data_plugin_funding_on_vectorbt_runs_with_a_declared_ignore(tmp_path):
    result = _run_api(tmp_path, engine="vectorbt", funding={"ignore": True, "reason": REASON})
    assert result.notes["funding"] == {"modelled": False, "ignored_reason": REASON}


def test_a_data_plugin_funding_on_rsims_runs(tmp_path):
    result = _run_api(tmp_path, engine="rsims")
    assert result.notes["funding"]["modelled"] is True
