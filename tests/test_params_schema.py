"""TOM-1350: every registered plugin carries a params schema; validate refuses unknown keys."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
import yaml
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from quantbox.cli import app
from quantbox.contracts import PluginMeta
from quantbox.params_schema import (
    CATALOG_SCHEMA,
    PLUGIN_GROUPS,
    config_fields,
    resolve_params_schema,
)
from quantbox.registry import PluginRegistry
from quantbox.validate import validate_config

runner = CliRunner()
REG = PluginRegistry.discover()
ALL = [(g, n, c) for g, attr in PLUGIN_GROUPS.items() for n, c in sorted(getattr(REG, attr).items())]
IDS = [f"{g}:{n}" for g, n, _ in ALL]


# --- the contract: mandatory, complete, derived from the dataclass ----------------


@pytest.mark.parametrize(("group", "name", "cls"), ALL, ids=IDS)
def test_every_registered_plugin_declares_params_schema(group, name, cls):
    assert cls.meta.params_schema is not None, f"{group} plugin {name} declares no params_schema"


@pytest.mark.parametrize(("group", "name", "cls"), ALL, ids=IDS)
def test_params_schema_is_complete(group, name, cls):
    schema = resolve_params_schema(cls)
    Draft202012Validator.check_schema(schema)
    for key, prop in schema["properties"].items():
        assert prop.get("type"), f"{name}.{key}: no type"
        assert str(prop.get("description", "")).strip(), f"{name}.{key}: no description"


@pytest.mark.parametrize(("group", "name", "cls"), ALL, ids=IDS)
def test_schema_covers_every_constructor_param_with_its_real_default(group, name, cls):
    """The schema is derived from the constructor: a declared default may not contradict it."""
    declared = (cls.meta.params_schema or {}).get("properties", {})
    resolved = resolve_params_schema(cls)["properties"]
    for p in config_fields(cls):
        assert p.name in resolved
        if "default" in declared.get(p.name, {}):
            real = json.loads(json.dumps(p.default))
            assert declared[p.name]["default"] == real, f"{name}.{p.name}: schema default != constructor default"


def test_missing_schema_is_reported_as_missing():
    @dataclass
    class NoSchema:
        meta = PluginMeta(name="t.none.v1", kind="strategy", version="0", core_compat="*")
        window: int = 5

    assert resolve_params_schema(NoSchema) is None


def test_dataclass_fields_fill_type_and_default():
    @dataclass
    class Tiny:
        meta = PluginMeta(
            name="t.tiny.v1",
            kind="strategy",
            version="0",
            core_compat="*",
            params_schema={"properties": {"window": {"description": "lookback", "minimum": 1}}},
        )
        window: int = 5
        tickers: tuple[str, ...] = ("BTC",)
        cap: float | None = None
        _private: int = 0

    props = resolve_params_schema(Tiny)["properties"]
    assert props["window"] == {"type": "integer", "default": 5, "description": "lookback", "minimum": 1}
    assert props["tickers"] == {"type": "array", "default": ["BTC"]}
    assert props["cap"] == {"type": ["number", "null"], "default": None}
    assert "_private" not in props


# --- plugins schema --json --------------------------------------------------------


def test_plugins_schema_json_lists_every_plugin_with_param_rows():
    result = runner.invoke(app, ["plugins", "schema", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    Draft202012Validator(CATALOG_SCHEMA).validate(payload)
    by_id = {p["id"]: p for p in payload["plugins"]}
    assert set(by_id) == {n for _, n, _ in ALL}
    carry = by_id["strategy.carry.v1"]
    assert carry["status"] == "research"
    row = next(r for r in carry["params"] if r["name"] == "signal_span_days")
    assert row == {
        "name": "signal_span_days",
        "type": "integer",
        "default": 3,
        "description": row["description"],
        "required": False,
    }
    assert row["description"]


def test_plugins_schema_name_filter():
    result = runner.invoke(app, ["plugins", "schema", "--json", "--name", "strategy.carry.v1"])
    assert result.exit_code == 0, result.output
    assert [p["id"] for p in json.loads(result.output)["plugins"]] == ["strategy.carry.v1"]


# --- validate fails loudly --------------------------------------------------------


def _config(strategy_params: dict) -> dict:
    return {
        "run": {"mode": "backtest", "asof": "2026-01-31"},
        "artifacts": {"root": "./artifacts"},
        "plugins": {
            "pipeline": {"name": "backtest.pipeline.v1", "params": {"engine": "vectorbt", "fees": 0.001}},
            "data": {"name": "local_file_data", "params_init": {"prices_path": "./p.parquet"}},
            "strategies": [{"name": "strategy.carry.v1", "weight": 1.0, "params": strategy_params}],
        },
    }


def _invoke_validate(tmp_path, cfg):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return runner.invoke(app, ["validate", "-c", str(path)])


def test_validate_misspelled_param_exits_nonzero_naming_key_and_plugin(tmp_path):
    result = _invoke_validate(tmp_path, _config({"signal_span_dayz": 5}))
    assert result.exit_code != 0
    assert "signal_span_dayz" in result.output
    assert "strategy.carry.v1" in result.output
    assert "did you mean 'signal_span_days'" in result.output


def test_validate_schema_violation_exits_nonzero(tmp_path):
    result = _invoke_validate(tmp_path, _config({"signal_span_days": "three"}))
    assert result.exit_code != 0
    assert "invalid_param" in result.output and "strategy.carry.v1" in result.output


def test_validate_valid_config_prints_explicit_ok(tmp_path):
    result = _invoke_validate(tmp_path, _config({"signal_span_days": 5, "top_n_long": 2}))
    assert result.exit_code == 0, result.output
    assert "OK:" in result.output


def test_validate_checks_params_init_and_variants():
    cfg = _config({})
    cfg["plugins"]["data"]["params_init"]["price_path"] = "./typo.parquet"
    cfg["plugins"]["pipeline"]["params"]["variants"] = [
        {"name": "v1", "strategy": {"name": "strategy.carry.v1", "params": {"top_n": 2}}}
    ]
    msgs = [f.message for f in validate_config(cfg, REG) if f.level == "error"]
    assert any("'price_path'" in m and "local_file_data" in m for m in msgs), msgs
    assert any("'top_n'" in m and "variants[0]" in m for m in msgs), msgs


def test_validate_unregistered_plugin_is_a_warning_not_an_error():
    cfg = _config({})
    cfg["plugins"]["strategies"] = [{"name": "lab.strategy.elsewhere.v1", "params": {"x": 1}}]
    findings = validate_config(cfg, REG)
    assert not [f for f in findings if f.level == "error"]
    assert any("params_not_checked:lab.strategy.elsewhere.v1" in f.message for f in findings)


def test_run_warns_on_unknown_param_but_does_not_refuse(caplog):
    """A run only WARNS: configs in use carry keys their plugins always ignored (see runner)."""
    from unittest.mock import MagicMock, patch

    from quantbox.contracts import RunResult
    from quantbox.plugins.strategies import CarryStrategy
    from quantbox.runner import run_from_config

    pipeline = MagicMock()
    pipeline.kind = "research"
    pipeline.meta = PluginMeta(name="test.pipeline.v1", kind="pipeline", version="1", core_compat="*")
    pipeline.run.return_value = RunResult(
        run_id="r",
        pipeline_name="test.pipeline.v1",
        mode="backtest",
        asof="2026-01-31",
        artifacts={},
        metrics={},
        notes={},
    )
    reg = MagicMock()
    reg.pipelines = {"test.pipeline.v1": MagicMock(return_value=pipeline)}
    reg.data = {"test.data.v1": MagicMock()}
    reg.strategies = {"strategy.carry.v1": CarryStrategy}
    cfg = {
        "run": {"mode": "backtest", "asof": "2026-01-31", "pipeline": "test.pipeline.v1"},
        "artifacts": {"root": "/tmp/quantbox_test_artifacts"},
        "plugins": {
            "pipeline": {"name": "test.pipeline.v1", "params": {}},
            "data": {"name": "test.data.v1"},
            "strategies": [{"name": "strategy.carry.v1", "params": {"signal_span_dayz": 5}}],
        },
    }
    with patch("quantbox.runner.FileArtifactStore"), caplog.at_level("WARNING", logger="quantbox.runner"):
        run_from_config(cfg, reg)
    assert pipeline.run.called
    assert any("signal_span_dayz" in r.getMessage() for r in caplog.records)
