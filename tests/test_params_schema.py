"""TOM-1350: every registered plugin carries a params schema; validate refuses unknown keys."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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


# --- the schema is exactly what the plugin accepts (round 3 of #218) ------------
#
# ``validate`` accepts a schema property under ``params_init`` when it is a
# constructor parameter and under ``params`` always; so every property must be
# either a constructor parameter or a key the plugin reads off ``params`` at run
# time, and every constructor parameter must also be honoured at run time. A
# property nothing reads is a knob a config can set and the run silently drops
# (data.synthetic.v1 listed n_steps/model/... that only load_market_data's
# per-call params carried, so ``params_init: {n_steps: 504}`` was refused with a
# hint pointing at ``params``, which nothing reads either).
#
# "Reads" is read off the source: ``params.get("k")`` / ``params["k"]`` /
# ``"k" in params`` / ``params.pop|setdefault("k")`` in the plugin's module, or
# the override loop ``for k, v in params.items(): setattr(self, ...)`` that makes
# every attribute (and every ``_PARAM_ALIASES`` key) a run-time key.

# The runner never forwards these blocks' ``params`` (validate._UNREAD_PARAMS):
# their schema is the constructor and nothing else.
_INIT_ONLY_GROUPS = {"data", "broker"}
# Keys a pipeline ASSIGNS into a plugin's params; a config's copy is overwritten.
_PIPELINE_INJECTED = {"rebalancing": {"mode", "strategy_results"}}
# Plugins whose module hands ``params`` to a reader elsewhere.
_READ_ELSEWHERE = {
    # The rsims adapter reads its own params (docs/adr/0008).
    "backtest.pipeline.v1": ("quantbox.frequency", "quantbox.engine.rsims"),
    "trade.full_pipeline.v1": ("quantbox.frequency",),
}


def _ast_of(module_name: str):
    import ast
    import importlib
    import inspect

    return ast.parse(inspect.getsource(importlib.import_module(module_name)))


def _keys_read_from_params(tree) -> set[str]:
    import ast

    def is_params(node) -> bool:
        return isinstance(node, ast.Name) and node.id == "params"

    def const(node):
        return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

    keys: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and is_params(n.func.value):
            if n.func.attr in ("get", "pop", "setdefault") and n.args and const(n.args[0]):
                keys.add(const(n.args[0]))
        elif isinstance(n, ast.Subscript) and is_params(n.value) and const(n.slice):
            keys.add(const(n.slice))
        elif (
            isinstance(n, ast.Compare)
            and len(n.ops) == 1
            and isinstance(n.ops[0], (ast.In, ast.NotIn))
            and is_params(n.comparators[0])
            and const(n.left)
        ):
            keys.add(const(n.left))
    return keys


def _copies_params_onto_attributes(tree) -> bool:
    import ast

    for n in ast.walk(tree):
        it = n.iter if isinstance(n, ast.For) else None
        if (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Attribute)
            and it.func.attr == "items"
            and isinstance(it.func.value, ast.Name)
            and it.func.value.id == "params"
            and any(isinstance(m, ast.Call) and getattr(m.func, "id", None) == "setattr" for m in ast.walk(n))
        ):
            return True
    return False


def _param_aliases(tree) -> dict[str, str]:
    import ast

    out: dict[str, str] = {}
    for n in ast.walk(tree):
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AnnAssign) else []
        if not any(getattr(t, "id", getattr(t, "attr", None)) == "_PARAM_ALIASES" for t in targets) or n.value is None:
            continue
        for d in ast.walk(n.value):
            if isinstance(d, ast.Dict):
                out.update({k.value: v.value for k, v in zip(d.keys, d.values, strict=True)})
    return out


def _runtime_keys(name, cls) -> tuple[set[str], set[str], bool]:
    """(keys read off ``params``, keys accepted at run time, whether the override loop is used)."""
    tree = _ast_of(cls.__module__)
    read = _keys_read_from_params(tree)
    for mod in _READ_ELSEWHERE.get(name, ()):
        read |= _keys_read_from_params(_ast_of(mod))
    accepted = set(read)
    loop = _copies_params_onto_attributes(tree)
    if loop:
        attrs = {p.name for p in config_fields(cls)} | {a for a in dir(cls) if not a.startswith("_")}
        accepted |= attrs | {k for k, v in _param_aliases(tree).items() if v in attrs}
    return read, accepted, loop


@pytest.mark.parametrize(("group", "name", "cls"), ALL, ids=IDS)
def test_every_schema_property_is_accepted_by_the_plugin(group, name, cls):
    """A property is a constructor parameter (params_init) or a key run() reads (params)."""
    schema = resolve_params_schema(cls) or {"properties": {}}
    init = {p.name for p in config_fields(cls)}
    runtime = set() if group in _INIT_ONLY_GROUPS else _runtime_keys(name, cls)[1]
    unread = sorted(set(schema["properties"]) - init - runtime)
    where = "is not a constructor parameter" + (
        " (its `params` block is never read)" if group in _INIT_ONLY_GROUPS else " and nothing reads it off params"
    )
    assert not unread, f"{name}: schema properties {unread} {where}"


@pytest.mark.parametrize(
    ("group", "name", "cls"),
    [a for a in ALL if a[0] not in _INIT_ONLY_GROUPS],
    ids=[i for i, a in zip(IDS, ALL, strict=True) if a[0] not in _INIT_ONLY_GROUPS],
)
def test_every_runtime_key_is_a_schema_property(group, name, cls):
    """The reverse: a key run() reads that the schema omits is refused by validate yet honoured by the run."""
    schema = resolve_params_schema(cls) or {"properties": {}}
    read = _runtime_keys(name, cls)[0]
    private = {k for k in read if k.startswith("_")}
    missing = sorted(read - set(schema["properties"]) - private - _PIPELINE_INJECTED.get(group, set()))
    assert not missing, f"{name}: reads {missing} off params but its schema does not declare them"


@pytest.mark.parametrize(
    ("group", "name", "cls"),
    [a for a in ALL if a[0] not in _INIT_ONLY_GROUPS],
    ids=[i for i, a in zip(IDS, ALL, strict=True) if a[0] not in _INIT_ONLY_GROUPS],
)
def test_every_constructor_param_is_honoured_at_run_time(group, name, cls):
    """validate accepts a constructor parameter under ``params`` too, so run() must read it there."""
    read, _, loop = _runtime_keys(name, cls)
    if loop:
        return  # the override loop sets every attribute from params
    ignored = sorted({p.name for p in config_fields(cls)} - read)
    assert not ignored, f"{name}: constructor params {ignored} are accepted under params but run() never reads them"


def test_schema_accepted_check_catches_an_unread_property():
    """The guard is not vacuous: a declared key no one reads is caught, a read one is not."""
    import ast

    tree = ast.parse("def run(self, data, params):\n    return params.get('window', 3)\n")
    assert _keys_read_from_params(tree) == {"window"}
    assert not _copies_params_onto_attributes(tree)
    loop = ast.parse(
        "_PARAM_ALIASES = {'lookback': 'window'}\n"
        "def run(self, data, params):\n    for k, v in params.items():\n        setattr(self, k, v)\n"
    )
    assert _copies_params_onto_attributes(loop)
    assert _param_aliases(loop) == {"lookback": "window"}


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


def test_params_init_accepts_constructor_params_only():
    """params_init goes to the constructor: a run-time key there would raise TypeError at run."""
    cfg = _config({})
    cfg["plugins"]["pipeline"]["params_init"] = {"engine": "rsims"}
    msgs = [f.message for f in validate_config(cfg, REG) if f.level == "error"]
    assert any("'engine'" in m and "params_init" in m and "run-time param" in m for m in msgs), msgs


@pytest.mark.parametrize(
    ("slot", "block", "key"),
    [
        ("data", {"name": "data.synthetic.v1", "params": {"n_steps": 504}}, "n_steps"),
        ("broker", {"name": "sim.paper.v1", "params": {"spread_bps": 2.0}}, "spread_bps"),
    ],
)
def test_data_and_broker_params_are_refused_because_nothing_reads_them(slot, block, key):
    """review of #218: the runner builds data/broker plugins from params_init alone and the
    pipelines feed load_universe/load_market_data from pipeline.params.universe/prices, so a
    schema-valid key under plugins.data.params / plugins.broker.params is silently dropped."""
    cfg = _config({})
    cfg["plugins"][slot] = block
    msgs = [f.message for f in validate_config(cfg, REG) if f.level == "error"]
    assert any(f"'{key}'" in m and f"plugins.{slot}.params" in m and "never read" in m for m in msgs), msgs


def test_data_params_hint_names_where_load_params_go():
    cfg = _config({})
    cfg["plugins"]["data"] = {"name": "data.synthetic.v1", "params": {"n_steps": 504}}
    msgs = [f.message for f in validate_config(cfg, REG) if f.level == "error"]
    assert any("plugins.pipeline.params.universe" in m and "prices" in m for m in msgs), msgs


def test_synthetic_params_init_validates_and_reaches_the_generator():
    """round 3 of #218: ``params_init: {n_steps: 504}`` was refused ("set it under params",
    which nothing reads) because n_steps was a schema property but no constructor parameter."""
    from quantbox.plugins.datasources.synthetic_data import SyntheticDataPlugin

    init = {"n_assets": 3, "n_steps": 504, "model": "jump_diffusion", "random_state": 7}
    cfg = _config({})
    cfg["plugins"]["data"] = {"name": "data.synthetic.v1", "params_init": init}
    assert not [f.message for f in validate_config(cfg, REG) if f.level == "error"]

    plugin = SyntheticDataPlugin(**init)
    universe = plugin.load_universe({})
    assert plugin.load_market_data(universe, "2026-01-31", {})["prices"].shape == (504 + 1, 3)
    # the per-call params (pipeline.params.universe / .prices) still override the constructor
    assert plugin.load_market_data(universe, "2026-01-31", {"n_steps": 10})["prices"].shape == (10 + 1, 3)


def test_validate_unregistered_plugin_is_a_warning_not_an_error():
    cfg = _config({})
    cfg["plugins"]["strategies"] = [{"name": "lab.strategy.elsewhere.v1", "params": {"x": 1}}]
    findings = validate_config(cfg, REG)
    assert not [f for f in findings if f.level == "error"]
    assert any("params_not_checked:lab.strategy.elsewhere.v1" in f.message for f in findings)


# --- the repo's own examples validate clean (review of #218) --------------------

COOKBOOK = sorted((Path(__file__).resolve().parents[1] / "cookbook").glob("**/configs/*.yaml"))
RUN_CONFIGS = [p for p in COOKBOOK if "run" in (yaml.safe_load(p.read_text(encoding="utf-8")) or {})]


def test_cookbook_run_configs_are_found():
    """A glob that matches nothing would make the parametrized test below vacuously green."""
    assert len(RUN_CONFIGS) >= 15, [p.name for p in COOKBOOK]


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=[p.name for p in RUN_CONFIGS])
def test_cookbook_config_validates_clean(path):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors = [f.message for f in validate_config(cfg, REG, check_params=True) if f.level == "error"]
    assert not errors, f"{path.name}: {errors}"


# --- the legacy pipeline.params.strategies path is checked too -------------------


def _legacy_strategies_config(strategy_params: dict, name: str = "carry") -> dict:
    cfg = _config({})
    del cfg["plugins"]["strategies"]
    cfg["plugins"]["pipeline"]["params"]["strategies"] = [{"name": name, "weight": 1.0, "params": strategy_params}]
    return cfg


def test_validate_checks_pipeline_params_strategies():
    """backtest/trading pipelines run ``pipeline.params.strategies`` when ``plugins.strategies`` is absent:
    each ``name`` is a module under ``quantbox.plugins.strategies`` whose ``run()`` takes ``params``."""
    msgs = [f.message for f in validate_config(_legacy_strategies_config({"signal_span_dayz": 5}), REG)]
    assert any(
        "'signal_span_dayz'" in m and "strategy.carry.v1" in m and "pipeline.params.strategies[0]" in m for m in msgs
    ), msgs


def test_pipeline_params_strategies_valid_params_pass():
    findings = validate_config(_legacy_strategies_config({"signal_span_days": 5}), REG)
    assert not [f for f in findings if f.level == "error"], findings
    assert not [f for f in findings if "params_not_checked" in f.message], findings


def test_pipeline_params_strategies_unknown_module_is_a_warning():
    findings = validate_config(_legacy_strategies_config({"x": 1}, name="no_such_module"), REG)
    assert not [f for f in findings if f.level == "error"], findings
    assert any("params_not_checked:no_such_module" in f.message for f in findings), findings


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


class _Reached(Exception):
    pass


def test_synthetic_cookbook_knobs_reach_the_data_plugin(tmp_path, monkeypatch):
    """The synthetic example's knobs must be what the run generates, not the plugin defaults."""
    from quantbox.plugins.datasources.synthetic_data import SyntheticDataPlugin
    from quantbox.runner import run_from_config

    seen: dict[str, dict] = {}
    orig_u, orig_m = SyntheticDataPlugin.load_universe, SyntheticDataPlugin.load_market_data

    def spy_u(self, params):
        seen["universe"] = dict(params)
        return orig_u(self, params)

    def spy_m(self, universe, asof, params):
        seen["prices"] = dict(params)
        out = orig_m(self, universe, asof, params)
        seen["shape"] = out["prices"].shape
        # Stop here: what is under test is the routing of the knobs. (The synthetic plugin's
        # load_universe returns a list, which the backtest pipeline cannot store as parquet.)
        raise _Reached

    monkeypatch.setattr(SyntheticDataPlugin, "load_universe", spy_u)
    monkeypatch.setattr(SyntheticDataPlugin, "load_market_data", spy_m)
    path = next(p for p in RUN_CONFIGS if p.name == "run_synthetic_backtest.yaml")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["artifacts"]["root"] = str(tmp_path)
    with pytest.raises(_Reached):
        run_from_config(cfg, REG)
    assert seen["universe"] == {"n_assets": 10}
    assert seen["prices"]["model"] == "jump_diffusion"
    assert seen["shape"] == (504 + 1, 10)  # n_steps plus the initial bar; plugin defaults give (253, 10)
