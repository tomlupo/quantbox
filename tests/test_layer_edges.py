"""TOM-1451 (4b-1): the edges that crossed the layer map are cut, and the old paths still work.

Layers (ADR-0010): core < plugins < research, trade. These tests pin the cuts
by what a fresh interpreter IMPORTS, so an eager import that comes back goes
red here before import-linter (4b-3) exists:

- the registry holds builtins as ``"module:Class"`` and imports a plugin only
  when it is asked for;
- the engine seam does not import the vectorbt adapter until it is named;
- ``quantbox.plugins.pipeline`` imports neither the backtest nor the trading
  pipeline until one is named, and neither drags the other;
- ``from quantbox import *`` imports core only;
- every moved module keeps its old path as a shim (same object, DeprecationWarning).
"""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
import warnings

import pytest

from quantbox._lazy import load
from quantbox.exceptions import MissingExtraError
from quantbox.plugins.builtins import BUILTIN_PLUGINS
from quantbox.registry import PluginMap, PluginRegistry


def _modules_after(code: str) -> set[str]:
    """The quantbox modules a fresh interpreter has imported after running *code*."""
    probe = code + "\nimport json, sys\nprint(json.dumps(sorted(m for m in sys.modules if m.startswith('quantbox'))))"
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    mods = set(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert "quantbox" in mods, "the probe imported nothing — it is blind"
    return mods


BROKERS = "quantbox.plugins.broker"
TRADING = "quantbox.plugins.pipeline.trading_pipeline"
BACKTEST = "quantbox.plugins.pipeline.backtest_pipeline"


# ----------------------------------------------------------------------
# The builtin table
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "name", "target"),
    [(k, n, t) for k, table in BUILTIN_PLUGINS.items() for n, t in table.items()],
)
def test_every_builtin_entry_resolves_to_a_class_named_by_its_key(kind, name, target):
    cls = load(target)
    assert isinstance(cls, type)
    assert cls.meta.name == name, f"{kind} {name!r} -> {target} declares meta.name {cls.meta.name!r}"


def test_the_builtin_table_counts_every_kind():
    # A table that lost a kind (or all its rows) must not pass the test above vacuously.
    counts = {k: len(v) for k, v in BUILTIN_PLUGINS.items()}
    assert counts["strategy"] > 20 and counts["broker"] > 5 and counts["pipeline"] == 4, counts


# ----------------------------------------------------------------------
# The registry is lazy
# ----------------------------------------------------------------------


def test_discover_imports_no_plugin_module():
    mods = _modules_after("from quantbox.registry import PluginRegistry\nPluginRegistry.discover()")
    assert not {m for m in mods if m.startswith("quantbox.plugins.") and m != "quantbox.plugins.builtins"}


def test_listing_names_imports_no_plugin_module():
    mods = _modules_after(
        "from quantbox.registry import PluginRegistry\n"
        "r = PluginRegistry.discover()\n"
        "names = sorted(r.pipelines) + sorted(r.brokers.keys()) + [len(r.strategies)]\n"
        "assert 'backtest.pipeline.v1' in r.pipelines"
    )
    assert not {m for m in mods if m.startswith("quantbox.plugins.") and m != "quantbox.plugins.builtins"}


def test_a_backtest_lookup_imports_no_broker_and_no_trading_pipeline():
    mods = _modules_after(
        "from quantbox.registry import PluginRegistry\n"
        "r = PluginRegistry.discover()\n"
        "r.pipelines['backtest.pipeline.v1']; r.strategies['strategy.static_weights.v1']; r.data['data.synthetic.v1']"
    )
    assert BACKTEST in mods
    assert TRADING not in mods
    assert not {m for m in mods if m.startswith(BROKERS)}


def test_a_trading_lookup_imports_no_backtest_pipeline():
    mods = _modules_after(
        "from quantbox.registry import PluginRegistry\nPluginRegistry.discover().pipelines['trade.full_pipeline.v1']"
    )
    assert TRADING in mods
    assert BACKTEST not in mods
    assert "quantbox.plugins.pipeline._report" not in mods


def test_a_plugin_map_reads_classes_through_every_dict_door():
    target = BUILTIN_PLUGINS["strategy"]["strategy.static_weights.v1"]
    cls = load(target)

    def fresh() -> PluginMap:
        return PluginMap({"strategy.static_weights.v1": target})

    m = fresh()
    assert isinstance(m, dict)
    assert "strategy.static_weights.v1" in m and len(m) == 1 and list(m) == ["strategy.static_weights.v1"]
    assert fresh()["strategy.static_weights.v1"] is cls
    assert fresh().get("strategy.static_weights.v1") is cls
    assert fresh().get("missing", 7) == 7
    assert list(fresh().values()) == [cls]
    assert list(fresh().items()) == [("strategy.static_weights.v1", cls)]
    assert {**fresh()} == {"strategy.static_weights.v1": cls}
    assert dict(fresh()) == {"strategy.static_weights.v1": cls}
    merged: dict = {}
    merged.update(fresh())
    assert merged == {"strategy.static_weights.v1": cls}
    assert fresh() == {"strategy.static_weights.v1": cls}
    assert (fresh() | {"x": int})["strategy.static_weights.v1"] is cls
    assert ({"x": int} | fresh())["strategy.static_weights.v1"] is cls
    assert fresh().copy()["strategy.static_weights.v1"] is cls
    assert fresh().pop("strategy.static_weights.v1") is cls
    assert fresh().setdefault("strategy.static_weights.v1") is cls
    assert repr(fresh()) == repr({"strategy.static_weights.v1": cls})  # the class, never the "module:Class" string


def test_an_entry_point_overrides_a_builtin_of_the_same_name():
    m = PluginMap(BUILTIN_PLUGINS["strategy"])
    m.update({"strategy.static_weights.v1": int})
    assert m["strategy.static_weights.v1"] is int


def test_discover_still_finds_every_builtin_class():
    reg = PluginRegistry.discover()
    assert reg.pipelines["backtest.pipeline.v1"].meta.name == "backtest.pipeline.v1"
    assert reg.brokers["sim.paper.v1"].meta.name == "sim.paper.v1"
    assert all(isinstance(v, type) for v in reg.strategies.values())


# ----------------------------------------------------------------------
# The engine seam and the pipeline package
# ----------------------------------------------------------------------


def test_the_engine_seam_imports_no_vectorbt_adapter_until_named():
    mods = _modules_after(
        "from quantbox.engine import get_engine, engine_names\nassert engine_names() == ['vectorbt', 'rsims']\n"
        "get_engine('rsims')"
    )
    assert "quantbox.engine.vectorbt" not in mods
    assert "quantbox.plugins.backtesting" not in mods
    assert "quantbox.engine.rsims_sim" in mods
    named = _modules_after("from quantbox.engine import get_engine\nget_engine('vectorbt', require_installed=False)")
    assert "quantbox.engine.vectorbt" in named


def test_the_pipeline_package_imports_no_pipeline_until_one_is_named():
    mods = _modules_after("import quantbox.plugins.pipeline as p\nassert 'BacktestPipeline' in p.__all__")
    assert not {TRADING, BACKTEST, "quantbox.plugins.pipeline.alloc2orders"} & mods


def test_a_pipeline_named_from_the_package_drags_not_the_other_layer():
    research = _modules_after("from quantbox.plugins.pipeline import BacktestPipeline, FundSelectionPipeline")
    assert TRADING not in research and "quantbox.plugins.pipeline.alloc2orders" not in research
    assert not {m for m in research if m.startswith(BROKERS)}
    trade = _modules_after("from quantbox.plugins.pipeline import TradingPipeline, AllocationsToOrdersPipeline")
    assert BACKTEST not in trade and "quantbox.plugins.pipeline.fund_selection" not in trade


def test_the_pipeline_package_returns_the_class_and_refuses_an_unknown_name():
    import quantbox.plugins.pipeline as pkg
    from quantbox.plugins.pipeline.backtest_pipeline import BacktestPipeline

    assert pkg.BacktestPipeline is BacktestPipeline
    with pytest.raises(AttributeError):
        _ = pkg.NoSuchPipeline


def test_star_import_of_the_package_imports_core_only():
    mods = _modules_after("from quantbox import *")
    upper = {"quantbox.bt", "quantbox.adapters", "quantbox.simulation", "quantbox.sweep", "quantbox.warehouse"}
    assert not upper & mods
    assert not {m for m in mods if m.startswith("quantbox.plugins.") and m != "quantbox.plugins.builtins"}
    import quantbox

    assert {"bt", "adapters", "plugins"}.isdisjoint(quantbox.__all__)
    for name in ("quantbox.bt", "quantbox.adapters", "quantbox.plugins"):
        importlib.import_module(name)  # still importable by name


# ----------------------------------------------------------------------
# The one lazy helper
# ----------------------------------------------------------------------


def test_load_names_the_extra_when_a_third_party_module_is_missing():
    with pytest.raises(MissingExtraError, match=r"\[research\]") as info:
        load("no_such_third_party_pkg_tom1451.sub:thing", extra="research")
    assert info.value.extra == "research"
    assert isinstance(info.value, ImportError)


def test_load_raises_a_missing_quantbox_module_as_the_bug_it_is():
    with pytest.raises(ModuleNotFoundError) as info:
        load("quantbox.no_such_module_tom1451", extra="research")
    assert not isinstance(info.value, MissingExtraError)


def test_load_without_an_extra_raises_the_import_error_unchanged():
    with pytest.raises(ModuleNotFoundError) as info:
        load("no_such_third_party_pkg_tom1451")
    assert not isinstance(info.value, MissingExtraError)


def test_load_returns_a_module_or_a_dotted_attribute():
    import quantbox.metrics as metrics

    assert load("quantbox.metrics") is metrics
    assert load("quantbox.metrics:compute_backtest_metrics") is metrics.compute_backtest_metrics
    with pytest.raises(AttributeError):
        load("quantbox.metrics:no_such_name")


# ----------------------------------------------------------------------
# Shims: every old path resolves to the new object and warns
# ----------------------------------------------------------------------

_MOVED = [
    *(
        ("quantbox.plugins.backtesting.rsims_engine", name, "quantbox.engine.rsims_sim")
        for name in ("fixed_commission_backtest_with_funding", "positions_from_no_trade_buffer")
    ),
    *(
        ("quantbox.simulation.models", name, "quantbox.montecarlo.models")
        for name in ("GBM", "GBMParams", "GARCH", "JumpDiffusion", "MeanReversion", "RegimeSwitching", "BaseModel")
    ),
    *(
        ("quantbox.simulation.engine", name, "quantbox.montecarlo.engine")
        for name in ("MarketSimulator", "SimulationConfig", "SimulationResult", "generate_correlated_returns")
    ),
    *(
        ("quantbox.simulation.correlation", name, "quantbox.montecarlo.correlation")
        for name in ("CorrelationEngine", "CorrelationResult", "generate_random_correlation_matrix")
    ),
    *(
        ("quantbox.simulation.stress_testing", name, "quantbox.montecarlo.stress_testing")
        for name in (
            "StressTestEngine",
            "StressTestResult",
            "StressScenario",
            "HistoricalScenario",
            "HISTORICAL_SCENARIOS",
        )
    ),
]


@pytest.mark.parametrize(("old", "name", "new"), _MOVED)
def test_an_old_path_resolves_to_the_new_object_with_a_deprecation_warning(old, name, new):
    module = importlib.import_module(old)
    pattern = re.escape(f"{old}.{name} is deprecated: ") + ".*" + re.escape(new) + r".*TOM-1451"
    with pytest.warns(DeprecationWarning, match=pattern):
        obj = getattr(module, name)
    assert obj is getattr(importlib.import_module(new), name)


@pytest.mark.parametrize(
    "old",
    [
        "quantbox.plugins.backtesting.rsims_engine",
        "quantbox.simulation.models",
        "quantbox.simulation.engine",
        "quantbox.simulation.correlation",
        "quantbox.simulation.stress_testing",
    ],
)
def test_an_unknown_name_on_an_old_path_is_still_an_attribute_error(old):
    with pytest.raises(AttributeError):
        _ = importlib.import_module(old).no_such_name


def test_the_research_package_re_exports_the_kernels_without_a_warning():
    import quantbox.montecarlo as mc

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        import quantbox.simulation as sim

        for name in mc.__all__:
            assert getattr(sim, name) is getattr(mc, name)
        from quantbox.plugins.backtesting import positions_from_no_trade_buffer
    from quantbox.engine.rsims_sim import positions_from_no_trade_buffer as new

    assert positions_from_no_trade_buffer is new


def test_validation_finding_is_one_class_under_every_name():
    from quantbox import config_checks, exceptions, validate

    assert validate.ValidationFinding is exceptions.ValidationFinding is config_checks.ValidationFinding
    assert validate.UNKNOWN_PLUGIN == exceptions.UNKNOWN_PLUGIN == "unknown_plugin"
