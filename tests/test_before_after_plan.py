"""TOM-2458: guards on the compare plan (.hotl/compare.yml) and scripts/before_after.py's map.

The before/after decides what a PR runs. Its two failure modes are silent: a path
mapped to no step when it should run, and a config skipped although its code
changed. Each test below fails on one of them. The file lists are real PR diffs
(#264 docs, #313 inference, #310 rebalancing); the runs themselves are proved in
the PR, not here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("before_after", ROOT / "scripts" / "before_after.py")
ba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ba)

PLAN = ba.load_plan(ROOT)
GRAPH = ba.build_graph(ROOT, "src/quantbox", set(PLAN["import_graph"]["catalogues"]))
LAYERS = ba.read_layers(ROOT / "pyproject.toml")


def steps_of(files: list[str]) -> dict[str, str | None]:
    return {m.path: m.step for m in ba.map_files(files, PLAN, [GRAPH], LAYERS)}


def test_the_default_block_is_read_and_merged() -> None:
    """compare.yml carries the quant-lib default once, in a comment; the script must parse it."""
    assert PLAN["steps"] == ["data", "signal", "decision", "engine", "trade"]
    assert "docs/**" in PLAN["ignore"]
    assert "**/strategies/**" in PLAN["paths"]["signal"]  # the default's glob
    assert "src/quantbox/inference.py" in PLAN["paths"]["engine"]  # quantbox's addition
    assert all((ROOT / c).is_file() for c in PLAN["configs"]), "compare.yml lists a config that does not exist"


def test_docs_and_tests_map_to_no_step() -> None:  # #264
    files = ["CLAUDE.md", "docs/adr/0010-audience-layers-one-distribution.md", "tests/test_docs_module_refs.py"]
    assert set(steps_of(files).values()) == {None}


def test_inference_diff_is_engine_only() -> None:  # #313
    files = ["src/quantbox/gates.py", "src/quantbox/gates_cli.py", "src/quantbox/inference.py"]
    assert set(steps_of(files).values()) == {"engine"}


def test_rebalancing_diff_is_trade() -> None:  # #310
    files = [
        "src/quantbox/plugins/broker/binance_futures.py",
        "src/quantbox/plugins/rebalancing/futures_rebalancer.py",
        "src/quantbox/trade/executor.py",
        "src/quantbox/trade/reconciliation/cycle.py",
    ]
    assert set(steps_of(files).values()) == {"trade"}


def test_an_unknown_path_runs_everything() -> None:
    assert steps_of(["misc/unknown-path.txt"]) == {"misc/unknown-path.txt": ba.ALL}


def test_an_unglobbed_module_takes_its_importers_step() -> None:
    """frequency.py matches no glob; the import graph must still give it a step, never none."""
    step = steps_of(["src/quantbox/frequency.py"])["src/quantbox/frequency.py"]
    assert step in PLAN["steps"] or step == ba.ALL


def _closure(config: str) -> set[str]:
    cfg = yaml.safe_load((ROOT / config).read_text())
    from quantbox.plugins.builtins import BUILTIN_PLUGINS

    roots = ba.config_roots(cfg, BUILTIN_PLUGINS, {"quantbox.runner", "quantbox.registry"}, GRAPH)
    assert roots is not None
    return GRAPH.closure(roots)


def test_a_config_reaches_its_own_strategy() -> None:
    assert "quantbox.plugins.strategies.cross_asset_momentum" in _closure("cookbook/canonical/configs/momentum.yaml")


def test_a_sibling_strategy_is_LOADED_so_it_is_in_the_closure() -> None:
    """plugins/strategies/__init__ imports every strategy, so their import-time code runs at every run
    that loads one (PR #320 review: findings 2 and 102, then round 2 finding 1)."""
    closure = _closure("cookbook/configs/run_backtest_beglobal.yaml")
    assert "quantbox.plugins.strategies.beglobal_strategy" in closure
    assert "quantbox.plugins.strategies.cross_asset_momentum" in closure
    assert "quantbox.plugins.strategies.carry" in closure


def test_a_symlink_in_a_dataset_is_refused(tmp_path, monkeypatch) -> None:  # PR #320 review round 2, 2
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside the dataset")
    root = tmp_path / "clone" / "datasets"
    (root / "etf-daily").mkdir(parents=True)
    (root / "etf-daily" / "prices.parquet").write_bytes(b"pinned")
    (root / "etf-daily" / "extra").symlink_to(outside)
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "datasets.lock").write_text(f"etf-daily: {ba._sha256_bytes(b'pinned')}\n")
    monkeypatch.setenv("QUANTBOX_DATASETS_ROOT", str(root))
    try:
        ba.pinned_slice(tree, tmp_path / "slice")
    except ba.CouldNotRun as exc:
        assert "symlink" in str(exc)
    else:
        raise AssertionError("a dataset holding a symlink was copied")
    assert not (tmp_path / "slice" / "etf-daily").exists()


def test_a_name_from_a_lock_or_a_run_cannot_leave_its_directory() -> None:  # PR #320 review, 103
    assert ba.single_name("etf-daily", "dataset") == "etf-daily"
    for bad in ("../private", "/etc", "a/b", "..", "."):
        try:
            ba.single_name(bad, "dataset")
        except ba.CouldNotRun:
            continue
        raise AssertionError(f"{bad!r} was accepted")


def test_a_same_day_data_timestamp_still_differs() -> None:  # PR #320 review, 3
    a = {"asof": "2026-10-26T00:00:00Z"}
    b = {"asof": "2026-10-26T12:00:00Z"}
    assert ba.diff_json(a, b) == ['asof: base "2026-10-26T00:00:00Z" -> head "2026-10-26T12:00:00Z"']


def test_trade_code_loads_the_decision_and_inference_but_not_gates() -> None:
    """A touched module in the trade layer's closure owes the shadow run.

    Trade code LOADS inference: risk.stress_test_risk imports montecarlo.stress_testing, whose
    package __init__ imports montecarlo.engine, which imports inference. Importing every trade
    module in a fresh process confirms it (quantbox.inference in sys.modules, quantbox.gates not).
    """
    trade = {m for m in GRAPH.edges if "trade" in ba.layers_of(m, LAYERS)}
    closure = GRAPH.closure(trade)
    assert "quantbox.decision" in closure
    assert "quantbox.inference" in closure
    assert "quantbox.gates" not in closure
    assert "quantbox.plugins.strategies.cross_asset_momentum" not in closure


LAZY_PKG = """
_PKG = "quantbox.lazy"
_LAZY = {"Fetcher": f"{_PKG}.fetch:Fetcher", "Other": (f"{_PKG}.other:Other", "data")}
def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(name)
    return load(_LAZY[name])
"""


def _lazy_tree(tmp_path: Path, importer: str) -> ba.Graph:
    """A tree with plugins.datasources' shape: a package whose __getattr__ loads the module its map names."""
    pkg = tmp_path / "src" / "quantbox"
    (pkg / "lazy").mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "lazy" / "__init__.py").write_text(LAZY_PKG)
    (pkg / "lazy" / "fetch.py").write_text("class Fetcher: ...\n")
    (pkg / "lazy" / "other.py").write_text("class Other: ...\n")
    (pkg / "user.py").write_text(importer)
    return ba.build_graph(tmp_path, "src/quantbox", set())


def test_a_name_taken_from_a_lazy_export_package_loads_its_target(tmp_path) -> None:  # PR #320 round 4, codex P2
    """`from pkg import Name` runs pkg.__getattr__, which loads the module pkg's _LAZY names for it."""
    graph = _lazy_tree(tmp_path, "from quantbox.lazy import Fetcher\n")
    assert "quantbox.lazy.fetch" in graph.closure({"quantbox.user"})
    assert "quantbox.lazy.other" not in graph.closure({"quantbox.user"})  # the map is read: one name, one module
    assert "quantbox.lazy.fetch" not in graph.closure({"quantbox.lazy"})  # loading the package loads no export


def test_a_lazy_map_not_read_in_full_widens_to_every_module_below(tmp_path) -> None:
    graph = _lazy_tree(tmp_path, "from quantbox.lazy import Fetcher\n")
    (tmp_path / "src" / "quantbox" / "lazy" / "__init__.py").write_text(
        LAZY_PKG.replace('"Other": (', '"Other": pick(')  # a value this reader cannot evaluate
    )
    graph = ba.build_graph(tmp_path, "src/quantbox", set())
    assert {"quantbox.lazy.fetch", "quantbox.lazy.other"} <= graph.closure({"quantbox.user"})


def test_an_import_of_the_lazy_package_itself_may_take_any_name(tmp_path) -> None:
    for importer in (
        "import quantbox.lazy\n",
        "from quantbox import lazy\n",
        "from quantbox.lazy import *\n",
        "import quantbox.lazy.fetch\n",  # binds `quantbox`: quantbox.lazy.Other is reachable too
    ):
        closure = _lazy_tree(tmp_path, importer).closure({"quantbox.user"})
        assert {"quantbox.lazy.fetch", "quantbox.lazy.other"} <= closure, importer


def test_a_lazy_map_bound_in_more_than_one_place_is_not_trusted(tmp_path) -> None:  # round 4 fix-delta review
    """Each shape below adds or moves a name the dict literal does not show; the closure must still hold it."""
    base = '_PKG = "quantbox.lazy"\n_LAZY = {"Fetcher": f"{_PKG}.fetch:Fetcher"}\n'
    getattr_def = "def __getattr__(name):\n    return load({maps}[name])\n"
    shapes = {
        "item set": base + '_LAZY["Other"] = f"{_PKG}.other:Other"\n' + getattr_def.format(maps="_LAZY"),
        "update": base + '_LAZY.update({"Other": f"{_PKG}.other:Other"})\n' + getattr_def.format(maps="_LAZY"),
        "second map": base
        + '_MORE = {k: f"{_PKG}.other:Other" for k in ["Other"]}\n'
        + getattr_def.format(maps="{**_LAZY, **_MORE}"),
        "conditional": base + "if True:\n    " + getattr_def.format(maps="_LAZY").replace("\n    ", "\n        "),
    }
    for shape, source in shapes.items():
        graph = _lazy_tree(tmp_path, "from quantbox.lazy import Other\n")
        (tmp_path / "src" / "quantbox" / "lazy" / "__init__.py").write_text(source)
        graph = ba.build_graph(tmp_path, "src/quantbox", set())
        assert "quantbox.lazy.other" in graph.closure({"quantbox.user"}), shape
    # a constant rebound after the map: the map's targets cannot be read
    source = base.replace('"quantbox.lazy"', '"quantbox.lazy.sub"') + '_PKG = "quantbox.lazy"\n'
    assert ba.lazy_exports(ba.ast.parse(source + getattr_def.format(maps="_LAZY")), lambda m: m or None) == (
        {},
        False,
    )


def test_every_lazy_export_package_is_read_and_an_unread_one_widens() -> None:
    for pkg, export, module in (
        ("quantbox.plugins.datasources", "BinanceDataFetcher", "quantbox.plugins.datasources.binance_data"),
        ("quantbox.plugins.pipeline", "BacktestPipeline", "quantbox.plugins.pipeline.backtest_pipeline"),
        ("quantbox.plugins.broker", "BinanceFuturesBroker", "quantbox.plugins.broker.binance_futures"),
        ("quantbox.plugins.publisher", "TelegramPublisher", "quantbox.plugins.publisher.telegram"),
    ):
        found = ba.lazy_exports(
            ba.ast.parse((ROOT / "src" / Path(*pkg.split("."))).joinpath("__init__.py").read_text()), lambda m: m
        )
        assert found is not None and found[1], pkg
        assert found[0][export] == {module}, pkg
    # only raises (plugins.backtesting) or refuses (removed_names): loads nothing, not lazy
    assert ba.lazy_exports(ba.ast.parse("def __getattr__(n):\n    raise AttributeError(f'{n}')\n"), lambda m: m) is None
    assert ba.lazy_exports(ba.ast.parse("__getattr__ = removed_names(__name__)\n"), lambda m: m) is None
    # a map this reader cannot read in full is not trusted: the caller widens to every module below
    src = "_M = {'A': make()}\ndef __getattr__(n):\n    return load(_M[n])\n"
    assert ba.lazy_exports(ba.ast.parse(src), lambda m: m or None) == ({}, False)


def _no_datasets_root(monkeypatch) -> None:
    """This runner as public CI sees it: no $QUANTBOX_DATASETS_ROOT and no quantbox-datasets clone."""
    monkeypatch.delenv("QUANTBOX_DATASETS_ROOT", raising=False)
    monkeypatch.setitem(sys.modules, "quantbox_datasets", None)  # import then raises ImportError
    monkeypatch.setitem(sys.modules, "quantbox_datasets.lock", None)


def test_a_config_whose_dataset_does_not_resolve_is_not_available(monkeypatch) -> None:  # PR #320 CI run 40550761629
    _no_datasets_root(monkeypatch)
    assert ba.unavailable_datasets("cookbook/configs/run_backtest_beglobal.yaml", (ROOT,)) == ["etf-daily"]
    assert ba.unavailable_datasets("cookbook/configs/run_stress_test.yaml", (ROOT,)) == ["crypto-spot-daily"]
    assert ba.unavailable_datasets("cookbook/canonical/configs/momentum.yaml", (ROOT,)) == []  # bundled data


def test_a_datasets_root_or_a_line_copy_makes_it_available(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QUANTBOX_DATASETS_ROOT", str(tmp_path))
    assert ba.unavailable_datasets("cookbook/configs/run_backtest_beglobal.yaml", (ROOT,)) == []
    _no_datasets_root(monkeypatch)
    line = tmp_path / "line"
    (line / "datasets" / "frozen").mkdir(parents=True)
    (line / "datasets.lock").write_text("frozen: abc\n")
    (line / "config.yaml").write_text("plugins:\n  data:\n    params_init:\n      dataset: frozen\n")
    assert ba.unavailable_datasets("config.yaml", (line,)) == []
    (line / "config.yaml").write_text("plugins:\n  data:\n    params_init:\n      dataset: absent\n")
    assert ba.unavailable_datasets("config.yaml", (line,)) == ["absent"]


def test_exit_codes_for_configs_not_available_on_this_runner() -> None:
    missing = {"cookbook/configs/run_stress_test.yaml": ["crypto-spot-daily"]}
    assert ba.verdict_of(8, 4, missing, [], [], False)[1] == 3  # the rest identical: PARTIAL, never 0
    assert ba.verdict_of(4, 0, missing, [], [], False)[1] == 2  # nothing could run: nothing was looked at
    assert ba.verdict_of(4, 0, missing, [], [], True)[1] == 2
    assert ba.verdict_of(8, 4, missing, ["a.yaml"], [], False)[1] == 2  # failed for another reason
    assert ba.verdict_of(8, 4, missing, [], ["a.yaml"], False)[1] == 1  # a difference stays red
    assert ba.verdict_of(8, 4, missing, [], [], True)[1] == 1  # so does an owed shadow run
    assert ba.verdict_of(8, 8, {}, [], [], False)[1] == 0
    assert ba.verdict_of(0, 0, {}, [], [], False)[1] == 0  # no config reaches a touched module


def test_the_backtest_path_reaches_inference() -> None:
    """The backtest pipeline imports inference (blocks.py), so an inference change runs the backtests."""
    assert "quantbox.inference" in _closure("cookbook/configs/run_synthetic_backtest.yaml")
