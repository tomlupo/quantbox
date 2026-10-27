"""TOM-2458: guards on the compare plan (.hotl/compare.yml) and scripts/before_after.py's map.

The before/after decides what a PR runs. Its two failure modes are silent: a path
mapped to no step when it should run, and a config skipped although its code
changed. Each test below fails on one of them. The file lists are real PR diffs
(#264 docs, #313 inference, #310 rebalancing); the runs themselves are proved in
the PR, not here.
"""

from __future__ import annotations

import importlib.util
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


def test_the_backtest_path_reaches_inference() -> None:
    """The backtest pipeline imports inference (blocks.py), so an inference change runs the backtests."""
    assert "quantbox.inference" in _closure("cookbook/configs/run_synthetic_backtest.yaml")
