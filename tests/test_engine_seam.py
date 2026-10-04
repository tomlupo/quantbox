"""The engine seam (docs/adr/0008, TOM-1447): every backtest door, both adapters, one lag.

Same toy as ``test_execution_timing``: one asset ``A`` jumps +10% on bar ``J``.
Next-bar, a weight decided on ``J-1`` fills at close[J] — after the jump — and
earns nothing; one decided on ``J-2`` earns the jump. Every door (the single
run, variants, the sweep, ``backtest()``, ``optimize()``) on every engine must
say so. Same-bar (the lag deleted) would make the ``J-1`` book earn the jump,
so these are also the tests that go red when the seam's lag line is removed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest
from test_execution_timing import JUMP, J, _Data, _FixedWeights, _prices, _run_pipeline, _weights_decided_on

from quantbox.analysis.parameter_grid import sweep
from quantbox.engine import Costs, TradedBook, engine_names, get_engine, simulate_weights
from quantbox.execution import resolve_execution
from quantbox.plugins.backtesting import backtest, optimize
from quantbox.plugins.pipeline.backtest_pipeline import BacktestPipeline
from quantbox.store import FileArtifactStore

ENGINES = ["vectorbt", "rsims"]
# (decided on, total return next-bar): the J-1 row is the one same-bar would get wrong.
DECISIONS = [(J - 1, 0.0), (J - 2, JUMP)]


def test_both_adapters_are_registered():
    assert sorted(engine_names()) == sorted(ENGINES)


# ----------------------------------------------------------------------
# The same config, only `engine` changed, through every door
# ----------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(("decided_on", "expected"), DECISIONS)
def test_single_run(tmp_path, engine, decided_on, expected):
    result, _ = _run_pipeline(tmp_path, {"engine": engine}, decided_on=decided_on)
    assert result.notes["engine"] == engine
    assert result.metrics["total_return"] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_variants(tmp_path, engine):
    """Variants ran on vectorbt only until the seam; rsims runs them too."""
    result = BacktestPipeline().run(
        mode="backtest",
        asof="2024-02-09",
        params={
            "fees": 0.0,
            "engine": engine,
            "variants": [
                {"name": "late", "strategy": {"name": "late"}},
                {"name": "early", "strategy": {"name": "early"}},
            ],
        },
        data=_Data(),
        store=FileArtifactStore(str(tmp_path), "run"),
        broker=None,
        risk=[],
        variant_plugins={"late": _FixedWeights(J - 1), "early": _FixedWeights(J - 2)},
    )
    assert result.notes["engine"] == engine
    assert result.metrics["late__total_return"] == pytest.approx(0.0, abs=1e-9)
    assert result.metrics["early__total_return"] == pytest.approx(JUMP, abs=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(("decided_on", "expected"), DECISIONS)
def test_backtest_helper(engine, decided_on, expected):
    result = backtest(_prices(), _weights_decided_on(decided_on), fees=0.0, engine=engine)
    assert result["engine"] == engine
    assert result["metrics"]["total_return"] == pytest.approx(expected, abs=1e-9)
    assert isinstance(result["book"], TradedBook)


@pytest.mark.parametrize("engine", ENGINES)
def test_optimize_helper(engine):
    def decided(prices, params):
        return _weights_decided_on(params["decided_on"]).loc[prices.index]

    result = optimize(
        _prices(), decided, {"decided_on": [J - 1, J - 2]}, metric="total_return", fees=0.0, engine=engine
    )
    assert result["best_params"] == {"decided_on": J - 2}
    assert result["best_metric"] == pytest.approx(JUMP, abs=1e-9)
    late = result["all_results"].set_index("decided_on").loc[J - 1, "total_return"]
    assert late == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_sweep(engine):
    grid = sweep(
        strategy_cls=_FixedWeights,
        base_params={},
        sweep_params={"decided_on": [J - 1, J - 2]},
        data={"prices": _prices()},
        backtest_kwargs={"fees": 0.0, "rebalancing_freq": 1, "engine": engine},
        metrics=["total_return"],
    )
    by = grid.set_index("decided_on")["total_return"]
    assert by.loc[J - 1] == pytest.approx(0.0, abs=1e-9)
    assert by.loc[J - 2] == pytest.approx(JUMP, abs=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(("decided_on", "expected"), DECISIONS)
def test_the_seam_itself(engine, decided_on, expected):
    book = simulate_weights(
        _prices(), _weights_decided_on(decided_on), engine=engine, timing=resolve_execution(None), costs=Costs()
    )
    assert book.engine == engine
    assert book.value.iloc[-1] / book.value.iloc[0] - 1 == pytest.approx(expected, abs=1e-9)
    assert book.execution["lag_bars"] == 1


# ----------------------------------------------------------------------
# What a traded book carries
# ----------------------------------------------------------------------


def test_native_is_the_vectorbt_portfolio_on_vectorbt():
    vbt = pytest.importorskip("vectorbt")
    result = backtest(_prices(), _weights_decided_on(J - 2), fees=0.0, engine="vectorbt")
    assert isinstance(result["book"].native, vbt.Portfolio)
    assert result["native"] is result["book"].native
    assert result["vbt_portfolio"] is result["native"]  # the pre-seam key still answers


def test_native_is_the_rsims_results_frame_on_rsims():
    result = backtest(_prices(), _weights_decided_on(J - 2), fees=0.0, engine="rsims")
    assert isinstance(result["native"], pd.DataFrame)
    assert {"ticker", "Trades", "Value"} <= set(result["native"].columns)
    assert "vbt_portfolio" not in result


@pytest.mark.parametrize("engine", ENGINES)
def test_a_book_reports_turnover_and_trades(engine):
    book = simulate_weights(_prices(), _weights_decided_on(J - 2), engine=engine, timing=resolve_execution(None))
    # One entry trade, on the bar the J-2 decision fills (next-bar: J-1).
    assert book.turnover.sum() == pytest.approx(1.0)
    assert book.turnover.idxmax() == _prices().index[J - 1]
    trades = book.trades
    assert list(trades.columns) == ["date", "symbol", "size", "price", "value", "fees"]
    # The entry is a buy on that bar. (rsims re-sizes every bar off its initial cash, so it also
    # trims after the jump; vectorbt holds between rebalances. Each engine's own policy.)
    bought = trades[trades["symbol"] == "A"].sort_values("date")
    assert bought["size"].iloc[0] > 0
    assert pd.Timestamp(bought["date"].iloc[0]) == _prices().index[J - 1]


def test_an_adapter_refuses_a_parameter_it_does_not_own():
    with pytest.raises(ValueError, match="does not take"):
        backtest(_prices(), _weights_decided_on(J - 2), engine="rsims", engine_params={"use_numba": True})
    with pytest.raises(ValueError, match="does not take"):
        backtest(_prices(), _weights_decided_on(J - 2), engine="vectorbt", engine_params={"trade_buffer": 0.1})


def test_an_unknown_engine_is_refused_everywhere(tmp_path):
    with pytest.raises(ValueError, match="Unknown engine"):
        get_engine("zipline")
    with pytest.raises(ValueError, match="Unknown engine"):
        backtest(_prices(), _weights_decided_on(J - 2), engine="zipline")
    with pytest.raises(ValueError, match="Unknown engine"):
        BacktestPipeline().plan({"engine": "zipline"})


# ----------------------------------------------------------------------
# No branch on the engine name outside the adapters
# ----------------------------------------------------------------------

SRC = Path(__file__).resolve().parents[1] / "src" / "quantbox"
#: Where an engine name may be compared: the adapters, their registry, and the primitives they wrap.
ADAPTER_FILES = {
    SRC / "engine" / "vectorbt.py",
    SRC / "engine" / "rsims.py",
    SRC / "engine" / "registry.py",
    SRC / "adapters" / "vectorbt.py",  # the L0 re-export: it names its own module on an ImportError
    SRC / "plugins" / "backtesting" / "vectorbt_engine.py",
    SRC / "plugins" / "backtesting" / "rsims_engine.py",
}


def _engine_name_branches(path: Path) -> list[str]:
    """Comparisons (``==``, ``!=``, ``in``, ``not in``) and ``match`` cases against an engine name literal."""
    names = set(ENGINES)
    label = path.relative_to(SRC.parent) if path.is_relative_to(SRC.parent) else path.name
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            literals = set()
            for op in operands:
                if isinstance(op, ast.Constant):
                    literals.add(op.value)
                elif isinstance(op, (ast.Tuple, ast.List, ast.Set)):
                    literals |= {e.value for e in op.elts if isinstance(e, ast.Constant)}
            if literals & names:
                found.append(f"{label}:{node.lineno}")
        elif isinstance(node, ast.MatchValue) and isinstance(node.value, ast.Constant) and node.value.value in names:
            found.append(f"{label}:{node.lineno}")
    return found


def test_no_code_outside_the_adapters_branches_on_the_engine_name():
    scanned = [p for p in SRC.rglob("*.py") if p not in ADAPTER_FILES]
    assert len(scanned) > 100, "the scan saw almost no files — it is blind"
    branches = [hit for p in scanned for hit in _engine_name_branches(p)]
    assert branches == []


def test_the_branch_scan_sees_a_branch(tmp_path):
    """The control: the scan finds the shape it forbids."""
    probe = tmp_path / "probe.py"
    probe.write_text('def f(engine):\n    return engine in ("vectorbt", "x")\n', encoding="utf-8")
    assert _engine_name_branches(probe) == ["probe.py:2"]
