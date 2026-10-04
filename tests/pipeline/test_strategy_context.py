"""Backtest and trading hand a strategy the SAME StrategyContext (TOM-1448).

TOM-1338 made both pipelines inject the same `_pipeline_annualize`; this removes
the cause: ONE strategy runner (`quantbox.strategy_runner.run_strategies`) builds
a `StrategyContext` (bars_per_year, mode, as-of, calendar, frequency) and hands
it to every strategy, from both pipelines. Both pipelines are run END TO END
with one recording strategy, so the test fails on the parity, not on an
implementation detail of either pipeline.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantbox.contracts import StrategyContext
from quantbox.plugins.broker.sim import SimPaperBroker
from quantbox.plugins.pipeline.backtest_pipeline import BacktestPipeline
from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline
from quantbox.store import FileArtifactStore
from quantbox.strategy_runner import resolve_annualize

ASOF = "2024-03-30"
N = 90


def _prices() -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.date_range(end=ASOF, periods=N, freq="D").values)
    rng = np.random.RandomState(0)
    a = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, N)))
    return pd.DataFrame({"A": a, "USD": 1.0}, index=idx)


class _Data:
    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame({"symbol": ["A", "USD"]})

    def load_market_data(self, universe: Any, asof: str, params: dict[str, Any]) -> dict[str, pd.DataFrame]:
        return {"prices": _prices()}


def _weights(data: Any) -> dict[str, Any]:
    prices = data["prices"]
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w["A"] = 0.5
    return {"weights": w}


class _ContextAware:
    """Records the params AND the context it was handed."""

    meta = type("M", (), {"name": "strategy.recording.v1"})()

    def __init__(self) -> None:
        self.seen: list[tuple[dict[str, Any], StrategyContext | None]] = []

    def run(self, data: Any, params: Any = None, context: StrategyContext | None = None) -> dict[str, Any]:
        self.seen.append((dict(params or {}), context))
        return _weights(data)


class _Legacy:
    """A third-party strategy written before StrategyContext: run(data, params) only."""

    meta = type("M", (), {"name": "strategy.legacy.v1"})()

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def run(self, data: Any, params: Any = None) -> dict[str, Any]:
        self.seen.append(dict(params or {}))
        return _weights(data)


def _run_backtest(tmp_path, strat, pipeline_params, strat_params=None, mode="backtest"):
    params = {
        "fees": 0.0,
        "strategies": [{"name": strat.meta.name, "weight": 1.0, "params": strat_params or {}}],
        **pipeline_params,
    }
    BacktestPipeline().run(
        mode=mode,
        asof=ASOF,
        params=params,
        data=_Data(),
        store=FileArtifactStore(str(tmp_path / "bt"), "run"),
        broker=None,
        risk=[],
        strategies=[strat],
    )
    assert strat.seen, "backtest never ran the strategy — the comparison would be vacuous"
    return strat.seen[-1]


def _run_trading(tmp_path, strat, pipeline_params, strat_params=None, mode="backtest"):
    params = {
        "strategies": [{"name": strat.meta.name, "weight": 1.0, "params": strat_params or {}}],
        **pipeline_params,
    }
    TradingPipeline().run(
        mode=mode,
        asof=ASOF,
        params=params,
        data=_Data(),
        store=FileArtifactStore(str(tmp_path / "tr"), "run"),
        broker=None if mode == "backtest" else SimPaperBroker(),
        risk=[],
        strategies=[strat],
    )
    assert strat.seen, "trading never ran the strategy — the comparison would be vacuous"
    return strat.seen[-1]


FREQUENCY_CONFIGS = {
    "default_1d_24_7": ({}, 365.0, "1d", "24/7"),
    "nyse_calendar_shorthand": ({"market_calendar": "NYSE"}, None, "1d", "NYSE"),
    "explicit_4h": ({"frequency": "4h"}, 2190.0, "4h", "24/7"),
    "prices_frequency_shorthand": ({"prices": {"frequency": "1h", "lookback_days": 90}}, 8760.0, "1h", "24/7"),
}


@pytest.mark.parametrize(
    ("pipeline_params", "bars", "bar_size", "calendar"), FREQUENCY_CONFIGS.values(), ids=FREQUENCY_CONFIGS.keys()
)
def test_backtest_and_trading_hand_an_identical_context(tmp_path, pipeline_params, bars, bar_size, calendar):
    _, bt_ctx = _run_backtest(tmp_path, _ContextAware(), pipeline_params)
    _, tr_ctx = _run_trading(tmp_path, _ContextAware(), pipeline_params)
    assert isinstance(bt_ctx, StrategyContext), f"backtest handed {bt_ctx!r}, not a StrategyContext"
    assert tr_ctx == bt_ctx, f"trading handed {tr_ctx!r}, backtest handed {bt_ctx!r}"
    assert bt_ctx.mode == "backtest"
    assert bt_ctx.asof == ASOF
    assert bt_ctx.frequency == bar_size
    assert bt_ctx.calendar == calendar
    if bars is not None:
        assert bt_ctx.bars_per_year == pytest.approx(bars)
    else:  # an exchange calendar: ~250 sessions a year
        assert 245 <= bt_ctx.bars_per_year <= 255


def test_paper_differs_from_backtest_only_in_mode(tmp_path):
    """Live is the point of the parity: everything but `mode` must match the backtest."""
    _, bt_ctx = _run_backtest(tmp_path, _ContextAware(), {})
    _, tr_ctx = _run_trading(tmp_path, _ContextAware(), {}, mode="paper")
    assert tr_ctx.mode == "paper"
    assert tr_ctx == dataclasses.replace(bt_ctx, mode="paper")


def test_context_aware_strategy_gets_no_legacy_annualize_param(tmp_path):
    """The context REPLACES the `_pipeline_annualize` injection; it is not added beside it."""
    bt_params, _ = _run_backtest(tmp_path, _ContextAware(), {})
    tr_params, _ = _run_trading(tmp_path, _ContextAware(), {})
    assert "_pipeline_annualize" not in bt_params
    assert "_pipeline_annualize" not in tr_params


@pytest.mark.parametrize("runner", [_run_backtest, _run_trading], ids=["backtest", "trading"])
def test_legacy_strategy_still_gets_annualize_with_a_deprecation_warning(tmp_path, runner):
    """A strategy without a `context` parameter keeps working one more minor version — loudly."""
    strat = _Legacy()
    with pytest.warns(DeprecationWarning, match="_pipeline_annualize"):
        params = runner(tmp_path, strat, {})
    assert params["_pipeline_annualize"] == 365.0


@pytest.mark.parametrize("runner", [_run_backtest, _run_trading], ids=["backtest", "trading"])
def test_explicit_legacy_value_is_not_overwritten(tmp_path, runner):
    """An explicit `_pipeline_annualize` in a strategy's params still wins, in both pipelines."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        params = runner(tmp_path, _Legacy(), {}, {"_pipeline_annualize": 252.0})
    assert params["_pipeline_annualize"] == 252.0


# ---------------------------------------------------------------------------
# resolve_annualize — the one annualisation block every strategy calls
# ---------------------------------------------------------------------------

_CTX = StrategyContext(bars_per_year=365.0, mode="backtest", asof=ASOF, calendar="24/7", frequency="1d")


def test_resolve_reads_the_context():
    assert resolve_annualize(None, {}, _CTX, owner="S") == 365.0


def test_resolve_explicit_field_wins_over_context():
    assert resolve_annualize(252.0, {}, _CTX, owner="S") == 252.0


def test_resolve_explicit_legacy_param_wins_over_context():
    with pytest.warns(DeprecationWarning, match="_pipeline_annualize"):
        assert resolve_annualize(None, {"_pipeline_annualize": 252.0}, _CTX, owner="S") == 252.0


def test_resolve_without_context_falls_back_to_252():
    """An L3 call with no context and no value: the historical equity default."""
    assert resolve_annualize(None, {}, None, owner="S") == 252.0


def test_every_builtin_strategy_accepts_a_context():
    """No built-in strategy is left on the deprecated injection path."""
    from quantbox.plugins import strategies as pkg

    missing = []
    for name in pkg.__all__:
        obj = getattr(pkg, name)
        is_strategy = isinstance(obj, type) and hasattr(obj, "meta") and callable(getattr(obj, "run", None))
        if is_strategy and "context" not in inspect.signature(obj.run).parameters:
            missing.append(name)
    assert not missing, f"strategies whose run() takes no `context`: {missing}"
