"""Backtest and trading must hand a strategy the SAME annualisation (TOM-1338).

`backtest.pipeline.v1` derives bars-per-year from the run's frequency and
injects it into every strategy's params as `_pipeline_annualize`. Seven
strategies read that key and fall back to 252 when it is absent. If the trading
pipeline does not inject it, the same strategy on a 24/7 crypto book sizes by
sqrt(365/252) ~= 1.20x differently in paper/live than in its backtest.

Both pipelines are run END TO END here with one recording strategy, and the
value each one handed it is compared — so the test fails on the parity, not on
an implementation detail of either pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.pipeline.backtest_pipeline import BacktestPipeline
from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline
from quantbox.store import FileArtifactStore

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


class _Recording:
    """Strategy stub that records the params it was handed."""

    meta = type("M", (), {"name": "strategy.recording.v1"})()

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def run(self, data: Any, params: Any = None) -> dict[str, Any]:
        self.seen.append(dict(params or {}))
        prices = data["prices"]
        w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        w["A"] = 0.5
        return {"weights": w}


def _annualize_from_backtest(tmp_path, pipeline_params: dict[str, Any], strat_params: dict[str, Any]) -> Any:
    strat = _Recording()
    params = {
        "fees": 0.0,
        "strategies": [{"name": strat.meta.name, "weight": 1.0, "params": strat_params}],
        **pipeline_params,
    }
    BacktestPipeline().run(
        mode="backtest",
        asof=ASOF,
        params=params,
        data=_Data(),
        store=FileArtifactStore(str(tmp_path / "bt"), "run"),
        broker=None,
        risk=[],
        strategies=[strat],
    )
    assert strat.seen, "backtest never ran the strategy — the comparison would be vacuous"
    return strat.seen[-1].get("_pipeline_annualize")


def _annualize_from_trading(tmp_path, pipeline_params: dict[str, Any], strat_params: dict[str, Any]) -> Any:
    strat = _Recording()
    params = {
        "strategies": [{"name": strat.meta.name, "weight": 1.0, "params": strat_params}],
        **pipeline_params,
    }
    TradingPipeline().run(
        mode="backtest",
        asof=ASOF,
        params=params,
        data=_Data(),
        store=FileArtifactStore(str(tmp_path / "tr"), "run"),
        broker=None,
        risk=[],
        strategies=[strat],
    )
    assert strat.seen, "trading never ran the strategy — the comparison would be vacuous"
    return strat.seen[-1].get("_pipeline_annualize")


FREQUENCY_CONFIGS = {
    "default_1d_24_7": {},
    "nyse_calendar_shorthand": {"market_calendar": "NYSE"},
    "explicit_4h": {"frequency": "4h"},
    "prices_frequency_shorthand": {"prices": {"frequency": "1h", "lookback_days": 90}},
}


@pytest.mark.parametrize("pipeline_params", FREQUENCY_CONFIGS.values(), ids=FREQUENCY_CONFIGS.keys())
def test_trading_hands_strategy_the_same_annualisation_as_backtest(tmp_path, pipeline_params):
    bt = _annualize_from_backtest(tmp_path, pipeline_params, {})
    tr = _annualize_from_trading(tmp_path, pipeline_params, {})
    assert bt is not None, "backtest stopped injecting _pipeline_annualize — the baseline moved"
    assert tr == bt, f"trading handed {tr!r}, backtest handed {bt!r}"


def test_default_crypto_book_is_365_in_both(tmp_path):
    """The live books run 1d bars on a 24/7 calendar: both sides must say 365, not 252."""
    assert _annualize_from_trading(tmp_path, {}, {}) == 365.0
    assert _annualize_from_backtest(tmp_path, {}, {}) == 365.0


def test_explicit_strategy_value_is_not_overwritten_by_trading(tmp_path):
    """An explicit `_pipeline_annualize` in a strategy's params wins in trading, as in backtest."""
    assert _annualize_from_trading(tmp_path, {}, {"_pipeline_annualize": 252.0}) == 252.0
    assert _annualize_from_backtest(tmp_path, {}, {"_pipeline_annualize": 252.0}) == 252.0
