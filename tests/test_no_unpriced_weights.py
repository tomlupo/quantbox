"""No weight on a ticker without a price column (TOM-1500).

``CrossAssetMomentumStrategy`` with its unset ``risk_off_ticker`` (None) used to
emit a ``None`` column carrying the unallocated weight. The calendar schedule
dropped it with a ``WEIGHTS:`` warning; ``schedule: bars`` refused it. Now:

- no built-in strategy emits a ``None`` or unpriced column on a plain panel;
- the calendar schedule refuses a weight on a ticker with no price column, as
  ``schedule: bars`` already did.
"""

from __future__ import annotations

import logging
from dataclasses import is_dataclass

import numpy as np
import pandas as pd
import pytest

from quantbox.engine import Costs, simulate
from quantbox.execution import resolve_execution
from quantbox.plugins import strategies as strategies_pkg
from quantbox.plugins.strategies.cross_asset_momentum import CrossAssetMomentumStrategy, apply_core_satellite

TICKERS = ["BTC", "ETH", "SOL", "ADA", "XRP", "DOGE"]


def _panel() -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2023-01-01", periods=400, freq="D")
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.03, (len(idx), len(TICKERS))), axis=0), idx, TICKERS
    )
    return {
        "prices": prices,
        "volume": prices * 1e6,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "market_cap": prices * 1e9,
    }


#: Strategies a plain price panel cannot run, and what each needs. Declared, so a
#: NEW strategy that cannot run here fails instead of silently escaping the check.
NEEDS_INPUT = {
    "strategy.dual_momentum.v1": "asset_a / asset_b params",
    "strategy.trend_following.v1": "a level-ticker MultiIndex panel",
    "strategy.weighted_avg.v1": "an aggregator: the other strategies' weights",
    "strategy.beglobal.v1": "ETF or asset-class columns",
    "strategy.eth_mean_reversion_24h.v1": "an ETH intraday panel",
    "strategy.frozen_weights.v1": "a weights_path",
    "strategy.hmm_regime_allocation.v1": "asset-class columns",
}


def _strategy_classes() -> list[type]:
    out = []
    for name in strategies_pkg.__all__:
        obj = getattr(strategies_pkg, name)
        if isinstance(obj, type) and hasattr(obj, "meta") and is_dataclass(obj):
            out.append(obj)
    return out


@pytest.mark.parametrize("cls", _strategy_classes(), ids=lambda c: c.meta.name)
def test_no_builtin_strategy_emits_a_none_or_unpriced_column(cls):
    if cls.meta.name in NEEDS_INPUT:
        pytest.skip(f"NOT CHECKED: {cls.meta.name} needs {NEEDS_INPUT[cls.meta.name]}")
    weights = cls().run(_panel(), {})["weights"]
    tickers = weights.columns.get_level_values(-1) if weights.columns.nlevels > 1 else weights.columns
    unpriced = [c for c in tickers if c not in TICKERS]
    assert unpriced == [], f"{cls.meta.name} puts weight on {unpriced}, which have no price column"


def test_the_check_covers_most_strategies():
    """The skip list stays a minority: the check above is not vacuous."""
    names = {c.meta.name for c in _strategy_classes()}
    assert set(NEEDS_INPUT) <= names, sorted(set(NEEDS_INPUT) - names)
    assert len(names - set(NEEDS_INPUT)) >= 10


def test_momentum_without_a_risk_off_ticker_leaves_the_unallocated_weight_in_cash():
    prices = _panel()["prices"]
    out = CrossAssetMomentumStrategy(ewma_min_periods=20, output_periods=400).run({"prices": prices})
    weights = out["weights"]
    assert list(weights.columns) == TICKERS
    # The core-satellite blend leaves part of the book unallocated on some bars: that part is cash.
    assert (weights.sum(axis=1) < 1.0 - 1e-9).any()
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()
    assert None not in out["simple_weights"]


def test_core_satellite_with_a_risk_off_ticker_still_routes_the_unallocated_weight():
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    active = pd.DataFrame({"A": [0.5, 0.0], "B": [0.0, 0.0]}, index=idx)
    passive = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]}, index=idx)
    routed = apply_core_satellite(active, passive, core_weight=0.6, risk_off_ticker="B")
    assert routed["B"].tolist() == pytest.approx([0.3 + 0.2, 0.3 + 0.4])
    cash = apply_core_satellite(active, passive, core_weight=0.6, risk_off_ticker=None)
    assert list(cash.columns) == ["A", "B"]
    assert cash.sum(axis=1).tolist() == pytest.approx([0.8, 0.6])


@pytest.mark.parametrize("schedule", ["calendar", "bars"])
def test_a_weight_on_a_ticker_with_no_price_column_is_refused(schedule):
    prices = _panel()["prices"][["BTC", "ETH"]]
    weights = pd.DataFrame({"BTC": 0.5, "ETH": 0.3, "GHOST": 0.2}, index=prices.index)
    with pytest.raises(ValueError, match=r"GHOST"):
        simulate(prices, weights, engine="rsims", timing=resolve_execution({"schedule": schedule}), costs=Costs())


def test_a_none_column_is_refused_on_the_calendar():
    prices = _panel()["prices"][["BTC", "ETH"]]
    weights = pd.DataFrame({"BTC": 0.5, "ETH": 0.3}, index=prices.index)
    weights[None] = 0.2
    with pytest.raises(ValueError, match=r"None"):
        simulate(prices, weights, engine="rsims", timing=resolve_execution(None), costs=Costs())


def test_an_all_zero_unpriced_column_is_not_a_weight(caplog):
    """The control: a column of zeros holds nothing, so it is not refused."""
    prices = _panel()["prices"][["BTC", "ETH"]]
    weights = pd.DataFrame({"BTC": 0.5, "ETH": 0.5, "GHOST": 0.0}, index=prices.index)
    with caplog.at_level(logging.WARNING):
        book = simulate(prices, weights, engine="rsims", timing=resolve_execution(None), costs=Costs())
    assert book.value.iloc[-1] > 0
