"""Live trading uses the SAME rebalancing policy as the backtest (TOM-1518, docs/adr/0008 decision 11).

One config, one price panel, one held book, two doors:

- the backtest seam (:func:`quantbox.engine.simulate`) decides on bar ``d``
  and trades on bar ``d + 1`` (next-bar), against the held book it tracks;
- the trading pipeline (``trade.full_pipeline.v1``), run END TO END once per
  bar with the data up to ``d`` and a paper broker that holds that same book.

On every bar both must say the same thing: whether the rebalance is placed,
which instruments trade, and the target weight of each traded instrument. The
held book the broker carries is rebuilt here from the seam's own orders and
targets with the cost-free drift ADR-0008 states (not read from the seam), so
the test fails on a decision that differs, not on a shared helper.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantbox.engine import Costs, simulate
from quantbox.execution import resolve_execution
from quantbox.plugins.broker.sim import SimPaperBroker
from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline
from quantbox.plugins.rebalancing.standard_rebalancer import StandardRebalancer
from quantbox.store import FileArtifactStore

TICKERS = ("A", "B", "C")
N_BARS = 70
BOOK_VALUE = 100_000.0
RSIMS = {"capitalise_profits": True, "trade_buffer": 0.0, "margin": 0.0, "initial_cash": 10_000.0}

#: The same `rebalancing_policy` block both doors read.
POLICIES = {
    # robo's "tranches 5 + 2% corridor" (TOM-1513): a corridor hit rebalances the whole book
    "tranche5_corridor2": {
        "cadence": "tranche",
        "tranches": 5,
        "frequency": "daily",
        "trigger": "corridor",
        "width": 0.02,
    },
    # a period-END frequency: live must know the week ends today without seeing tomorrow
    "band_weekly_min_trade": {"policy": "band", "band": 0.03, "frequency": "weekly", "min_trade": 0.01},
    # tranche cadence + min_trade drops (partial rebalances), no trigger
    "tranche3_weekly_min_trade": {"policy": "tranche", "tranches": 3, "frequency": "weekly", "min_trade": 0.02},
}


def _prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="D")
    rng = np.random.default_rng(11)
    steps = rng.normal(0.0005, 0.03, size=(N_BARS, len(TICKERS)))
    return pd.DataFrame(100.0 * np.cumprod(1.0 + steps, axis=0), index=idx, columns=list(TICKERS))


def _decided(prices: pd.DataFrame) -> pd.DataFrame:
    """Long-only targets that move a little every bar and jump every 9 bars (net <= 0.95)."""
    rng = np.random.default_rng(5)
    base = rng.dirichlet(np.ones(len(TICKERS)), size=N_BARS // 9 + 1).repeat(9, axis=0)[:N_BARS]
    noise = rng.normal(0.0, 0.01, size=base.shape)
    w = np.clip(base + noise, 0.0, None)
    w = 0.9 * w / w.sum(axis=1, keepdims=True)
    return pd.DataFrame(w, index=prices.index, columns=list(TICKERS))


def _drift(held: np.ndarray, growth: np.ndarray) -> np.ndarray:
    """ADR-0008's cost-free drift: each weight moves with its price against a cash remainder."""
    value = held * growth
    total = (1.0 - held.sum()) + value.sum()
    return value / total


class _Data:
    def __init__(self, prices: pd.DataFrame) -> None:
        self.prices = prices

    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame({"symbol": list(TICKERS)})

    def load_market_data(self, universe: Any, asof: str, params: dict[str, Any]) -> dict[str, pd.DataFrame]:
        return {"prices": self.prices}


class _Replay:
    """A strategy that returns the decided history up to the run's last bar."""

    meta = type("M", (), {"name": "strategy.replay.v1"})()

    def __init__(self, decided: pd.DataFrame) -> None:
        self.decided = decided

    def run(self, data: Any, params: Any = None, context: Any = None) -> dict[str, Any]:
        return {"weights": self.decided.loc[: data["prices"].index[-1]]}


def _trading_decision(tmp_path, policy, prices, decided, d: int, held: np.ndarray) -> dict[str, Any]:
    """trade.full_pipeline.v1, end to end, on bar d with a paper broker that holds *held*."""
    marks = prices.iloc[d]
    positions = {t: float(held[j] * BOOK_VALUE / marks[t]) for j, t in enumerate(TICKERS) if held[j] != 0}
    broker = SimPaperBroker(cash=BOOK_VALUE * (1.0 - held.sum()), quote_currency="USD", positions=positions)
    params = {
        "strategies": [{"name": "strategy.replay.v1", "weight": 1.0, "params": {}}],
        "stable_coin_symbol": "USD",
        "trading_enabled": False,  # paper, and nothing is sent: the decision is what is compared
        "rebalancing_policy": policy,
        "_rebalancer_cfg": {"params": {}},
    }
    result = TradingPipeline().run(
        mode="paper",
        asof=str(prices.index[d].date()),
        params=params,
        data=_Data(prices.iloc[: d + 1]),
        store=FileArtifactStore(str(tmp_path / f"bar{d}"), "run"),
        broker=broker,
        risk=[],
        strategies=[_Replay(decided)],
        rebalancer=StandardRebalancer(),
    )
    return result.notes["rebalancing_policy"]


@pytest.mark.parametrize("policy", POLICIES.values(), ids=POLICIES.keys())
def test_backtest_and_trading_take_the_same_decision_on_every_bar(tmp_path, policy):
    prices = _prices()
    decided = _decided(prices)
    book = simulate(
        prices,
        decided,
        engine="rsims",
        timing=resolve_execution({}),
        costs=Costs(),
        engine_params=RSIMS,
        policy=policy,
    )
    orders = book.orders.reindex(columns=list(TICKERS)).to_numpy()
    targets = book.weights.reindex(columns=list(TICKERS)).to_numpy()
    p = prices.to_numpy()

    held = np.zeros(len(TICKERS))
    last: int | None = None
    placed_bars = 0
    for r in range(1, N_BARS):  # the decision on bar d = r - 1 trades on bar r (next-bar)
        drifted = held.copy() if last is None else _drift(held, p[r] / p[last])
        live = _trading_decision(tmp_path, policy, prices, decided, r - 1, drifted)

        seam_placed = bool(orders[r].any())
        assert live["placed"] == seam_placed, (
            f"bar {prices.index[r - 1].date()}: backtest placed={seam_placed}, trading placed={live['placed']} "
            f"({live['reason']})"
        )
        if seam_placed:
            placed_bars += 1
            seam_trades = [t for j, t in enumerate(TICKERS) if orders[r, j]]
            assert sorted(live["trades"]) == seam_trades, f"bar {prices.index[r - 1].date()}: traded instruments"
            for j, t in enumerate(TICKERS):
                if orders[r, j]:
                    assert live["targets"][t] == pytest.approx(targets[r, j], abs=1e-9), (
                        f"bar {prices.index[r - 1].date()}: target of {t}"
                    )
            held = drifted.copy()
            held[orders[r]] = targets[r, orders[r]]
            last = r

    # Not vacuous: the policy both placed and skipped rebalances over the panel.
    assert placed_bars >= 3, f"only {placed_bars} placed rebalance(s): the comparison would be vacuous"
    assert placed_bars < N_BARS - 1, "every bar placed: the trigger / frequency was never exercised"
