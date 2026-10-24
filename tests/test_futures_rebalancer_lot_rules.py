"""The order frame carries the quantity the venue gets (TOM-2207): a whole-path replay.

``rebalancing.futures.v1`` used to plan ``abs(delta)`` unrounded and mark it "To
be placed"; the broker then rounded it down to the step (TOM-1721) and judged it
again. So the run's order frame and the venue disagreed: carver-HL's
2026-10-09 frame planned BTC 0.000169452 while the venue got a rounded size.

The replay runs the real path: ``FuturesRebalancer.generate_orders`` against a
real ``HyperliquidBroker`` (its own snapshot, its own ``place_order``), then the
real execution stage (``quantbox.trade.executor.execute_orders``). Only the venue
boundary is faked: ccxt's market data, prices, positions, equity, and
``create_order``, which records each amount it is sent.

Fixture source (read-only copies, 2026-10-22):

* The book: prod run
  ``2026-10-09__trade_full_pipeline_v1__e7d6467a1fcd__20261009T040026Z``
  (carver-HL, quantbox v0.10.0) on core. ``rebalancing.parquet`` gives each
  name's current quantity, price and target weight; ``run_manifest.json`` gives
  the broker equity ``portfolio_value_broker_equity_usd`` = 70.290281. The
  rebalancer params are quantbox-live ``configs/carver_hyperliquid.yaml``.
* The lot rules: Hyperliquid's public ccxt markets, read 2026-10-22:
  ``precision.amount`` (the step) and ``limits.cost.min`` ($10) per coin. The
  run stored no market snapshot.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from quantbox.plugins.broker.hyperliquid import HyperliquidBroker
from quantbox.plugins.rebalancing.futures_rebalancer import FuturesRebalancer
from quantbox.trade.executor import execute_orders

#: Broker equity of the run, the margined book's value.
EQUITY = 70.290281

#: name: (current quantity, price, target weight) from rebalancing.parquet.
BOOK = {
    "BTC": (0.0005, 82490.5, 0.387920787981984),
    "ETH": (0.0079, 2493.75, 0.18288730252899835),
    "HYPE": (0.0, 85.4425, 0.171539992459021),
    "NEAR": (2.5, 4.7932, 0.0),
}

#: name: Hyperliquid ccxt ``precision.amount`` (the step), read 2026-10-22.
STEPS = {"BTC": 1e-05, "ETH": 0.0001, "HYPE": 0.01, "NEAR": 0.1}

#: quantbox-live configs/carver_hyperliquid.yaml, rebalancing params; mode as the run.
PARAMS = {"tranches": 1, "max_leverage": 2, "min_trade_size": 0.01, "mode": "live"}


class _Venue:
    """The ccxt boundary: records the amount of every ``create_order``."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, float]] = []

    def create_order(self, symbol, type, side, amount, price=None, params=None):  # noqa: A002 - ccxt's name
        self.sent.append((symbol.split("/")[0], side, amount))
        return {"id": str(len(self.sent)), "status": "closed", "filled": amount, "amount": amount, "average": price}


def _broker() -> tuple[HyperliquidBroker, _Venue]:
    venue = _Venue()
    b = object.__new__(HyperliquidBroker)
    b._exchange = venue
    b._markets = {
        f"{coin}/USDC:USDC": {
            "precision": {"amount": step},
            "limits": {"amount": {"min": None}, "cost": {"min": 10.0}},
        }
        for coin, step in STEPS.items()
    }
    b._coin_to_market = {}
    b.risk = SimpleNamespace(min_order_notional=10.0)
    b.get_price = lambda s: BOOK[s][1]
    b.get_balance = lambda: {"total": EQUITY}
    b._get_positions_dict = lambda: {
        coin: {"side": "long", "size": qty, "notional": qty * px, "entry_price": px, "unrealized_pnl": 0.0}
        for coin, (qty, px, _) in BOOK.items()
        if qty
    }
    return b, venue


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.setattr("quantbox.plugins.broker.hyperliquid.time.sleep", lambda s: None)
    broker, venue = _broker()
    weights = {coin: w for coin, (_, _, w) in BOOK.items() if w}
    result = FuturesRebalancer().generate_orders(weights=weights, broker=broker, params=dict(PARAMS))
    orders = result["orders"]
    execute_orders(broker=broker, orders_df=orders, trading_enabled=True, alert=lambda *a, **k: True)
    return orders, venue


def test_the_replay_sent_orders(replay):
    """The run had orders to send: an empty replay would prove nothing."""
    orders, venue = replay
    executable = orders[orders["Executable"]]
    print(orders[["Asset", "Action", "Raw Quantity", "Adjusted Quantity", "Order Status", "Reason"]].to_string())
    print("venue calls:", venue.sent)
    assert len(executable) >= 3
    assert len(venue.sent) >= 3


def test_every_planned_order_reaches_the_venue_at_its_planned_quantity(replay):
    """No shrink and no drop at the broker: each executable row is one venue call at its own quantity."""
    orders, venue = replay
    planned = sorted(
        (str(r["Asset"]), "sell" if str(r["Action"]).lower() == "sell" else "buy", float(r["Adjusted Quantity"]))
        for _, r in orders[orders["Executable"]].iterrows()
    )
    assert sorted(venue.sent) == planned


def test_btc_is_planned_on_the_venue_step(replay):
    """The 2026-10-09 BTC trim: 0.000169452 unrounded, 0.00016 on the 1e-05 step."""
    orders, _ = replay
    btc = orders.set_index("Asset").loc["BTC"]
    assert btc["Raw Quantity"] == pytest.approx(0.00016945210426661162)
    assert btc["Adjusted Quantity"] == 0.00016
    assert bool(btc["Executable"])
