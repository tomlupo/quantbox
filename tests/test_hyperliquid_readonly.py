"""HyperliquidBroker's ``readonly`` contract, the one Kraken, Binance and IBKR carry.

quantbox-live's out-of-cycle working-order check (``resolve_working_orders.py``)
builds every book's broker with ``params["readonly"] = True``: resolution only
READS (``fetch_order_result``), and the flag makes "this job cannot trade"
structural. HyperliquidBroker had no such field, so building it raised
``TypeError: unexpected keyword argument 'readonly'`` and the Hyperliquid check
exited "could not check" every morning (2026-10-30).

ccxt is stubbed at the module boundary: ``ccxt.hyperliquid`` is replaced by a
fake exchange that records every call, so the broker is built through its real
``__init__`` / ``__post_init__`` (the path that failed in prod) with no
network, no credentials and no capital. A write that slipped past the guard
would show up in the fake's call log, which every write-path test asserts is
unchanged by the refused call.
"""

from __future__ import annotations

import types

import pandas as pd
import pytest

from quantbox.plugins.broker import hyperliquid as hl
from quantbox.plugins.broker.hyperliquid import HyperliquidBroker

BTC = "BTC/USDC:USDC"
_WRITE_METHODS = ("create_order", "cancel_order", "cancel_orders", "cancel_all_orders", "edit_order")


class _FakeExchange:
    """ccxt.hyperliquid stand-in: the reads the broker makes, and a log of every call."""

    def __init__(self, config):
        self.config = config
        self.calls: list[str] = []

    def __getattr__(self, name):
        # The venue's write methods (create_order, cancel_order, edit_order, ...):
        # recorded, and answered "accepted", so the default broker can still place
        # an order. A readonly test asserts none of them appears in the log.
        if name in _WRITE_METHODS:

            def _write(*args, **kwargs):
                self.calls.append(name)
                return {"id": "should-not-exist", "status": "open"}

            return _write
        raise AttributeError(name)

    def set_sandbox_mode(self, on):
        self.calls.append("set_sandbox_mode")

    def load_markets(self):
        self.calls.append("load_markets")
        return {
            BTC: {
                "symbol": BTC,
                "base": "BTC",
                "swap": True,
                "info": {"name": "BTC"},
                "precision": {"amount": 0.00001},
                "limits": {"amount": {"min": 0.00001}, "cost": {"min": 10.0}},
            }
        }

    def fetch_balance(self, params=None):
        self.calls.append("fetch_balance")
        return {"USDC": {"total": 1000.0, "free": 1000.0, "used": 0.0}}

    def fetch_positions(self):
        self.calls.append("fetch_positions")
        return [
            {
                "symbol": BTC,
                "side": "long",
                "contracts": 0.01,
                "notional": 600.0,
                "entryPrice": 60000.0,
                "unrealizedPnl": 0.0,
            }
        ]

    def fetch_ticker(self, symbol):
        self.calls.append("fetch_ticker")
        return {"last": 60000.0}

    def fetch_order(self, order_id, symbol):
        self.calls.append("fetch_order")
        return {"id": order_id, "status": "closed", "amount": 0.01, "filled": 0.01, "average": 60000.0}

    has = {"fetchFundingHistory": True}

    def fetch_my_trades(self, symbol, since=None):
        self.calls.append("fetch_my_trades")
        return [{"side": "buy", "amount": 0.01, "price": 60000.0, "datetime": "2026-10-30T05:00:00Z"}]

    def fetch_funding_history(self, since=None, params=None):
        self.calls.append("fetch_funding_history")
        return [{"amount": -0.5, "timestamp": since}]


@pytest.fixture
def fake_ccxt(monkeypatch):
    """Swap the module's ccxt for one whose ``hyperliquid`` is the fake; collect every instance."""
    built: list[_FakeExchange] = []

    def _factory(config):
        ex = _FakeExchange(config)
        built.append(ex)
        return ex

    fake = types.SimpleNamespace(
        hyperliquid=_factory,
        InsufficientFunds=type("InsufficientFunds", (Exception,), {}),
        InvalidOrder=type("InvalidOrder", (Exception,), {}),
    )
    monkeypatch.setattr(hl, "ccxt", fake)
    return built


def _build(**params) -> HyperliquidBroker:
    """Build the broker the way quantbox-live does: the registry's class, params from config."""
    from quantbox.registry import PluginRegistry

    cls = PluginRegistry.discover().brokers["hyperliquid.perps.v1"]
    return cls(wallet_address="0xwallet", private_key="0xkey", **params)


def _writes(broker) -> list[str]:
    return [c for c in broker._exchange.calls if c in _WRITE_METHODS]


# ---------------------------------------------------------------------------
# The prod failure: the broker could not be built with readonly=True
# ---------------------------------------------------------------------------


def test_readonly_broker_builds_through_the_registry(fake_ccxt):
    b = _build(readonly=True)
    assert b.readonly is True
    assert b.describe()["readonly"] is True
    assert len(fake_ccxt) == 1


def test_readonly_is_declared_in_the_params_schema():
    prop = HyperliquidBroker.meta.params_schema["properties"]["readonly"]
    assert prop["type"] == "boolean"
    assert prop["default"] is False


# ---------------------------------------------------------------------------
# Reads work on a readonly broker
# ---------------------------------------------------------------------------


def test_readonly_broker_resolves_an_order(fake_ccxt):
    b = _build(readonly=True)
    res = b.fetch_order_result("123", "BTC")
    assert res is not None
    assert res["status"] == "FILLED"
    assert res["qty"] == pytest.approx(0.01)
    assert "fetch_order" in b._exchange.calls
    assert _writes(b) == []


def test_readonly_broker_reads_positions_and_equity(fake_ccxt):
    b = _build(readonly=True)
    pos = b.get_positions()
    assert pos.set_index("symbol").loc["BTC", "qty"] == pytest.approx(0.01)
    assert b.get_equity() > 0
    assert _writes(b) == []


def test_readonly_broker_reads_fills_and_funding(fake_ccxt):
    b = _build(readonly=True)
    fills = b.fetch_fills("2026-10-29")
    assert len(fills) == 1
    assert b.fetch_funding_payments("2026-10-29", "2026-10-30") == pytest.approx(-0.5)
    assert _writes(b) == []


def test_readonly_broker_dry_run_rebalance_plans_without_placing(fake_ccxt):
    b = _build(readonly=True)
    out = b.rebalance_to_weights({"BTC": 0.0}, dry_run=True)
    assert out["dry_run"] is True
    assert out["trades"], "the dry run must still compute the close"
    assert _writes(b) == []


# ---------------------------------------------------------------------------
# Every write path refuses and never reaches the exchange
# ---------------------------------------------------------------------------

_ORDERS = pd.DataFrame([{"symbol": "BTC", "side": "buy", "qty": 0.01}])

WRITE_PATHS = {
    "place_orders": lambda b: b.place_orders(_ORDERS),
    "place_orders_empty": lambda b: b.place_orders(_ORDERS.iloc[0:0]),
    "place_order": lambda b: b.place_order("BTC", "buy", 0.01),
    "place_order_reduce_only": lambda b: b.place_order("BTC", "sell", 0.01, reduce_only=True),
    "close_position": lambda b: b.close_position("BTC"),
    "rebalance_to_weights": lambda b: b.rebalance_to_weights({"BTC": 0.0}),
}


@pytest.mark.parametrize("path", sorted(WRITE_PATHS))
def test_readonly_write_path_raises_before_the_exchange(fake_ccxt, path):
    b = _build(readonly=True)
    before = list(b._exchange.calls)
    with pytest.raises(PermissionError, match="readonly broker: order placement disabled"):
        WRITE_PATHS[path](b)
    # The guard is the first thing each write path does: not even a read follows it.
    assert b._exchange.calls == before
    assert _writes(b) == []


# ---------------------------------------------------------------------------
# The default is unchanged: a broker built without the flag still trades
# ---------------------------------------------------------------------------


def test_default_is_not_readonly(fake_ccxt):
    b = _build()
    assert b.readonly is False
    assert b.describe()["readonly"] is False


def test_default_broker_still_places_orders(fake_ccxt):
    b = _build()
    order = b.place_order("BTC", "buy", 0.01)
    assert order is not None
    assert _writes(b) == ["create_order"]
