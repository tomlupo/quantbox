"""Every builtin broker returns one place_orders row per submitted order, in order (TOM-2422).

The contract is ``BrokerPlugin`` in ``quantbox.contracts``: one result row per
submitted order, in submission order, a failed or skipped order included. The
executor binds results to orders by position within (symbol, side). PR #310's
review found two brokers that dropped a rejected or unpriced order's row, so
the next same-key order's fill was booked to it.

Each broker is faked at its venue call only; its own ``place_orders`` loop
runs. The batch is a flip (two ETH sells: a reduce-only close, then an open)
plus a BTC buy, and each broker is told to reject, skip or fill the orders
the way its venue can. A builtin broker with no factory here fails the test,
so a new broker cannot skip the contract.

Red at 5b19df7 (before #310): binance.futures.v1 and sim.futures_paper.v1.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from quantbox.plugins.builtins import BUILTIN_PLUGINS

ORDERS = pd.DataFrame(
    [
        {"symbol": "ETH", "side": "sell", "qty": 0.02, "price": 2000.0, "reduce_only": True},
        {"symbol": "ETH", "side": "sell", "qty": 0.03, "price": 2000.0, "reduce_only": False},
        {"symbol": "BTC", "side": "buy", "qty": 0.001, "price": 60000.0, "reduce_only": False},
    ]
)


def _ccxt_fill(qty: float, price: float = 2000.0, oid: str = "1") -> dict:
    return {"id": oid, "status": "closed", "filled": qty, "amount": qty, "average": price}


def _ccxt_perp(module: str, cls: str):
    """Hyperliquid or Binance futures: the venue rejects the reduce-only close, fills the rest."""
    import importlib

    broker_cls = getattr(importlib.import_module(f"quantbox.plugins.broker.{module}"), cls)
    broker = broker_cls.__new__(broker_cls)
    broker.quote_currency = "USDT"

    def place_order(symbol, side, quantity, order_type="market", reduce_only=False):
        return None if reduce_only else _ccxt_fill(quantity, oid=f"{symbol}-{side}")

    broker.place_order = place_order
    broker._refetch_order = lambda order, symbol: None
    return broker


def _kraken():
    from quantbox.plugins.broker.kraken import SKIP_ORDER, KrakenBroker

    broker = KrakenBroker.__new__(KrakenBroker)
    broker.readonly = False
    outcomes = iter([None, SKIP_ORDER, _ccxt_fill(0.001, 60000.0)])  # failed, dust, filled
    broker._place_one = lambda symbol, side, qty, price=None: next(outcomes)
    broker._refetch_order = lambda order, symbol: None
    return broker


def _binance_spot():
    from quantbox.plugins.broker.binance import BinanceBroker

    broker = BinanceBroker.__new__(BinanceBroker)
    broker.readonly = False
    calls = iter([RuntimeError("Filter failure: LOT_SIZE"), None, None])

    def create_order(**kwargs):
        exc = next(calls)
        if exc:
            raise exc
        qty = kwargs["quantity"]
        return {"orderId": 7, "status": "FILLED", "executedQty": qty, "origQty": qty, "cummulativeQuoteQty": qty * 10}

    broker.client = SimpleNamespace(create_order=create_order)
    return broker


def _ibkr(monkeypatch):
    from quantbox.plugins.broker import ibkr

    statuses = iter(
        [
            (SimpleNamespace(status="Inactive", filled=0.0, remaining=1.0, avgFillPrice=0.0), 201),  # rejected
            (SimpleNamespace(status="Filled", filled=0.03, remaining=0.0, avgFillPrice=2000.0), 0),
            (SimpleNamespace(status="Filled", filled=0.001, remaining=0.0, avgFillPrice=60000.0), 0),
        ]
    )

    class _IB:
        def qualifyContracts(self, *contracts):
            return list(contracts)

        def placeOrder(self, contract, order):
            status, code = next(statuses)
            order.orderId = 42
            log = [SimpleNamespace(message="Error 201: rejected" if code else "", errorCode=code)]
            return SimpleNamespace(contract=contract, order=order, orderStatus=status, fills=[], log=log)

        def sleep(self, seconds):
            return None

    monkeypatch.setattr(ibkr, "Stock", lambda sym, exch, ccy: SimpleNamespace(symbol=sym), raising=False)
    monkeypatch.setattr(
        ibkr, "MarketOrder", lambda a, q: SimpleNamespace(action=a, totalQuantity=q, lmtPrice=0.0), raising=False
    )
    monkeypatch.setattr(
        ibkr, "LimitOrder", lambda a, q, p: SimpleNamespace(action=a, totalQuantity=q, lmtPrice=p), raising=False
    )
    broker = ibkr.IBKRBroker.__new__(ibkr.IBKRBroker)
    broker.readonly = False
    broker.ib = _IB()
    return broker


def _futures_paper():
    from quantbox.plugins.broker.futures_paper import FuturesPaperBroker

    broker = FuturesPaperBroker(margin_balance=100_000.0)
    broker.prices = {"BTC": 60_000.0}  # ETH has no price: both ETH orders are skipped, visibly
    return broker


def _sim():
    from quantbox.plugins.broker.sim import SimPaperBroker

    broker = SimPaperBroker()
    broker.prices = {"BTC": 60_000.0}
    return broker


def _unpriced(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(price=0.0)


def _stub(module: str):
    import importlib

    return importlib.import_module(f"quantbox.plugins.broker.{module}").PaperBrokerStub()


FACTORIES = {
    "hyperliquid.perps.v1": lambda mp: (_ccxt_perp("hyperliquid", "HyperliquidBroker"), ORDERS),
    "binance.futures.v1": lambda mp: (_ccxt_perp("binance_futures", "BinanceFuturesBroker"), ORDERS),
    "kraken.spot.v1": lambda mp: (_kraken(), ORDERS),
    "binance.live.v1": lambda mp: (_binance_spot(), _unpriced(ORDERS)),
    "ibkr.live.v1": lambda mp: (_ibkr(mp), _unpriced(ORDERS)),
    # The paper brokers skip an order with no price: only BTC has one.
    "sim.futures_paper.v1": lambda mp: (_futures_paper(), _unpriced(ORDERS)),
    "sim.paper.v1": lambda mp: (_sim(), _unpriced(ORDERS)),
    "binance.paper.stub.v1": lambda mp: (_stub("binance_stub"), ORDERS),
    "ibkr.paper.stub.v1": lambda mp: (_stub("ibkr_stub"), ORDERS),
}


#: What each fake makes the venue do: the case is only a contract case if the failed or skipped row is really there.
EXPECTED = {
    "hyperliquid.perps.v1": ["FAILED", "FILLED", "FILLED"],
    "binance.futures.v1": ["FAILED", "FILLED", "FILLED"],
    "kraken.spot.v1": ["FAILED", "SKIPPED", "FILLED"],
    "binance.live.v1": ["FAILED", "FILLED", "FILLED"],
    "ibkr.live.v1": ["FAILED", "FILLED", "FILLED"],
    "sim.futures_paper.v1": ["SKIPPED", "SKIPPED", "FILLED"],
    "sim.paper.v1": ["SKIPPED", "SKIPPED", "FILLED"],
    "binance.paper.stub.v1": ["FILLED", "FILLED", "FILLED"],
    "ibkr.paper.stub.v1": ["FILLED", "FILLED", "FILLED"],
}


def test_every_builtin_broker_has_a_contract_case():
    assert sorted(FACTORIES) == sorted(BUILTIN_PLUGINS["broker"]) == sorted(EXPECTED)


@pytest.mark.parametrize("name", sorted(BUILTIN_PLUGINS["broker"]))
def test_place_orders_returns_one_row_per_order_in_submission_order(name, monkeypatch):
    broker, orders = FACTORIES[name](monkeypatch)
    rows = broker.place_orders(orders)
    assert len(rows) == len(orders), rows.to_string()
    sent = list(zip(orders["symbol"], orders["side"], strict=True))
    got = list(zip(rows["symbol"].astype(str), rows["side"].astype(str).str.lower(), strict=True))
    assert got == sent
    assert list(rows["status"]) == EXPECTED[name]
