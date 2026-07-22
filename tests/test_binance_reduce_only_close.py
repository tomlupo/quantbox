"""Binance futures must honour reduce_only for sub-min closes (quantbox#87 / #138).

The futures rebalancer is shared across perp venues and marks a flat-target close
`reduce_only=True`. If a broker ignores that flag, a sub-min close is sent as a
normal order and rejected by the venue — re-creating the trapped-residual bug on
Binance. This pins that BinanceFuturesBroker consumes the flag: a sub-min
reduce-only close is SENT (not gated), with reduceOnly in the venue params; a
sub-min OPEN is still gated.
"""

from __future__ import annotations

import pandas as pd

from quantbox.plugins.broker.binance_futures import BinanceFuturesBroker


class _FakeExchange:
    def __init__(self):
        self.calls: list[dict] = []

    def create_order(self, **kw):
        self.calls.append(kw)
        return {"id": "1", "average": kw["amount"] and 100.0, "filled": kw["amount"], "price": 100.0}


def _broker() -> BinanceFuturesBroker:
    b = BinanceFuturesBroker.__new__(BinanceFuturesBroker)
    b._exchange = _FakeExchange()
    b.target_leverage = 1
    b.quote_currency = "USDT"
    b.telegram_token = ""
    b.telegram_chat_id = ""
    # Symbol info: $5 min notional, price $100.
    b._get_symbol_info = lambda s: {"step_size": 0.001, "min_notional": 5.0}
    b.get_price = lambda s: 100.0
    b.set_leverage = lambda s, lev: None
    return b


def test_sub_min_reduce_only_close_is_sent_with_reduceonly_param():
    b = _broker()
    # 0.02 * $100 = $2 notional, below the $5 minimum.
    result = b.place_order("ARB", "sell", 0.02, reduce_only=True)
    assert result is not None, "a sub-min reduce-only close must be sent, not gated"
    assert b._exchange.calls[-1]["params"] == {"reduceOnly": True}


def test_sub_min_open_is_still_gated():
    b = _broker()
    result = b.place_order("ARB", "buy", 0.02, reduce_only=False)
    assert result is None, "a sub-min OPEN must still be blocked by the min-notional floor"
    assert not b._exchange.calls, "the open must never reach the venue"


def test_place_orders_threads_reduce_only_nan_safe():
    """A mixed-column order frame fills a missing reduce_only with NaN; bool(NaN)
    is True, which would send OPENS reduce-only. Must be NaN-safe."""
    b = _broker()
    orders = pd.DataFrame(
        [
            {"symbol": "PEPE", "side": "buy", "qty": 1.0},  # no reduce_only -> NaN in frame
            {"symbol": "ARB", "side": "sell", "qty": 0.02, "reduce_only": True},
        ]
    )
    b.place_orders(orders)
    params = [c.get("params", {}) for c in b._exchange.calls]
    # PEPE (open) must NOT be reduce-only; ARB (close) must be.
    assert {"reduceOnly": True} in params
    assert {} in params
