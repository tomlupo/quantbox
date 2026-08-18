"""Regression tests for reduce-only semantics in FuturesPaperBroker.

Live venues clamp ``reduceOnly`` orders: they can never open a position and
never flip one through zero. Paper used to ignore the flag entirely, taking the
full delta where live is clamped — a sim that flatters itself. TOM-402 widened
``reduce_only`` from terminal closes to every exposure-reducing order, which
made the divergence a recurring, price-driven condition.
"""

from __future__ import annotations

import pandas as pd

from quantbox.plugins.broker.futures_paper import FuturesPaperBroker


def _broker(**kwargs) -> FuturesPaperBroker:
    b = FuturesPaperBroker(margin_balance=100_000.0, **kwargs)
    b.prices = {"BTC": 60_000.0}
    return b


def _order(side: str, qty: float, **extra) -> pd.DataFrame:
    return pd.DataFrame([{"symbol": "BTC", "side": side, "qty": qty, **extra}])


def test_reduce_only_sell_clamps_at_zero_on_a_long():
    b = _broker()
    b.positions["BTC"] = 1.0
    b.entry_prices["BTC"] = 60_000.0

    fills = b.place_orders(_order("sell", 3.0, reduce_only=True))

    assert "BTC" not in b.positions  # exact flat, never flipped short
    assert len(fills) == 1
    # The reported fill describes the clamped order, not the 3.0 requested.
    assert fills.iloc[0]["qty"] == 1.0
    assert fills.iloc[0]["notional"] == 1.0 * fills.iloc[0]["price"]


def test_reduce_only_on_flat_position_is_a_noop():
    b = _broker()

    fills = b.place_orders(_order("sell", 1.0, reduce_only=True))

    assert fills.empty
    assert "BTC" not in b.positions
    assert b.margin_balance == 100_000.0  # no fee charged either


def test_reduce_only_that_would_add_to_a_position_is_a_noop():
    b = _broker()
    b.positions["BTC"] = 1.0
    b.entry_prices["BTC"] = 60_000.0

    fills = b.place_orders(_order("buy", 0.5, reduce_only=True))

    assert fills.empty
    assert b.positions["BTC"] == 1.0
    assert b.margin_balance == 100_000.0


def test_reduce_only_exact_close_realises_pnl_and_pops_position():
    b = _broker()
    b.positions["BTC"] = 1.0
    b.entry_prices["BTC"] = 50_000.0

    fills = b.place_orders(_order("sell", 1.0, reduce_only=True))

    assert "BTC" not in b.positions
    assert "BTC" not in b.entry_prices
    fill_price = fills.iloc[0]["price"]
    expected = 100_000.0 + (fill_price - 50_000.0) - fills.iloc[0]["fee"]
    assert b.margin_balance == expected


def test_reduce_only_short_cover_clamps_at_zero():
    b = _broker()
    b.positions["BTC"] = -1.0
    b.entry_prices["BTC"] = 60_000.0

    fills = b.place_orders(_order("buy", 2.5, reduce_only=True))

    assert "BTC" not in b.positions
    assert fills.iloc[0]["qty"] == 1.0


def test_normal_order_is_unchanged_by_the_flag():
    """reduce_only=False must still be free to flip the position through zero."""
    b = _broker()
    b.positions["BTC"] = 1.0
    b.entry_prices["BTC"] = 60_000.0

    fills = b.place_orders(_order("sell", 3.0, reduce_only=False))

    assert b.positions["BTC"] == -2.0
    assert fills.iloc[0]["qty"] == 3.0


def test_frame_without_reduce_only_column_behaves_as_before():
    b = _broker()
    b.positions["BTC"] = 1.0
    b.entry_prices["BTC"] = 60_000.0

    fills = b.place_orders(_order("sell", 3.0))

    assert b.positions["BTC"] == -2.0
    assert fills.iloc[0]["qty"] == 3.0


def test_nan_reduce_only_defaults_to_false():
    """A mixed-column frame fills the flag NaN; bool(NaN) is True, so guard it."""
    b = _broker()
    b.positions["BTC"] = 1.0
    b.entry_prices["BTC"] = 60_000.0

    orders = pd.DataFrame(
        [
            {"symbol": "BTC", "side": "sell", "qty": 3.0},
            {"symbol": "ETH", "side": "sell", "qty": 1.0, "reduce_only": True},
        ]
    )
    assert pd.isna(orders.loc[0, "reduce_only"])

    b.place_orders(orders)

    assert b.positions["BTC"] == -2.0


def test_reduce_only_partial_reduce_leaves_entry_price_untouched():
    b = _broker()
    b.positions["BTC"] = 2.0
    b.entry_prices["BTC"] = 50_000.0

    fills = b.place_orders(_order("sell", 0.5, reduce_only=True))

    assert b.positions["BTC"] == 1.5
    assert b.entry_prices["BTC"] == 50_000.0
    assert fills.iloc[0]["qty"] == 0.5
