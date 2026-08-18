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

    # A no-op on the BOOK — but reported, not silent: the order comes back as
    # SKIPPED so the pipeline can tell a deliberate decline from a missed fill.
    assert (fills["status"] == "SKIPPED").all()
    assert "BTC" not in b.positions
    assert b.margin_balance == 100_000.0  # no fee charged either


def test_reduce_only_that_would_add_to_a_position_is_a_noop():
    b = _broker()
    b.positions["BTC"] = 1.0
    b.entry_prices["BTC"] = 60_000.0

    fills = b.place_orders(_order("buy", 0.5, reduce_only=True))

    assert (fills["status"] == "SKIPPED").all()
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


def test_reduce_only_bypasses_the_position_limit_cap_and_never_flips():
    """The cap-skip is load-bearing, not tidiness — guard the reasoning.

    The position-limit cap sizes the fill *to* the limit from a zero base
    (``allowed_qty * sign(signed) - old_qty``), so on a position that is STILL
    over its limit after the reduce, the two terms add instead of cancelling and
    the order overshoots through zero.

    Without the ``not reduce_only`` guard this exact case fills as a 150-unit
    sell and lands at -50, flipping a +100 long into a 50 short. Every other
    reduce-only test leaves the cap disengaged, so this is the only test that
    fails if the guard is removed.
    """
    b = FuturesPaperBroker(margin_balance=100_000.0)
    b.prices = {"ETH": 100.0}
    b.positions["ETH"] = 100.0
    b.entry_prices["ETH"] = 100.0
    b.position_limits = {"ETH": 5_000.0}  # 50 units — the position is already over it

    orders = pd.DataFrame([{"symbol": "ETH", "side": "sell", "qty": 30.0, "reduce_only": True}])
    b.place_orders(orders)

    # Reduced by exactly what was asked, and still long.
    assert b.positions["ETH"] == 70.0
    assert b.positions["ETH"] > 0, "reduce-only order flipped the position through zero"


def test_non_reduce_only_reducing_order_still_hits_the_known_cap_bug():
    """Pins the CURRENT (broken) behaviour of the cap for unflagged orders.

    This is TOM-886, deliberately not fixed here: the same algebra flips a
    reducing order that is not flagged reduce_only. Asserting it keeps the bug
    visible and makes the TOM-886 fix announce itself by breaking this test —
    at which point the assertion should be inverted, not deleted.
    """
    b = FuturesPaperBroker(margin_balance=100_000.0)
    b.prices = {"ETH": 100.0}
    b.positions["ETH"] = 100.0
    b.entry_prices["ETH"] = 100.0
    b.position_limits = {"ETH": 5_000.0}

    orders = pd.DataFrame([{"symbol": "ETH", "side": "sell", "qty": 30.0}])
    b.place_orders(orders)

    # KNOWN BUG (TOM-886): a 30-unit trim overshoots and flips the book short.
    assert b.positions["ETH"] < 0, "TOM-886 appears fixed — invert this assertion"


# ---------------------------------------------------------------------------
# Declined orders must be VISIBLE, not silent (the pipeline's SKIPPED contract)
# ---------------------------------------------------------------------------


def _skipped_rows(fills: pd.DataFrame) -> pd.DataFrame:
    return fills[fills["status"] == "SKIPPED"] if "status" in fills.columns else fills.iloc[0:0]


def test_reduce_only_on_flat_emits_a_skipped_row():
    """A declined order must report SKIPPED, not vanish.

    `trading_pipeline` distinguishes filled / FAILED / intentional no-op, and a
    broker signals the third with `status="SKIPPED"`. Returning nothing drops the
    order out of `orders_details` and leaves a dangling intent-FIFO entry, which
    reads downstream as a MISSED FILL rather than a deliberate decline.
    """
    b = _broker()
    b.positions.pop("BTC", None)

    fills = b.place_orders(_order("sell", 0.5, reduce_only=True))

    rows = _skipped_rows(fills)
    assert len(rows) == 1, "declined reduce-only order emitted no SKIPPED row"
    row = rows.iloc[0]
    assert row["symbol"] == "BTC"
    assert row["side"] == "sell"
    assert row["qty"] == 0.5, "the REQUESTED qty is what the book wanted; keep it"
    assert row["price"] == 0.0
    assert row["notional"] == 0.0
    assert row["fee"] == 0.0
    assert "flat" in str(row["error"]).lower()


def test_reduce_only_increase_emits_a_skipped_row():
    b = _broker()
    b.positions["BTC"] = 1.0
    b.entry_prices["BTC"] = 60_000.0

    fills = b.place_orders(_order("buy", 0.5, reduce_only=True))

    rows = _skipped_rows(fills)
    assert len(rows) == 1
    assert "increase" in str(rows.iloc[0]["error"]).lower()
    assert b.positions["BTC"] == 1.0, "a declined order must not move the book"


def test_skipped_rows_do_not_pollute_the_fill_log_or_pnl():
    """SKIPPED is not a fill: it must not reach the fill log, fees or margin."""
    b = _broker()
    b.positions.pop("BTC", None)
    before_balance = b.margin_balance
    before_fees = b._cumulative_fees
    before_log = len(b._fill_log)

    b.place_orders(_order("sell", 0.5, reduce_only=True))

    assert len(b._fill_log) == before_log, "a skipped order was logged as a fill"
    assert b.margin_balance == before_balance
    assert b._cumulative_fees == before_fees


def test_skipped_columns_survive_an_all_skipped_batch():
    """A batch of nothing but declines must still carry status/error columns."""
    b = _broker()
    b.positions.pop("BTC", None)

    fills = b.place_orders(_order("sell", 0.5, reduce_only=True))

    assert "status" in fills.columns
    assert "error" in fills.columns
