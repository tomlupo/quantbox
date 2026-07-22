"""Regression tests for the carver-HL silent-failure incident (quantbox#87).

Live symptom: from 2026-06-20 to 2026-07-22 the carver-HL book emitted the SAME
sub-$10 ETH/SOL/ARB close-out orders every single cycle. Hyperliquid rejected
every one of them ($10 venue minimum), yet each run was recorded as healthy
(``rebalance_frozen=0.0``) and Telegram reported the orders as FILLED. 33
consecutive days, ~71 failed orders, zero escalation.

Four distinct defects made that possible; each has a test below.

1. ``place_order`` claimed a fill on ACCEPTANCE, defaulting ``filled`` to the
   REQUESTED quantity — so a rejected order logged "Order filled" and sent a
   green Telegram fill message.
2. The 2026-06-20 "closing positions are min-notional exempt" fix exempted exits
   from the VENUE's floor as well as our own, converting a silent client-side
   suppression into a doomed order re-sent forever.
3. Both freeze guards are conjunctions requiring a TOTAL standstill, so
   "some filled, some rejected" — the commonest real failure — alerted nobody.
4. ``get_balance`` returned ``{"total": 0}`` on an API exception: a fabricated
   value in a position-sizing context, which made an unreadable balance
   indistinguishable from an empty book.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantbox.exceptions import BrokerExecutionError
from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline
from quantbox.plugins.rebalancing.futures_rebalancer import FuturesRebalancer

MIN_TRADE = 0.01
MIN_NOTIONAL = 10.0


def _rebal_row(asset, *, action, delta_qty, price, weight_delta, target_weight, current_qty):
    return {
        "Asset": asset,
        "Trade Action": action,
        "Delta Quantity": delta_qty,
        "Price": price,
        "Weight Delta": weight_delta,
        "Target Weight": target_weight,
        "Current Quantity": current_qty,
    }


# ---------------------------------------------------------------------------
# 2. A close-out below the VENUE minimum is trapped, not re-sent forever
# ---------------------------------------------------------------------------
def test_close_out_below_venue_minimum_is_trapped_not_emitted():
    """The live ETH case: 0.0022 ETH @ $1930 = $4.25, flat target, HL min $10.

    Emitting this produces "placement failed" every cycle forever. It must be
    classified as a trapped residual and NOT sent.
    """
    reb = FuturesRebalancer()
    df = pd.DataFrame(
        [
            _rebal_row(
                "ETH",
                action="Sell",
                delta_qty=-0.0022,
                price=1930.85,  # $4.25 notional
                weight_delta=-0.05,
                target_weight=0.0,
                current_qty=0.0022,
            )
        ]
    )
    orders = reb._create_executable_orders(
        df,
        min_trade_size=MIN_TRADE,
        min_notional=MIN_NOTIONAL,
        # The venue's real floor — this is what the old code ignored for closes.
        min_notional_map={"ETH": 10.0},
    )
    row = orders.iloc[0]
    assert row["Order Status"] == "Trapped residual"
    assert not bool(row["Executable"]), "a venue-rejectable close-out must not be sent"


def test_close_out_above_venue_minimum_still_exits():
    """The exemption must still work: a close-out the venue WILL accept is sent,
    even though it is below an operator-configured churn floor."""
    reb = FuturesRebalancer()
    df = pd.DataFrame(
        [
            _rebal_row(
                "SOL",
                action="Sell",
                delta_qty=-0.5,
                price=80.0,  # $40 notional: above the $10 venue floor
                weight_delta=-0.4,
                target_weight=0.0,
                current_qty=0.5,
            )
        ]
    )
    orders = reb._create_executable_orders(
        df,
        min_trade_size=MIN_TRADE,
        min_notional=100.0,  # operator churn floor ABOVE the trade size
        min_notional_configured=100.0,
        min_notional_map={"SOL": 10.0},
    )
    row = orders.iloc[0]
    assert row["Order Status"] == "To be placed"
    assert bool(row["Executable"]), "exits must bypass our own churn floor"


def test_no_venue_floor_known_falls_back_to_exemption():
    """With no per-pair snapshot the old behaviour holds — we cannot prove the
    venue would reject it, so we still try to exit."""
    reb = FuturesRebalancer()
    df = pd.DataFrame(
        [
            _rebal_row(
                "SOL",
                action="Sell",
                delta_qty=-0.03,
                price=71.30,
                weight_delta=-0.025,
                target_weight=0.0,
                current_qty=0.03,
            )
        ]
    )
    orders = reb._create_executable_orders(df, min_trade_size=MIN_TRADE, min_notional=MIN_NOTIONAL)
    assert bool(orders.iloc[0]["Executable"])


# ---------------------------------------------------------------------------
# 3. Partial failures alert (the guard that could never fire)
# ---------------------------------------------------------------------------
class _Broker:
    """Broker that fills one order and rejects another — the live pattern."""

    def __init__(self):
        self.notices: list[str] = []

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, o in orders.iterrows():
            if o["symbol"] == "PEPE":
                rows.append({"symbol": "PEPE", "side": o["side"], "qty": o["qty"], "price": 1.0, "status": "FILLED"})
            else:
                rows.append(
                    {
                        "symbol": o["symbol"],
                        "side": o["side"],
                        "qty": 0.0,
                        "price": 0.0,
                        "status": "FAILED",
                        "error": "placement failed",
                    }
                )
        return pd.DataFrame(rows)

    def get_positions(self):
        return pd.DataFrame([{"symbol": "ETH", "qty": 0.0022}])

    def notify(self, message: str) -> bool:
        self.notices.append(message)
        return True


def _executable(asset: str, action: str, qty: float, price: float) -> dict:
    return {
        "Asset": asset,
        "Action": action,
        "Adjusted Quantity": qty,
        "Price": price,
        "Notional Value": qty * price,
        "Order Status": "To be placed",
        "Executable": True,
    }


def test_partial_order_failure_alerts():
    """One fill + one rejection must alert.

    This is the exact shape of every carver-HL run from 2026-06-20 on. Neither
    existing freeze guard fires here: ``total_executed`` is non-zero, and the
    broker returns populated rows rather than an empty frame.
    """
    pipe = TradingPipeline()
    broker = _Broker()
    orders = pd.DataFrame([_executable("PEPE", "Buy", 100.0, 1.0), _executable("ETH", "Sell", 0.0022, 1930.0)])

    report = pipe._execute_orders(
        broker=broker,
        orders_df=orders,
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )

    assert report["summary"]["total_failed"] == 1
    assert report["summary"]["total_executed"] == 1
    # The run is NOT frozen (something traded) — but it must still alert.
    assert not report.get("frozen")
    assert report.get("order_failures") == 1
    assert any("ORDER(S) FAILED" in m for m in broker.notices), (
        "a rejected order must reach a human even when other orders filled"
    )


def test_trapped_residual_alerts_even_when_other_orders_trade():
    """A trapped residual is real unwanted exposure regardless of the rest of the
    batch, so it must alert on its own."""
    pipe = TradingPipeline()
    broker = _Broker()
    orders = pd.DataFrame(
        [
            _executable("PEPE", "Buy", 100.0, 1.0),
            {
                "Asset": "ETH",
                "Action": "Sell",
                "Adjusted Quantity": 0.0,
                "Price": 1930.0,
                "Notional Value": 4.25,
                "Order Status": "Trapped residual",
                "Executable": False,
            },
        ]
    )

    report = pipe._execute_orders(
        broker=broker,
        orders_df=orders,
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )

    assert "ETH" in report.get("trapped_residuals", "")
    assert any("TRAPPED RESIDUAL" in m for m in broker.notices)


# ---------------------------------------------------------------------------
# 4. An unreadable balance is unknown, not zero
# ---------------------------------------------------------------------------
def test_get_balance_raises_instead_of_fabricating_zero():
    from quantbox.plugins.broker.hyperliquid import HyperliquidBroker

    broker = HyperliquidBroker.__new__(HyperliquidBroker)
    broker.telegram_token = ""
    broker.telegram_chat_id = ""

    class _Boom:
        def fetch_balance(self, *a, **k):
            raise RuntimeError("API down")

    broker._exchange = _Boom()

    with pytest.raises(BrokerExecutionError, match="unknown equity"):
        broker.get_balance()
