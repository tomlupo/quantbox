"""Regression tests for the carver-HL silent-failure incident (quantbox#87).

Live symptom: from 2026-06-20 to 2026-07-22 the carver-HL book emitted the SAME
sub-$10 ETH/SOL/ARB close-out orders every single cycle, all recorded as
"placement failed", yet each run was logged as healthy (``rebalance_frozen=0.0``)
and Telegram reported the orders as FILLED. 33 consecutive days, ~71 failures.

Root cause, VERIFIED LIVE 2026-07-22: the failures were NOT the venue's doing.
Hyperliquid ACCEPTS a full reduce-only close below its $10 minimum (SOL 0.03,
$2.33, filled at 77.773). The orders were rejected by OUR OWN client-side
min-notional guard (``hyperliquid.py:641``) before submission, and the
"placement failed" error string made it look like Hyperliquid's. The exit was
available the whole time — the incident was entirely self-inflicted.

Defects with tests below:

1. ``place_order`` claimed a fill on ACCEPTANCE, defaulting ``filled`` to the
   REQUESTED quantity — so an unfilled order logged "Order filled" and sent a
   green Telegram fill message.
2. A flat-target close must be sent REDUCE-ONLY (venue-exempt from the $10
   minimum, can't flip through zero), not suppressed. The earlier "trapped
   residual" suppression was built on the false premise that the venue rejected
   these; it is removed.
3. Both freeze guards are conjunctions requiring a TOTAL standstill, so
   "some filled, some failed" — the commonest real failure — alerted nobody.
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
# 2. A sub-$10 close-out is SENT reduce-only, not suppressed
# ---------------------------------------------------------------------------
def test_close_out_below_venue_minimum_is_sent_reduce_only():
    """The live ETH case: 0.0022 ETH @ $1930 = $4.25, flat target, HL min $10.

    Verified live 2026-07-22: Hyperliquid ACCEPTS a full reduce-only close below
    its $10 minimum. The sub-$10 close must therefore be executable and flagged
    reduce_only, NOT suppressed. (The earlier "trapped residual" behaviour was
    built on the false premise that the venue rejected these — it was our own
    client-side guard, hyperliquid.py:641.)
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
        min_notional_map={"ETH": 10.0},  # venue floor; closes are exempt reduce-only
    )
    row = orders.iloc[0]
    assert row["Order Status"] == "To be placed"
    assert bool(row["Executable"]), "a sub-$10 close must still be sent (reduce-only)"
    assert bool(row["reduce_only"]), "a flat-target close must carry the reduce_only flag"


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


def test_batch_submission_exception_still_alerts():
    """A broker-level exception from place_orders is the WORST case — nothing
    traded at all — so it must alert on the same path as a per-order rejection.

    Caught in review of #131: the exception handler returned early, past the new
    failure report, so a whole-batch failure counted total_failed and told nobody.
    """

    class _Exploding(_Broker):
        def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
            raise ConnectionError("venue unreachable")

    pipe = TradingPipeline()
    broker = _Exploding()
    orders = pd.DataFrame([_executable("PEPE", "Buy", 100.0, 1.0), _executable("ETH", "Sell", 0.0022, 1930.0)])

    report = pipe._execute_orders(
        broker=broker,
        orders_df=orders,
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )

    assert report["summary"]["total_failed"] == 2
    assert report.get("order_failures") == 2
    assert "venue unreachable" in report.get("order_failure_detail", "")
    assert any("ORDER(S) FAILED" in m for m in broker.notices), "a whole-batch submission failure must reach a human"


def test_reduce_only_flag_threads_to_broker():
    """A closing order's reduce_only flag must reach the broker so the venue
    exempts it from the $10 minimum."""

    class _CaptureBroker(_Broker):
        def __init__(self):
            super().__init__()
            self.seen: list[dict] = []

        def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
            self.seen = orders.to_dict("records")
            return pd.DataFrame(
                [
                    {"symbol": o["symbol"], "side": o["side"], "qty": o["qty"], "price": o["price"], "status": "FILLED"}
                    for _, o in orders.iterrows()
                ]
            )

    pipe = TradingPipeline()
    broker = _CaptureBroker()
    close = _executable("ETH", "Sell", 0.0022, 1930.0)
    close["reduce_only"] = True
    open_ = _executable("PEPE", "Buy", 100.0, 1.0)  # reduce_only absent -> False
    report = pipe._execute_orders(
        broker=broker,
        orders_df=pd.DataFrame([open_, close]),
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )
    assert report["summary"]["total_failed"] == 0
    by_sym = {o["symbol"]: o for o in broker.seen}
    assert by_sym["ETH"]["reduce_only"] is True, "the close must be sent reduce-only"
    assert not by_sym["PEPE"].get("reduce_only", False), "an open must not be reduce-only"


# ---------------------------------------------------------------------------
# 4. An unreadable balance is unknown, not zero
# ---------------------------------------------------------------------------
def test_enforce_dropping_every_order_still_alerts():
    """Enforce-mode intent capture dropping the whole batch is a total trading
    standstill — the exact case a human must hear about.

    Caught in review of #132: that early return also skipped the failure report.
    """
    pipe = TradingPipeline()
    broker = _Broker()
    report = {
        "executed_orders": [],
        "failed_orders": [],
        "summary": {"total_executed": 0, "total_partial": 0, "total_failed": 2, "total_value": 0.0, "total_cost": 0.0},
        "orders_details": [
            {"symbol": "ETH", "action": "sell", "status": "FAILED", "error": "intent capture failed"},
            {"symbol": "SOL", "action": "sell", "status": "FAILED", "error": "intent capture failed"},
        ],
        "api_errors": [],
    }

    pipe._report_order_failures(report, broker)

    assert report["order_failures"] == 2
    assert "intent capture failed" in report["order_failure_detail"]
    assert any("ORDER(S) FAILED" in m for m in broker.notices)


def test_spot_probe_failure_refuses_to_size_on_perps_residual():
    """A unified account holds its real collateral on the SPOT side. If that probe
    fails, the perps-side number is a residual, not a conservative estimate —
    sizing on it would silently downsize or flatten the book."""
    from quantbox.plugins.broker.hyperliquid import HyperliquidBroker

    broker = HyperliquidBroker.__new__(HyperliquidBroker)
    broker.telegram_token = ""
    broker.telegram_chat_id = ""

    class _SpotDown:
        def fetch_balance(self, params=None):
            if params and params.get("type") == "spot":
                raise RuntimeError("spot endpoint 503")
            # Perps side reports only a tiny residual.
            return {"USDC": {"total": 2.14, "free": 2.14, "used": 0.0}}

    broker._exchange = _SpotDown()

    with pytest.raises(BrokerExecutionError, match="unknown equity"):
        broker.get_balance()


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
