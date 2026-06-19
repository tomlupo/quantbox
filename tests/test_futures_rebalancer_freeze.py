"""Regression tests for the 2026-05-26 live rebalancer freeze.

Root cause: on the ~$85 Hyperliquid book the per-asset target legs are all
$1–4, below the default ``min_notional`` floor of $10. Every order was marked
``Executable=False`` ("Below min notional"), so ``place_orders`` was never
called and the pipeline recorded ``n_orders>0, n_fills=0, n_failed=0`` and
exited 0 with no alert. The stale ETH/SOL longs the strategy wanted to flatten
were themselves blocked by the same floor, so the book could never recover.

These tests pin the two fixes:
  1. Position-flattening (closing) orders bypass the min_notional / min_trade
     bands — you must always be able to exit a position.
  2. When a rebalance is intended but EVERY order is suppressed, the pipeline
     flags ``frozen`` and fires a loud alert (dead-man), instead of returning
     an empty report silently.
Plus a NaN-target guard so a missing-candle glitch fails loudly.
"""

from __future__ import annotations

import pandas as pd

from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline
from quantbox.plugins.rebalancing.futures_rebalancer import FuturesRebalancer

MIN_TRADE = 0.01
MIN_NOTIONAL = 10.0


def _rebal_row(
    asset: str,
    *,
    action: str,
    delta_qty: float,
    price: float,
    weight_delta: float,
    target_weight: float,
    current_qty: float,
) -> dict:
    return {
        "Asset": asset,
        "Trade Action": action,
        "Delta Quantity": delta_qty,
        "Price": price,
        "Weight Delta": weight_delta,
        "Target Weight": target_weight,
        "Current Quantity": current_qty,
    }


def _make_orders(rows: list[dict]) -> pd.DataFrame:
    reb = FuturesRebalancer()
    df = pd.DataFrame(rows)
    return reb._create_executable_orders(df, min_trade_size=MIN_TRADE, min_notional=MIN_NOTIONAL)


# ---------------------------------------------------------------------------
# Rebalancer: order classification
# ---------------------------------------------------------------------------


def test_closing_position_exempt_from_min_notional():
    """The exact live trap: a held SOL long, target flat, $2.14 notional.

    Before the fix this was 'Below min notional' / not executable, so the
    stale long could never be closed. It must now be executable.
    """
    orders = _make_orders(
        [
            _rebal_row(
                "SOL",
                action="Sell",
                delta_qty=-0.03,
                price=71.30,  # notional ~$2.14, well under $10
                weight_delta=-0.025,  # > min_trade_size, so it reaches the notional check
                target_weight=0.0,
                current_qty=0.03,
            )
        ]
    )
    row = orders.iloc[0]
    assert row["Order Status"] == "To be placed", row["Order Status"]
    assert bool(row["Executable"]) is True
    assert row["Adjusted Quantity"] > 0


def test_open_below_min_notional_still_blocked():
    """A NEW sub-$10 position (no existing holding) stays blocked — the floor
    still protects against opening dust positions."""
    orders = _make_orders(
        [
            _rebal_row(
                "kPEPE",
                action="Sell",
                delta_qty=-1496.5,
                price=0.002918,  # notional ~$4.37
                weight_delta=-0.051,
                target_weight=-0.051,
                current_qty=0.0,  # opening, not closing
            )
        ]
    )
    row = orders.iloc[0]
    assert row["Order Status"] == "Below min notional", row["Order Status"]
    assert bool(row["Executable"]) is False


def test_tiny_close_below_min_trade_size_still_exempt():
    """Regression: a close whose weight delta is below min_trade_size was
    already exempt before the fix; keep it so."""
    orders = _make_orders(
        [
            _rebal_row(
                "ETH",
                action="Sell",
                delta_qty=-0.0001,
                price=1720.0,  # notional ~$0.17
                weight_delta=-0.005,  # < min_trade_size
                target_weight=0.0,
                current_qty=0.0001,
            )
        ]
    )
    row = orders.iloc[0]
    assert row["Order Status"] == "To be placed"
    assert bool(row["Executable"]) is True


def test_nan_price_flagged_not_silent():
    """A NaN price (e.g. Hyperliquid missing-candle glitch) must be flagged
    loudly, not vanish as a silent no-op."""
    orders = _make_orders(
        [
            _rebal_row(
                "DOGE",
                action="Sell",
                delta_qty=-40.0,
                price=float("nan"),
                weight_delta=-0.04,
                target_weight=-0.04,
                current_qty=0.0,
            )
        ]
    )
    row = orders.iloc[0]
    assert row["Order Status"] == "Invalid (NaN)"
    assert bool(row["Executable"]) is False


def test_nan_delta_flagged_not_silent():
    orders = _make_orders(
        [
            _rebal_row(
                "ADA",
                action="Sell",
                delta_qty=float("nan"),
                price=0.18,
                weight_delta=-0.03,
                target_weight=-0.03,
                current_qty=0.0,
            )
        ]
    )
    assert orders.iloc[0]["Order Status"] == "Invalid (NaN)"
    assert bool(orders.iloc[0]["Executable"]) is False


def test_none_price_is_zero_price_not_nan():
    """``None`` price (no quote yet) must keep its own 'Zero price' handling,
    not be mistaken for NaN."""
    orders = _make_orders(
        [
            _rebal_row(
                "XRP",
                action="Sell",
                delta_qty=-10.0,
                price=None,
                weight_delta=-0.03,
                target_weight=-0.03,
                current_qty=0.0,
            )
        ]
    )
    assert orders.iloc[0]["Order Status"] == "Zero price"


# ---------------------------------------------------------------------------
# Pipeline: dead-man freeze detection + alert
# ---------------------------------------------------------------------------


class _FakeBroker:
    def __init__(self):
        self.messages: list[str] = []

    def notify(self, message: str) -> bool:
        self.messages.append(message)
        return True


def _frozen_orders_df() -> pd.DataFrame:
    """Mirror the live 2026-06-15 orders.parquet: every leg sub-$10, none
    executable."""
    rows = [
        ("ADA", "Sell", 2.90, "Below min notional"),
        ("DOGE", "Sell", 3.63, "Below min notional"),
        ("ETH", "Sell", 3.78, "Below min notional"),
        ("kPEPE", "Sell", 4.37, "Below min notional"),
    ]
    return pd.DataFrame(
        [
            {
                "Asset": a,
                "Action": act,
                "Adjusted Quantity": 0.0,
                "Price": 1.0,
                "Notional Value": notion,
                "Order Status": status,
                "Executable": False,
            }
            for (a, act, notion, status) in rows
        ]
    )


def test_pipeline_flags_freeze_and_alerts():
    pipe = TradingPipeline()
    broker = _FakeBroker()
    report = pipe._execute_orders(
        broker=broker,
        orders_df=_frozen_orders_df(),
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )
    assert report.get("frozen") is True
    assert report["summary"]["total_executed"] == 0
    assert report["summary"]["total_failed"] == 0
    assert len(broker.messages) == 1
    assert "FROZEN" in broker.messages[0]
    assert "Below min notional" in str(report.get("freeze_reasons"))


def test_pipeline_quiet_day_not_flagged():
    """A genuinely quiet day (only zero-delta rows) is NOT a freeze and must
    not alert."""
    pipe = TradingPipeline()
    broker = _FakeBroker()
    orders_df = pd.DataFrame(
        [
            {
                "Asset": "BTC",
                "Action": "Hold",
                "Adjusted Quantity": 0.0,
                "Price": 1.0,
                "Notional Value": 0.0,
                "Order Status": "Zero delta",
                "Executable": False,
            }
        ]
    )
    report = pipe._execute_orders(
        broker=broker,
        orders_df=orders_df,
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )
    assert not report.get("frozen")
    assert broker.messages == []


def test_pipeline_freeze_survives_broker_without_notify():
    """Freeze flag is still set even if the broker can't alert."""

    class _Mute:
        pass

    pipe = TradingPipeline()
    report = pipe._execute_orders(
        broker=_Mute(),
        orders_df=_frozen_orders_df(),
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )
    assert report.get("frozen") is True
