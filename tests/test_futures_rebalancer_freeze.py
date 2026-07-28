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

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Fill everything submitted.

        These tests are about the freeze / quiet-day CLASSIFICATION, which happens
        before submission, so most never reach here. The boundary test does, and it
        asserts a normal, silent trade — so this must actually succeed. Without it
        the call raised AttributeError, which the old code swallowed with no alert:
        the test passed only *because* whole-batch submission failures were silent
        (quantbox#87, caught in review of #131).
        """
        return pd.DataFrame(
            [
                {
                    "symbol": o["symbol"],
                    "side": o["side"],
                    "qty": o["qty"],
                    "price": o["price"],
                    "status": "FILLED",
                }
                for _, o in orders.iterrows()
            ]
        )


class _SkippingBroker(_FakeBroker):
    """Accepts every order, then SKIPS it — the live Kraken trapped-residual shape.

    Critically, the returned rows carry ONLY what a real broker returns
    (symbol/side/qty/price/status/error). No notional, no venue minimum: those
    are dropped at the submission boundary, which is the whole reason the
    broker-side freeze record has to recover them from the submitted intent.
    """

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "symbol": o["symbol"],
                    "side": o["side"],
                    "qty": o["qty"],
                    "price": o["price"],
                    "status": "SKIPPED",
                    "error": "below venue minimum",
                }
                for _, o in orders.iterrows()
            ]
        )


def _executable_sell_df() -> pd.DataFrame:
    """Orders that PASS the pre-submission filter and reach the broker."""
    return pd.DataFrame(
        [
            {
                "Asset": "ADA",
                "Action": "Sell",
                "Adjusted Quantity": 100.0,
                "Price": 0.12,
                "Notional Value": 12.0,
                "Min Notional": MIN_NOTIONAL,
                "Order Status": "To be placed",
                "Executable": True,
            }
        ]
    )


def test_broker_side_freeze_recovers_notional_from_intent():
    """Regression for the #143 review blocker.

    The broker-side freeze built its records from `orders_details`, which for a
    SKIPPED row carries only symbol/action/quantity/status. So it reported
    "SELL ADA $0.00" with no shortfall — in exactly the trapped-residual case the
    alert exists to explain. The original tests only covered the PRE-submission
    freeze site, so the gap shipped.
    """
    pipe = TradingPipeline()
    broker = _SkippingBroker()
    report = pipe._execute_orders(
        broker=broker,
        orders_df=_executable_sell_df(),
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )

    assert report.get("frozen") is True, "0 fills + a skipped close-out SELL must freeze"
    detail = report.get("frozen_orders")
    assert detail, "broker-side freeze produced no per-order detail"

    ada = next(r for r in detail if r["symbol"] == "ADA")
    assert ada["notional_usd"] == 12.0, f"notional lost at the broker boundary: {ada}"
    assert ada["min_notional_usd"] == MIN_NOTIONAL
    # The alert text must not read "$0.00" — that was the shipped bug.
    assert "$0.00" not in broker.messages[0]
    assert "ADA" in broker.messages[0]


def test_broker_side_freeze_recovers_notional_from_padded_broker_side():
    """Regression for the #144 review blocker (round 2).

    `orders_details` stores the broker's RAW side, so a whitespace-padded " SELL "
    (the #81 shape, already covered for freeze COUNTING) missed the intent key and
    put "$0.00" back in the alert — the exact bug the recovery exists to prevent.
    """

    class _PaddedSkippingBroker(_FakeBroker):
        def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "symbol": str(o["symbol"]),
                        "side": f" {str(o['side']).upper()} ",
                        "qty": float(o["qty"]),
                        "price": 0.0,
                        "status": " SKIPPED ",
                        "error": "below venue minimum",
                    }
                    for _, o in orders.iterrows()
                ]
            )

    pipe = TradingPipeline()
    broker = _PaddedSkippingBroker()
    report = pipe._execute_orders(
        broker=broker,
        orders_df=_executable_sell_df(),
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )

    assert report.get("frozen") is True
    ada = next(r for r in report["frozen_orders"] if r["symbol"] == "ADA")
    assert ada["notional_usd"] == 12.0, f"padded broker side lost the intent: {ada}"
    assert ada["min_notional_usd"] == MIN_NOTIONAL
    assert "$0.00" not in broker.messages[0]


def test_broker_side_freeze_keeps_duplicate_same_side_intents_distinct():
    """Regression for the #144 review blocker.

    The intent map was keyed only by (symbol, side), so two SELLs of the same
    symbol collapsed to the last one: a $12 trapped residual was reported as
    $1.20 (and its shortfall line vanished). Each frozen row must carry its OWN
    submitted notional.
    """
    pipe = TradingPipeline()
    broker = _SkippingBroker()
    orders = pd.concat([_executable_sell_df(), _executable_sell_df()], ignore_index=True)
    orders.loc[1, ["Adjusted Quantity", "Notional Value"]] = [10.0, 1.2]

    report = pipe._execute_orders(
        broker=broker,
        orders_df=orders,
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )

    assert report.get("frozen") is True
    notionals = [r["notional_usd"] for r in report["frozen_orders"] if r["symbol"] == "ADA"]
    assert sorted(notionals) == [1.2, 12.0], f"duplicate (symbol, side) intents collapsed: {notionals}"


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


def test_freeze_report_carries_per_order_detail():
    """The freeze must be EXPLAINABLE, not just counted.

    freeze_reasons is a status histogram ("Below min notional=4"). It says how
    many orders died and nothing about which, how large, or how far under the
    venue floor — so the first question on being paged ("what could not trade,
    and by how much?") always required opening the run log.
    """
    pipe = TradingPipeline()
    orders = _frozen_orders_df()
    orders["Min Notional"] = MIN_NOTIONAL
    report = pipe._execute_orders(
        broker=_FakeBroker(),
        orders_df=orders,
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )

    detail = report.get("frozen_orders")
    assert detail, "freeze produced no per-order detail"
    assert {r["symbol"] for r in detail} == {"ADA", "DOGE", "ETH", "kPEPE"}

    ada = next(r for r in detail if r["symbol"] == "ADA")
    assert ada["action"] == "SELL"
    assert ada["notional_usd"] == 2.90
    assert ada["min_notional_usd"] == MIN_NOTIONAL
    # The actionable number: how much the leg is short of being placeable.
    assert ada["shortfall_usd"] == MIN_NOTIONAL - 2.90


def test_freeze_alert_message_names_each_order():
    """The chat alert itself must carry the detail — the whole point is that a
    human reading the page does not have to go find the log."""
    pipe = TradingPipeline()
    broker = _FakeBroker()
    orders = _frozen_orders_df()
    orders["Min Notional"] = MIN_NOTIONAL
    pipe._execute_orders(
        broker=broker,
        orders_df=orders,
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )

    assert len(broker.messages) == 1
    msg = broker.messages[0]
    for symbol in ("ADA", "DOGE", "ETH", "kPEPE"):
        assert symbol in msg, f"{symbol} missing from freeze alert"
    assert "short $" in msg, "alert does not state the shortfall vs the venue minimum"


def test_freeze_detail_omits_shortfall_when_not_a_minimum_breach():
    """Not every suppression is a sub-minimum one (stale/NaN data also freezes).

    Reporting a shortfall for those would invent a number, so it must be absent
    rather than zero — 'no shortfall recorded' and 'short $0.00' read very
    differently at 6am.
    """
    pipe = TradingPipeline()
    orders = _frozen_orders_df()
    orders["Min Notional"] = 0.0  # venue minimum unknown
    report = pipe._execute_orders(
        broker=_FakeBroker(),
        orders_df=orders,
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )
    assert all("shortfall_usd" not in r for r in report["frozen_orders"])


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


class _FakeBrokerWithPositions(_FakeBroker):
    """Broker that also reports liquidatable positions (post dust-exclusion)."""

    def __init__(self, positions=None):
        super().__init__()
        self._positions = positions if positions is not None else {}

    def get_positions(self):
        return self._positions


def _quiet_day_orders_df() -> pd.DataFrame:
    """All-cash book on a weak-signal day: every ENTRY (buy) leg sub-min."""
    rows = [
        ("BTC", "Buy", 3.10, "Below min notional"),
        ("ETH", "Buy", 2.80, "Below min notional"),
        ("SOL", "Buy", 4.05, "Below min notional"),
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


def test_pipeline_quiet_day_all_cash_sub_min_targets():
    """All-cash + every target leg sub-min => QUIET DAY, no freeze alert."""
    pipe = TradingPipeline()
    broker = _FakeBrokerWithPositions(positions={})  # flat book
    report = pipe._execute_orders(
        broker=broker,
        orders_df=_quiet_day_orders_df(),
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )
    assert report.get("quiet_day") is True
    assert not report.get("frozen")
    assert broker.messages == []  # no loud freeze alert on a flat-trend day
    assert "Below min notional" in str(report.get("quiet_reasons"))


def test_pipeline_quiet_day_without_get_positions_fails_safe_to_freeze():
    """Issue #82: a broker that exposes NO get_positions cannot confirm the book
    is flat. All-buy suppression looks quiet on a long-only spot book, but on a
    futures book a suppressed BUY can be a short-close EXIT the has_suppressed_sell
    heuristic does not catch. Without the position probe we must NOT downgrade to
    quiet — fail safe to FROZEN and alert."""
    pipe = TradingPipeline()
    broker = _FakeBroker()  # no get_positions
    report = pipe._execute_orders(
        broker=broker,
        orders_df=_quiet_day_orders_df(),
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )
    assert report.get("frozen") is True
    assert not report.get("quiet_day")
    assert len(broker.messages) == 1
    assert "FROZEN" in broker.messages[0]


def test_pipeline_trapped_position_with_only_buy_orders_still_freezes():
    """Defensive: even if the suppressed legs are all buys, a broker reporting
    a liquidatable position means the book is NOT flat => FREEZE, not quiet."""
    pipe = TradingPipeline()
    broker = _FakeBrokerWithPositions(positions={"ADA": 1000.0})
    report = pipe._execute_orders(
        broker=broker,
        orders_df=_quiet_day_orders_df(),
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )
    assert report.get("frozen") is True
    assert not report.get("quiet_day")
    assert len(broker.messages) == 1
    assert "FROZEN" in broker.messages[0]


def test_pipeline_quiet_day_boundary_one_leg_executable():
    """One leg above min_notional => a normal trade, neither quiet nor frozen."""
    pipe = TradingPipeline()
    broker = _FakeBrokerWithPositions(positions={})
    orders_df = _quiet_day_orders_df()
    # Promote one leg to executable (above min, to-be-placed).
    orders_df.loc[0, "Adjusted Quantity"] = 5.0
    orders_df.loc[0, "Notional Value"] = 25.0
    orders_df.loc[0, "Order Status"] = "To be placed"
    orders_df.loc[0, "Executable"] = True
    report = pipe._execute_orders(
        broker=broker,
        orders_df=orders_df,
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )
    assert not report.get("quiet_day")
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


# ---------------------------------------------------------------------------
# Issue #81: all-skipped-at-broker close-out SELL must still trip the freeze
# ---------------------------------------------------------------------------


class _SkipAllBroker(_FakeBroker):
    """Broker that accepts the batch but SKIPS every order (sub-exchange-min).

    Mirrors the post-dust-fix KrakenBroker behaviour where a sub-minimum order
    is a clean SKIP (status SKIPPED) rather than a hard FAILED.
    """

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        cols = ["symbol", "side", "qty", "price", "order_id", "status", "error"]
        rows = []
        for _, o in orders.iterrows():
            rows.append(
                {
                    "symbol": str(o["symbol"]),
                    "side": str(o["side"]).lower(),
                    "qty": float(o["qty"]),
                    "price": 0.0,
                    "order_id": None,
                    "status": "SKIPPED",
                    "error": "below exchange minimum (skipped)",
                }
            )
        return pd.DataFrame(rows, columns=cols)


def _executable_closeout_sell_df() -> pd.DataFrame:
    """An *executable* close-out SELL (min-notional-exempt) plus a normal buy —
    both of which the broker will skip sub-minimum."""
    rows = [
        ("ADA", "Sell", 0.5, "To be placed", True),
        ("BTC", "Buy", 0.0001, "To be placed", True),
    ]
    return pd.DataFrame(
        [
            {
                "Asset": a,
                "Action": act,
                "Adjusted Quantity": qty,
                "Price": 1.0,
                "Notional Value": 0.5,
                "Order Status": status,
                "Executable": ex,
            }
            for (a, act, qty, status, ex) in rows
        ]
    )


def test_pipeline_all_skipped_closeout_sell_trips_freeze():
    """Issue #81: an executable close-out SELL the broker returns as SKIPPED
    (zero fills, no hard failures) is a trapped residual — it must trip the
    freeze/alert path, not read as a healthy run."""
    pipe = TradingPipeline()
    broker = _SkipAllBroker()
    report = pipe._execute_orders(
        broker=broker,
        orders_df=_executable_closeout_sell_df(),
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )
    assert report.get("frozen") is True
    assert report["summary"]["total_executed"] == 0
    assert report["summary"]["total_failed"] == 0
    assert report["freeze_reasons"]["skipped_sell"] >= 1
    assert len(broker.messages) == 1
    assert "FROZEN" in broker.messages[0]


def test_pipeline_all_skipped_padded_side_status_still_trips_freeze():
    """Issue #81 hardening: a broker that returns a whitespace-padded side/status
    (e.g. "SELL " / "SKIPPED ") must NOT evade the SKIPPED / close-out-SELL freeze
    counter. Before the .strip() normalisation the padded rows compared unequal to
    'sell'/'SKIPPED' and the all-skipped close-out SELL read as a clean run."""

    class _PaddedSkipAllBroker(_FakeBroker):
        def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
            cols = ["symbol", "side", "qty", "price", "order_id", "status", "error"]
            rows = [
                {
                    "symbol": str(o["symbol"]),
                    "side": f" {str(o['side']).upper()} ",  # padded + upper-cased
                    "qty": float(o["qty"]),
                    "price": 0.0,
                    "order_id": None,
                    "status": " SKIPPED ",  # padded
                    "error": "below exchange minimum (skipped)",
                }
                for _, o in orders.iterrows()
            ]
            return pd.DataFrame(rows, columns=cols)

    pipe = TradingPipeline()
    broker = _PaddedSkipAllBroker()
    report = pipe._execute_orders(
        broker=broker,
        orders_df=_executable_closeout_sell_df(),
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )
    assert report.get("frozen") is True
    assert report["summary"]["total_executed"] == 0
    assert report["freeze_reasons"]["skipped_sell"] >= 1
    assert len(broker.messages) == 1
    assert "FROZEN" in broker.messages[0]


def test_pipeline_all_skipped_buys_only_is_not_a_freeze():
    """Guard against false positives: an all-skipped batch with NO close-out
    SELL (entries only) is a quiet all-cash outcome, not a trapped book."""
    pipe = TradingPipeline()
    broker = _SkipAllBroker()
    orders_df = pd.DataFrame(
        [
            {
                "Asset": "BTC",
                "Action": "Buy",
                "Adjusted Quantity": 0.0001,
                "Price": 1.0,
                "Notional Value": 0.5,
                "Order Status": "To be placed",
                "Executable": True,
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


def test_pipeline_skipped_sell_with_a_real_fill_is_not_frozen():
    """A skipped SELL alongside a genuine fill is NOT a frozen book — there was
    real execution, so the dead-man must stay quiet."""

    class _PartlyFillBroker(_FakeBroker):
        def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
            cols = ["symbol", "side", "qty", "price", "order_id", "status", "error"]
            rows = []
            for _, o in orders.iterrows():
                side = str(o["side"]).lower()
                if side == "sell":
                    rows.append(
                        {
                            "symbol": o["symbol"],
                            "side": side,
                            "qty": float(o["qty"]),
                            "price": 0.0,
                            "order_id": None,
                            "status": "SKIPPED",
                            "error": "dust",
                        }
                    )
                else:
                    rows.append(
                        {
                            "symbol": o["symbol"],
                            "side": side,
                            "qty": float(o["qty"]),
                            "price": 100.0,
                            "order_id": "1",
                            "status": "FILLED",
                            "error": "",
                        }
                    )
            return pd.DataFrame(rows, columns=cols)

    pipe = TradingPipeline()
    broker = _PartlyFillBroker()
    report = pipe._execute_orders(
        broker=broker,
        orders_df=_executable_closeout_sell_df(),
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )
    assert not report.get("frozen")
    assert report["summary"]["total_executed"] == 1
    assert broker.messages == []


def test_skipped_close_out_sell_freezes_even_with_a_concurrent_failure():
    """The trapped-residual freeze (skipped close-out SELL, 0 fills) must fire even
    when another order ALSO hard-failed — total_failed>0 must NOT mask the residual
    signal behind the generic FAILED alert (#85 review)."""

    class _SkipAndFailBroker(_FakeBroker):
        def place_orders(self, orders):  # noqa: ANN001 - test double
            return pd.DataFrame(
                [
                    {
                        "symbol": "DOGE",
                        "side": "sell",
                        "qty": 0.0,
                        "price": 0.1,
                        "order_id": "",
                        "status": "SKIPPED",
                        "error": "below exchange minimum",
                    },
                    {
                        "symbol": "ETH",
                        "side": "buy",
                        "qty": 0.0,
                        "price": 2000.0,
                        "order_id": "",
                        "status": "FAILED",
                        "error": "placement failed",
                    },
                ]
            )

    orders = pd.DataFrame(
        [
            {
                "Asset": "DOGE",
                "Action": "Sell",
                "Adjusted Quantity": 1.0,
                "Price": 0.1,
                "Notional Value": 600.0,
                "Order Status": "To be placed",
                "Executable": True,
            },
            {
                "Asset": "ETH",
                "Action": "Buy",
                "Adjusted Quantity": 0.3,
                "Price": 2000.0,
                "Notional Value": 600.0,
                "Order Status": "To be placed",
                "Executable": True,
            },
        ]
    )
    pipe = TradingPipeline()
    broker = _SkipAndFailBroker()
    report = pipe._execute_orders(
        broker=broker,
        orders_df=orders,
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )
    assert report.get("frozen") is True
    assert report["summary"]["total_executed"] == 0
    assert report["summary"]["total_failed"] == 1
    assert report["freeze_reasons"]["skipped_sell"] >= 1
    assert report["freeze_reasons"]["failed"] == 1
    assert any("FROZEN" in m for m in broker.messages)
