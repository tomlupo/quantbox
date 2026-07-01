"""Tier-2 exception INPUTS produced by the trading run (issue #62).

quantbox-live's notifier built three Tier-2 alerts that were DORMANT because the
run did not yet produce their input signals: data-staleness (feed older than Nx
bar interval — the HL 429 feed-gap class), API-error (a survived broker/data API
failure), and pipeline-failure (a run crash). These tests pin the lib side: the
run now captures feed age and API errors so the notifier's BookContext can fire.
"""

from __future__ import annotations

import pandas as pd

from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline, _compute_data_age

# ---------------------------------------------------------------------------
# Data-staleness input: _compute_data_age
# ---------------------------------------------------------------------------


def _daily_prices(last_day: str, n: int = 10) -> pd.DataFrame:
    idx = pd.date_range(end=last_day, periods=n, freq="D")
    return pd.DataFrame({"BTC": range(n)}, index=idx)


def test_fresh_daily_feed_has_small_age_and_daily_interval():
    prices = _daily_prices("2026-06-30")
    age, interval = _compute_data_age(prices, "2026-06-30")
    assert age == 0.0
    assert interval == 86400.0  # 1 day


def test_stale_feed_age_reflects_gap():
    # Latest bar is 2026-06-25 but the run is as-of 2026-06-30 => 5-day gap.
    prices = _daily_prices("2026-06-25")
    age, interval = _compute_data_age(prices, "2026-06-30")
    assert age == 5 * 86400.0
    assert interval == 86400.0
    # 5 days > 2x the 1-day bar interval => stale under the default factor.
    assert age > 2.0 * interval


def test_empty_feed_returns_none():
    age, interval = _compute_data_age(pd.DataFrame(), "2026-06-30")
    assert age is None and interval is None


def test_non_datetime_index_returns_none():
    df = pd.DataFrame({"BTC": [1, 2, 3]})  # default RangeIndex
    age, interval = _compute_data_age(df, "2026-06-30")
    assert age is None and interval is None


def test_unparseable_asof_returns_none():
    prices = _daily_prices("2026-06-30")
    age, interval = _compute_data_age(prices, "not-a-date")
    assert age is None


def test_tz_aware_feed_does_not_raise():
    idx = pd.date_range(end="2026-06-30", periods=5, freq="D", tz="UTC")
    prices = pd.DataFrame({"BTC": range(5)}, index=idx)
    age, interval = _compute_data_age(prices, "2026-06-30")
    assert age == 0.0
    assert interval == 86400.0


def test_hourly_feed_interval():
    idx = pd.date_range(end="2026-06-30 12:00", periods=6, freq="h")
    prices = pd.DataFrame({"BTC": range(6)}, index=idx)
    _age, interval = _compute_data_age(prices, "2026-06-30 12:00")
    assert interval == 3600.0


# ---------------------------------------------------------------------------
# API-error input: _execute_orders captures a broker failure as an API error
# ---------------------------------------------------------------------------


class _RaisingBroker:
    def place_orders(self, orders):  # noqa: ANN001 - test double
        raise RuntimeError("429 Too Many Requests")


def _executable_order_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Asset": "BTC",
                "Action": "Buy",
                "Adjusted Quantity": 0.01,
                "Price": 60000.0,
                "Notional Value": 600.0,
                "Order Status": "To be placed",
                "Executable": True,
            }
        ]
    )


def test_broker_api_error_captured_as_exception_input():
    pipe = TradingPipeline()
    report = pipe._execute_orders(
        broker=_RaisingBroker(),
        orders_df=_executable_order_df(),
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )
    assert report["api_errors"], "broker API error must be captured, not swallowed"
    assert report["api_errors"][0]["stage"] == "place_orders"
    assert "429" in report["api_errors"][0]["error"]
    # The order itself is still reported FAILED (existing behaviour preserved).
    assert report["summary"]["total_failed"] == 1


def test_no_api_error_on_clean_execution():
    class _FillBroker:
        def place_orders(self, orders):  # noqa: ANN001 - test double
            cols = ["symbol", "side", "qty", "price", "order_id", "status", "error"]
            rows = [
                {
                    "symbol": o["symbol"],
                    "side": str(o["side"]).lower(),
                    "qty": float(o["qty"]),
                    "price": 60000.0,
                    "order_id": "1",
                    "status": "FILLED",
                    "error": "",
                }
                for _, o in orders.iterrows()
            ]
            return pd.DataFrame(rows, columns=cols)

    pipe = TradingPipeline()
    report = pipe._execute_orders(
        broker=_FillBroker(),
        orders_df=_executable_order_df(),
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )
    assert report["api_errors"] == []
    assert report["summary"]["total_executed"] == 1


def test_object_index_with_numeric_labels_returns_none():
    """An object-dtype index holding numeric labels must NOT coerce to 1970-epoch
    nanoseconds and fabricate a huge false staleness (the half-fix gap)."""
    import numpy as np

    df = pd.DataFrame({"BTC": [1.0, 2.0, 3.0]}, index=pd.Index([1, 2, 3], dtype=object))
    assert _compute_data_age(df, "2026-06-30") == (None, None)
    df2 = pd.DataFrame({"BTC": [1.0, 2.0]}, index=pd.Index([np.int64(1), np.int64(2)], dtype=object))
    assert _compute_data_age(df2, "2026-06-30") == (None, None)


def test_partial_fill_surfaced_as_total_partial():
    """A partial fill counts as executed (keeps freeze logic honest) AND is surfaced
    as total_partial so the notifier can raise an incomplete-fill exception instead
    of reading the run as a clean success."""

    class _PartialBroker:
        def place_orders(self, orders):  # noqa: ANN001 - test double
            return pd.DataFrame(
                [
                    {
                        "symbol": str(o["symbol"]),
                        "side": str(o["side"]).lower(),
                        "qty": float(o["qty"]) / 2.0,
                        "price": 60000.0,
                        "order_id": "1",
                        "status": "PARTIAL",
                        "error": "partial fill: half/full",
                    }
                    for _, o in orders.iterrows()
                ]
            )

    pipe = TradingPipeline()
    report = pipe._execute_orders(
        broker=_PartialBroker(),
        orders_df=_executable_order_df(),
        stable_coin="USDC",
        trading_enabled=True,
        mode="live",
    )
    assert report["summary"]["total_executed"] == 1
    assert report["summary"]["total_partial"] == 1
