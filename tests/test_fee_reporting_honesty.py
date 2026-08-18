"""#92 — an unmeasured cost is UNKNOWN, never a fabricated zero.

Every daily report showed `Fees (cumulative) $0.00` for 165 days while the
carver-HL book traded ~$2,080 notional on an ~$85 account — roughly 24x equity
turnover. Not zero cost: a reporting blind spot. Two causes, both pinned here:

  1. `_cumulative_fees` exists only on the SIM brokers, so the pipeline's
     `hasattr` guard silently yielded 0.0 for every LIVE run.
  2. `fetch_fills` fetched the venue's fee and then dropped it on the floor.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantbox.plugins.broker._fills import trade_fee, trade_fee_currency

# ---------------------------------------------------------------------------
# The extractor: absence is None, never 0.0
# ---------------------------------------------------------------------------


def test_absent_fee_is_unknown_not_free():
    """The whole bug in one assertion."""
    assert trade_fee({"amount": 1.0, "price": 100.0}) is None
    assert trade_fee({"fee": None}) is None
    assert trade_fee({}) is None
    assert trade_fee(None) is None


def test_a_reported_zero_fee_is_kept_as_zero():
    """A venue that explicitly says 'no fee' is believed — that is data."""
    assert trade_fee({"fee": {"cost": 0.0, "currency": "USDC"}}) == 0.0


def test_single_fee_mapping():
    assert trade_fee({"fee": {"cost": 0.37, "currency": "USDC"}}) == 0.37
    assert trade_fee_currency({"fee": {"cost": 0.37, "currency": "USDC"}}) == "USDC"


def test_fees_list_is_summed():
    """Some venues split maker/taker or charge in several currencies."""
    trade = {"fees": [{"cost": 0.10, "currency": "USDC"}, {"cost": 0.05, "currency": "USDC"}]}
    assert trade_fee(trade) == pytest.approx(0.15)
    assert trade_fee_currency(trade) == "USDC"


def test_unparseable_and_nan_fees_are_unknown():
    assert trade_fee({"fee": {"cost": "not-a-number"}}) is None
    assert trade_fee({"fee": {"cost": float("nan")}}) is None


def test_partial_fees_list_sums_what_is_known():
    trade = {"fees": [{"cost": 0.10}, {"cost": None}]}
    assert trade_fee(trade) == 0.10


# ---------------------------------------------------------------------------
# The pipeline: unmeasured stays null
# ---------------------------------------------------------------------------


def test_artifact_reports_null_not_zero_when_unmeasured():
    """A live broker has no `_cumulative_fees`; the artifact must say null."""
    from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline

    payload = TradingPipeline._build_artifact_payload(
        TradingPipeline(),
        rebalancing_df=pd.DataFrame(),
        orders_df=pd.DataFrame(),
        execution_report={},
        final_weights={},
        total_value=1000.0,
        mode="live",
        cumulative_fees=None,
    )
    assert payload["trading_costs"]["cumulative_fees"] is None, "an unmeasured fee must be null, not a fabricated 0.0"


def test_artifact_reports_a_measured_fee():
    from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline

    payload = TradingPipeline._build_artifact_payload(
        TradingPipeline(),
        rebalancing_df=pd.DataFrame(),
        orders_df=pd.DataFrame(),
        execution_report={},
        final_weights={},
        total_value=1000.0,
        mode="live",
        cumulative_fees=1.2345678,
    )
    assert payload["trading_costs"]["cumulative_fees"] == 1.2346
