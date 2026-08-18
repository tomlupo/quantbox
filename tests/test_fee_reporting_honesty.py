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


def test_fees_list_is_summed_when_the_currency_agrees():
    """Some venues split a single trade's commission into maker/taker legs."""
    trade = {"fees": [{"cost": 0.10, "currency": "USDC"}, {"cost": 0.05, "currency": "USDC"}]}
    assert trade_fee(trade) == pytest.approx(0.15)
    assert trade_fee_currency(trade) == "USDC"


def test_mixed_currency_fees_are_unknown_not_summed():
    """0.10 USDT + 0.002 BNB is not 0.102 of anything.

    Real case: Binance futures with the BNB fee discount charges commission in
    BNB on a USDT-quoted trade. Summing and labelling the result with the first
    currency would be a fabricated number wearing a plausible unit — the exact
    sin this module exists to prevent.
    """
    trade = {"fees": [{"cost": 0.10, "currency": "USDT"}, {"cost": 0.002, "currency": "BNB"}]}
    assert trade_fee(trade) is None


def test_fees_list_without_currencies_is_still_summed():
    """No currency stated on any entry — nothing to disagree about."""
    assert trade_fee({"fees": [{"cost": 0.10}, {"cost": 0.05}]}) == pytest.approx(0.15)


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


# ---------------------------------------------------------------------------
# The brokers: the fee must survive being put into a DataFrame
# ---------------------------------------------------------------------------


def test_every_live_broker_frames_the_fee_columns():
    """The seam that broke: Kraken built the fee then dropped it.

    `pd.DataFrame(rows, columns=cols)` SELECTS — a key absent from `cols` is
    discarded with no error, so the fee was fetched, parsed and thrown away on
    one of the three brokers while the suite stayed green. Nothing in this repo
    consumes fetch_fills, so there was no symptom to notice locally.

    Asserts against the SOURCE column lists rather than a live call, since these
    brokers need credentials to run.
    """
    import inspect

    from quantbox.plugins.broker import binance_futures, hyperliquid, kraken

    for mod in (hyperliquid, binance_futures, kraken):
        src = inspect.getsource(mod.__dict__[[n for n in dir(mod) if n.endswith("Broker")][0]].fetch_fills)
        assert '"fee": trade_fee(t)' in src, f"{mod.__name__}: fee not captured"
        # Any explicit column list in this method must carry the fee columns.
        for literal in ("columns=cols", "cols = ["):
            if literal in src and literal == "cols = [":
                assert '"fee", "fee_currency"' in src, f"{mod.__name__}: builds a fee key but its column list drops it"


def test_funding_charge_is_null_when_unmeasured():
    """The same fabricated zero, one field over — #92 MAJOR 2."""
    from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline

    payload = TradingPipeline._build_artifact_payload(
        TradingPipeline(),
        rebalancing_df=pd.DataFrame(),
        orders_df=pd.DataFrame(),
        execution_report={},
        final_weights={},
        total_value=1000.0,
        mode="live",
        funding_charge=None,
        cumulative_fees=None,
    )
    costs = payload["trading_costs"]
    assert costs["funding_charge"] is None, "unmeasured funding must be null, not $0.00"
    assert costs["cumulative_fees"] is None
