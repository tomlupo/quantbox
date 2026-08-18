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


def _column_lists(func) -> list[list[str]]:
    """Every literal column list in a function: `columns=[...]` and `cols = [...]`.

    Parsed, not grepped — a source-substring check is blind to formatting and
    to which of the two styles a broker uses, which is exactly how the first
    version of this test missed two of the three brokers.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    out: list[list[str]] = []

    def literal(node):
        if isinstance(node, ast.List) and all(isinstance(e, ast.Constant) for e in node.elts):
            return [e.value for e in node.elts]
        return None

    for node in ast.walk(tree):
        # `pd.DataFrame(..., columns=[...])`
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "columns" and (lit := literal(kw.value)):
                    out.append(lit)
        # `cols = [...]` — by target name, so an unrelated `rows = []` is not
        # mistaken for a column list.
        if isinstance(node, ast.Assign) and (lit := literal(node.value)):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any("col" in n.lower() for n in names):
                out.append(lit)
    return out


def test_every_live_broker_frames_the_fee_columns():
    """The seam that broke: Kraken built the fee then dropped it.

    `pd.DataFrame(rows, columns=cols)` SELECTS — a key absent from the column
    list is discarded with no error, so the fee was fetched, parsed and thrown
    away on one broker while the suite stayed green. Nothing in this repo
    consumes fetch_fills, so there was no symptom to notice locally.

    EVERY literal column list in the method must carry the fee columns, not just
    the one style a given broker happens to use: Kraken assigns `cols = [...]`
    and reuses it, while Hyperliquid and Binance write `columns=[...]` inline on
    their empty-frame paths. The first version of this test only inspected the
    former, so a drop in either of the latter would still have passed — the same
    blind spot as the bug it was written to catch.
    """
    import inspect

    from quantbox.plugins.broker import binance_futures, hyperliquid, kraken

    for mod in (hyperliquid, binance_futures, kraken):
        broker = mod.__dict__[[n for n in dir(mod) if n.endswith("Broker")][0]]
        src = inspect.getsource(broker.fetch_fills)
        assert '"fee": trade_fee(t)' in src, f"{mod.__name__}: fee not captured"

        lists = _column_lists(broker.fetch_fills)
        assert lists, f"{mod.__name__}: no literal column list found — has the framing changed?"
        for cols in lists:
            assert "fee" in cols and "fee_currency" in cols, (
                f"{mod.__name__}: a column list {cols} drops the fee it just built"
            )


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


def test_stated_and_unstated_currencies_do_not_mix():
    """0.10 USDT + 0.002 <unstated> is not 0.102 USDT.

    Narrower than the mixed case but the identical error, with the second unit
    hidden rather than visible. Sum only when the entries are homogeneous in
    what they state.
    """
    assert trade_fee({"fees": [{"cost": 0.10, "currency": "USDT"}, {"cost": 0.002}]}) is None


def test_an_unknown_fee_is_never_given_a_currency_label():
    """A currency on an unmeasured fee implies it was denominated in it."""
    mixed = {"fees": [{"cost": 0.10, "currency": "USDT"}, {"cost": 0.002, "currency": "BNB"}]}
    assert trade_fee(mixed) is None
    assert trade_fee_currency(mixed) is None, "labelled an unknown fee with a currency"


def test_fees_this_run_is_null_when_a_fill_reports_no_fee():
    """The THIRD fabricated zero — no live broker's place_orders emits a fee."""
    from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline

    payload = TradingPipeline._build_artifact_payload(
        TradingPipeline(),
        rebalancing_df=pd.DataFrame(),
        orders_df=pd.DataFrame(),
        # One executed order whose fee the broker never reported.
        execution_report={"orders_details": [{"status": "FILLED", "symbol": "BTC", "fee": None}]},
        final_weights={},
        total_value=1000.0,
        mode="live",
    )
    assert payload["trading_costs"]["fees_this_run"] is None, "an unreported fill fee must not sum to a confident $0.00"


def test_fees_this_run_sums_when_every_fee_is_known():
    from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline

    payload = TradingPipeline._build_artifact_payload(
        TradingPipeline(),
        rebalancing_df=pd.DataFrame(),
        orders_df=pd.DataFrame(),
        execution_report={
            "orders_details": [
                {"status": "FILLED", "symbol": "BTC", "fee": 0.25},
                {"status": "PARTIAL", "symbol": "ETH", "fee": 0.10},
            ]
        },
        final_weights={},
        total_value=1000.0,
        mode="live",
    )
    assert payload["trading_costs"]["fees_this_run"] == pytest.approx(0.35)
