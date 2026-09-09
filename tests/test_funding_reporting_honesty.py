"""#92, funding half — an unfetched cost is UNKNOWN, never a fabricated zero.

The fee half of #92 was fetched from the venue and then dropped on the floor.
The funding half was worse: no broker ever asked the venue for its funding
history at all, so ``funding_charge`` fell through to a default and 165 daily
reports said ``Funding charge $0.00`` on a perps book that is charged funding
every hour on its full notional.

These tests pin the three places the zero could come back:

  1. the summer (``_funding.net_funding``) — absence and ambiguity are None;
  2. the live broker (``HyperliquidBroker.fetch_funding_payments``) — every
     failure path returns None;
  3. the pipeline's source selection (``_resolve_funding_charge``) — a broker
     with no funding source yields None, which the report renders UNKNOWN.
"""

from __future__ import annotations

import pytest

from quantbox.plugins.broker._funding import funding_payment, net_funding

# ---------------------------------------------------------------------------
# The summer: absence is None, an answered "nothing happened" is 0.0
# ---------------------------------------------------------------------------


def test_unfetchable_funding_is_unknown_not_free():
    """The whole bug in one assertion: None in, None out — never 0.0."""
    assert net_funding(None) is None


def test_an_empty_window_is_a_measured_zero():
    """The venue was asked and answered 'no funding events'. That is data."""
    assert net_funding([]) == 0.0


def test_entry_without_an_amount_poisons_the_whole_total():
    """A sum missing an unknown number of its terms is a fabricated number."""
    assert net_funding([{"amount": -1.5, "code": "USDC"}, {"code": "USDC"}]) is None
    assert net_funding([{"amount": "not-a-number"}]) is None
    assert funding_payment({}) is None
    assert funding_payment(None) is None


def test_mixed_currency_funding_is_unknown_not_summed():
    """-3.6 USDC + -0.002 BNB is not -3.602 of anything (mirrors trade_fee)."""
    entries = [{"amount": -3.6, "code": "USDC"}, {"amount": -0.002, "code": "BNB"}]
    assert net_funding(entries) is None


def test_payments_and_receipts_net_out_with_the_venue_sign():
    """Both documented Hyperliquid examples, in one net.

    long 49.1477 ETH at fundingRate +0.0000417 -> the long PAYS -> usdc is
    negative; short 7375.9 SOL at +0.00004381 -> the short RECEIVES -> usdc is
    positive. ccxt passes ``delta.usdc`` through un-negated, so a negative total
    means the book paid. A sign flip here would report a cost as income.
    """
    entries = [
        {"amount": -3.625312, "code": "USDC"},
        {"amount": 75.635093, "code": "USDC"},
    ]
    assert net_funding(entries) == pytest.approx(72.009781)
    assert net_funding([{"amount": -3.625312, "code": "USDC"}]) < 0


# ---------------------------------------------------------------------------
# The live broker: every failure path is None
# ---------------------------------------------------------------------------


class _FakeExchange:
    """Minimal stand-in for ccxt.hyperliquid (ccxt is an optional dep here)."""

    def __init__(self, *, supported=True, entries=None, raises=None):
        self.has = {"fetchFundingHistory": supported}
        self._entries = entries
        self._raises = raises
        self.calls: list[int | None] = []

    def fetch_funding_history(self, since=None, **_kw):
        self.calls.append(since)
        if self._raises is not None:
            raise self._raises
        return self._entries


def _broker(exchange):
    """A HyperliquidBroker with the exchange swapped, no __post_init__ network."""
    from quantbox.plugins.broker.hyperliquid import HyperliquidBroker

    broker = HyperliquidBroker.__new__(HyperliquidBroker)
    broker._exchange = exchange
    return broker


def test_broker_sums_what_the_venue_reports():
    ex = _FakeExchange(entries=[{"amount": -0.31, "code": "USDC"}, {"amount": -0.12, "code": "USDC"}])
    assert _broker(ex).fetch_funding_payments("2026-09-08T00:00:00") == pytest.approx(-0.43)
    # The window actually reached the venue, in ms.
    import pandas as pd

    assert ex.calls == [int(pd.Timestamp("2026-09-08T00:00:00").timestamp() * 1000)]


def test_broker_reports_unknown_when_the_venue_call_raises():
    """A dead API is an unmeasured cost, not a free one."""
    ex = _FakeExchange(raises=RuntimeError("connection reset"))
    assert _broker(ex).fetch_funding_payments("2026-09-08T00:00:00") is None


def test_broker_reports_unknown_when_the_endpoint_is_unsupported():
    ex = _FakeExchange(supported=False)
    assert _broker(ex).fetch_funding_payments("2026-09-08T00:00:00") is None
    assert ex.calls == [], "must not call an endpoint the build does not have"


def test_broker_reports_unknown_without_an_exchange():
    assert _broker(None).fetch_funding_payments("2026-09-08T00:00:00") is None


def test_broker_reports_unknown_on_an_unparseable_window():
    ex = _FakeExchange(entries=[])
    assert _broker(ex).fetch_funding_payments("not-a-timestamp") is None


def test_broker_zero_is_only_ever_a_venue_answer():
    """0.0 is reachable ONLY from an actual empty response, never from a failure."""
    assert _broker(_FakeExchange(entries=[])).fetch_funding_payments("2026-09-08") == 0.0


# ---------------------------------------------------------------------------
# The pipeline: which source, and what a missing source means
# ---------------------------------------------------------------------------


def test_pipeline_reads_funding_from_a_live_broker():
    """The regression itself: a live broker is now ASKED, not defaulted."""
    from quantbox.plugins.pipeline.trading_pipeline import _resolve_funding_charge

    class LiveBroker:
        def __init__(self):
            self.asked_since = None

        def fetch_funding_payments(self, since):
            self.asked_since = since
            return -0.87

    broker = LiveBroker()
    assert _resolve_funding_charge(broker, "2026-09-09T07:00:00") == pytest.approx(-0.87)
    assert broker.asked_since is not None, "the live broker was never asked for funding"
    assert broker.asked_since.startswith("2026-09-08T07:00:00"), "window is not the 24h before asof"


def test_pipeline_keeps_an_unmeasured_funding_charge_unknown():
    """A live broker that could not read the venue must NOT become $0.00."""
    from quantbox.plugins.pipeline.trading_pipeline import _resolve_funding_charge

    class BlindBroker:
        def fetch_funding_payments(self, since):
            return None

    assert _resolve_funding_charge(BlindBroker(), "2026-09-09") is None


def test_pipeline_reports_unknown_when_the_broker_has_no_funding_source():
    """The original #92 shape: a hasattr miss must not fabricate a zero."""
    from quantbox.plugins.pipeline.trading_pipeline import _resolve_funding_charge

    class NoFundingBroker:
        pass

    assert _resolve_funding_charge(NoFundingBroker(), "2026-09-09") is None
    assert _resolve_funding_charge(None, "2026-09-09") is None


def test_pipeline_still_prefers_a_simulated_book_that_applies_funding():
    from quantbox.plugins.pipeline.trading_pipeline import _resolve_funding_charge

    class PaperBroker:
        def apply_funding(self):
            return -1.25

        def fetch_funding_payments(self, since):  # pragma: no cover - must not be used
            raise AssertionError("a book that applies its own funding must not be re-read")

    assert _resolve_funding_charge(PaperBroker(), "2026-09-09") == pytest.approx(-1.25)


def test_unknown_funding_renders_as_null_in_the_artifact_payload():
    """End of the chain: None survives into trading_costs, where the report
    renderer turns it into UNKNOWN rather than $0.00."""
    import pandas as pd

    from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline

    payload = TradingPipeline._build_artifact_payload(
        TradingPipeline(),
        rebalancing_df=pd.DataFrame(),
        orders_df=pd.DataFrame(),
        execution_report={},
        final_weights={},
        total_value=1000.0,
        mode="live",
        funding_charge=_resolved_unknown(),
    )
    assert payload["trading_costs"]["funding_charge"] is None


def _resolved_unknown():
    from quantbox.plugins.pipeline.trading_pipeline import _resolve_funding_charge

    class NoFundingBroker:
        pass

    return _resolve_funding_charge(NoFundingBroker(), "2026-09-09")
