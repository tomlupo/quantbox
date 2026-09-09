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

from quantbox.plugins.broker._funding import funding_payment, net_funding, select_window

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
# The window: a total is only a measurement of the interval it claims
# ---------------------------------------------------------------------------

START = 1_757_289_600_000  # an arbitrary window start, in ms
END = START + 86_400_000  # +24h


def test_entries_outside_the_window_are_excluded():
    """What an unbounded fetch hands back must not land in the total.

    The first cut sent only `since` and no end bound, so the venue answered
    through wall-clock now and a historical `asof` summed days of payments into
    a figure the report labels "24h". The three entries here are exactly that
    shape: one before the window, one inside, one after.
    """
    entries = [
        {"timestamp": START - 1, "amount": -50.0, "code": "USDC"},
        {"timestamp": START + 3_600_000, "amount": -0.25, "code": "USDC"},
        {"timestamp": END + 1, "amount": -50.0, "code": "USDC"},
    ]
    kept = select_window(entries, START, END)
    assert [e["amount"] for e in kept] == [-0.25]
    assert net_funding(kept) == pytest.approx(-0.25)


def test_the_window_is_half_open_so_daily_reports_abut_exactly():
    """A payment at the boundary belongs to exactly one of two adjacent windows."""
    at_start = {"timestamp": START, "amount": -1.0, "code": "USDC"}
    at_end = {"timestamp": END, "amount": -2.0, "code": "USDC"}
    assert select_window([at_start, at_end], START, END) == [at_start]
    # ...and the one at END is picked up by the NEXT window, not dropped.
    assert select_window([at_start, at_end], END, END + 86_400_000) == [at_end]


def test_an_entry_that_cannot_be_placed_in_time_is_unknown():
    """No timestamp: neither keeping nor dropping it would be a fact."""
    assert select_window([{"amount": -1.0, "code": "USDC"}], START, END) is None
    assert select_window([{"timestamp": "nonsense", "amount": -1.0}], START, END) is None
    assert select_window(None, START, END) is None


# ---------------------------------------------------------------------------
# The live broker: every failure path is None
# ---------------------------------------------------------------------------


class _FakeExchange:
    """Minimal stand-in for ccxt.hyperliquid (ccxt is an optional dep here)."""

    def __init__(self, *, supported=True, entries=None, raises=None, ignores_until=False):
        self.has = {"fetchFundingHistory": supported}
        self._entries = entries
        self._raises = raises
        # A venue (or ccxt build) that accepts `until` and answers as if it had
        # not been sent. The broker must not be fooled by one.
        self._ignores_until = ignores_until
        self.calls: list[tuple[int | None, dict]] = []

    def fetch_funding_history(self, since=None, params=None, **_kw):
        params = params or {}
        self.calls.append((since, params))
        if self._raises is not None:
            raise self._raises
        if self._entries is None:
            return None
        until = params.get("until")
        if until is None or self._ignores_until:
            return list(self._entries)
        return [e for e in self._entries if e.get("timestamp", since) < until]


def _broker(exchange):
    """A HyperliquidBroker with the exchange swapped, no __post_init__ network."""
    from quantbox.plugins.broker.hyperliquid import HyperliquidBroker

    broker = HyperliquidBroker.__new__(HyperliquidBroker)
    broker._exchange = exchange
    return broker


SINCE = "2026-09-08T00:00:00"
UNTIL = "2026-09-09T00:00:00"


def _ms(iso: str) -> int:
    import pandas as pd

    return int(pd.Timestamp(iso).timestamp() * 1000)


def test_broker_sums_what_the_venue_reports():
    ex = _FakeExchange(
        entries=[
            {"timestamp": _ms(SINCE) + 1, "amount": -0.31, "code": "USDC"},
            {"timestamp": _ms(SINCE) + 2, "amount": -0.12, "code": "USDC"},
        ]
    )
    assert _broker(ex).fetch_funding_payments(SINCE, UNTIL) == pytest.approx(-0.43)


def test_broker_sends_both_bounds_to_the_venue():
    """The end bound must actually leave the process.

    A test that only checked `since=` was passed would have stayed green through
    the whole unbounded-window bug — the start was always right; the end was
    never sent, so the venue answered through wall-clock now.
    """
    ex = _FakeExchange(entries=[])
    _broker(ex).fetch_funding_payments(SINCE, UNTIL)
    ((sent_since, sent_params),) = ex.calls
    assert sent_since == _ms(SINCE)
    assert sent_params.get("until") == _ms(UNTIL), "the window has no end bound"


def test_broker_attributes_only_the_requested_window():
    """The bug, as arithmetic.

    Nine days of payments answered to a one-day request: unbounded, the total is
    -9.00 and the report labels it a day's funding. Bounded, it is -1.00.
    """
    day = 86_400_000
    entries = [{"timestamp": _ms(SINCE) + i * day + 1, "amount": -1.0, "code": "USDC"} for i in range(9)]
    assert _broker(_FakeExchange(entries=entries)).fetch_funding_payments(SINCE, UNTIL) == pytest.approx(-1.0)


def test_broker_re_applies_the_bound_a_venue_ignored():
    """Sending `until` is not the same as it being honoured."""
    day = 86_400_000
    entries = [{"timestamp": _ms(SINCE) + i * day + 1, "amount": -1.0, "code": "USDC"} for i in range(9)]
    ex = _FakeExchange(entries=entries, ignores_until=True)
    assert _broker(ex).fetch_funding_payments(SINCE, UNTIL) == pytest.approx(-1.0)


def test_broker_reports_unknown_when_an_entry_has_no_timestamp():
    """Unattributable to the window — so the window has no honest total."""
    ex = _FakeExchange(entries=[{"amount": -0.31, "code": "USDC"}])
    assert _broker(ex).fetch_funding_payments(SINCE, UNTIL) is None


def test_broker_reports_unknown_when_the_venue_call_raises():
    """A dead API is an unmeasured cost, not a free one."""
    ex = _FakeExchange(raises=RuntimeError("connection reset"))
    assert _broker(ex).fetch_funding_payments(SINCE, UNTIL) is None


def test_broker_reports_unknown_when_the_endpoint_is_unsupported():
    ex = _FakeExchange(supported=False)
    assert _broker(ex).fetch_funding_payments(SINCE, UNTIL) is None
    assert ex.calls == [], "must not call an endpoint the build does not have"


def test_broker_reports_unknown_without_an_exchange():
    assert _broker(None).fetch_funding_payments(SINCE, UNTIL) is None


def test_broker_reports_unknown_on_an_unparseable_window():
    ex = _FakeExchange(entries=[])
    assert _broker(ex).fetch_funding_payments("not-a-timestamp", UNTIL) is None
    assert _broker(ex).fetch_funding_payments(SINCE, "not-a-timestamp") is None


def test_broker_zero_is_only_ever_a_venue_answer():
    """0.0 is reachable ONLY from an actual empty response, never from a failure."""
    assert _broker(_FakeExchange(entries=[])).fetch_funding_payments(SINCE, UNTIL) == 0.0


# ---------------------------------------------------------------------------
# The pipeline: which source, and what a missing source means
# ---------------------------------------------------------------------------


def test_pipeline_reads_funding_from_a_live_broker():
    """The regression itself: a live broker is now ASKED, not defaulted."""
    from quantbox.plugins.pipeline.trading_pipeline import _resolve_funding_charge

    class LiveBroker:
        def __init__(self):
            self.asked = None

        def fetch_funding_payments(self, since, until):
            self.asked = (since, until)
            return -0.87

    broker = LiveBroker()
    assert _resolve_funding_charge(broker, "2026-09-09T07:00:00") == pytest.approx(-0.87)
    assert broker.asked is not None, "the live broker was never asked for funding"
    since, until = broker.asked
    assert since.startswith("2026-09-08T07:00:00"), "window does not start 24h before asof"
    # The end is `asof`, NOT the moment of the run: a report for a historical
    # asof must not sweep up everything since, and consecutive daily windows
    # must abut rather than overlap.
    assert until.startswith("2026-09-09T07:00:00"), "window is not closed at asof"


def test_pipeline_keeps_an_unmeasured_funding_charge_unknown():
    """A live broker that could not read the venue must NOT become $0.00."""
    from quantbox.plugins.pipeline.trading_pipeline import _resolve_funding_charge

    class BlindBroker:
        def fetch_funding_payments(self, since, until):
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

        def fetch_funding_payments(self, since, until):  # pragma: no cover - must not be used
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


def test_consecutive_daily_windows_abut_without_overlap():
    """Yesterday's end is today's start — no gap, no double-counted payment."""
    from quantbox.plugins.pipeline.trading_pipeline import _funding_window

    d8_start, d8_end = _funding_window("2026-09-08T07:00:00")
    d9_start, d9_end = _funding_window("2026-09-09T07:00:00")
    assert d8_end == d9_start
    assert d8_start < d8_end < d9_end
