"""Tests for Kraken ledger-based cashflow retrieval.

ccxt is mocked via an injected fake exchange — these tests NEVER hit live
Kraken, use no API credentials and touch no real capital.
"""

from __future__ import annotations

import pytest

from quantbox.plugins.broker.kraken import LEDGER_PAGE_SIZE, KrakenBroker


def _entry(entry_id, ts_ms, type_, asset, amount, fee="0.0", refid="REF-1"):
    """A ccxt-shaped ledger entry: parsed fields + the raw Kraken payload in
    ``info``. ccxt's parsed ``amount`` is UNSIGNED (sign lives in ``direction``),
    which is exactly the trap the broker must avoid — so the fixture reproduces it.
    """
    signed = float(amount)
    return {
        "id": entry_id,
        "timestamp": ts_ms,
        "datetime": None,
        "direction": "out" if signed < 0 else "in",
        "type": "transaction" if type_ in ("deposit", "withdrawal") else type_,
        "currency": asset,
        "amount": abs(signed),
        "referenceId": refid,
        "info": {
            "refid": refid,
            "time": ts_ms / 1000,
            "type": type_,
            "aclass": "currency",
            "asset": asset,
            "amount": str(amount),
            "fee": str(fee),
            "balance": "0.0",
        },
    }


class _LedgerExchange:
    """Fake ccxt.kraken exposing only what the ledger path uses, with Kraken's
    real ``ofs`` pagination semantics (50 rows/page, offset-driven)."""

    def __init__(self, entries, page_size=LEDGER_PAGE_SIZE, fail_on_page=None):
        self.entries = entries
        self.page_size = page_size
        self.fail_on_page = fail_on_page
        self.calls: list[dict] = []
        self.markets = {}

    def load_markets(self):
        return self.markets

    def fetch_ledger(self, code=None, since=None, params=None):
        params = params or {}
        ofs = int(params.get("ofs", 0))
        self.calls.append({"since": since, "ofs": ofs})
        if self.fail_on_page is not None and ofs // self.page_size == self.fail_on_page:
            raise RuntimeError("kraken exploded")
        rows = self.entries
        if since is not None:
            rows = [e for e in rows if e["timestamp"] >= since]
        return rows[ofs : ofs + self.page_size]


def _broker(exchange):
    return KrakenBroker(quote_asset="USD", readonly=True, _exchange=exchange)


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def test_pagination_walks_every_page():
    entries = [_entry(f"L{i}", 1_700_000_000_000 + i * 1000, "deposit", "ZUSD", "10.0") for i in range(120)]
    ex = _LedgerExchange(entries)
    out = _broker(ex).fetch_ledger_entries()
    assert len(out) == 120
    assert [c["ofs"] for c in ex.calls] == [0, 50, 100]


def test_single_short_page_stops_immediately():
    ex = _LedgerExchange([_entry("L1", 1_700_000_000_000, "deposit", "ZUSD", "10.0")])
    assert len(_broker(ex).fetch_ledger_entries()) == 1
    assert len(ex.calls) == 1


def test_duplicate_page_terminates_loop():
    """Kraken can re-serve rows when the ledger grows mid-walk. An all-duplicate
    page must terminate rather than loop forever."""
    entries = [_entry(f"L{i}", 1_700_000_000_000 + i * 1000, "deposit", "ZUSD", "10.0") for i in range(50)]

    class _Stuck(_LedgerExchange):
        def fetch_ledger(self, code=None, since=None, params=None):
            self.calls.append({"since": since, "ofs": (params or {}).get("ofs", 0)})
            return self.entries  # always the same full page

    ex = _Stuck(entries)
    out = _broker(ex).fetch_ledger_entries()
    assert len(out) == 50
    assert len(ex.calls) == 2  # second page was entirely duplicates


def test_max_pages_exceeded_raises_rather_than_truncating():
    entries = [_entry(f"L{i}", 1_700_000_000_000 + i * 1000, "deposit", "ZUSD", "10.0") for i in range(500)]
    ex = _LedgerExchange(entries)
    with pytest.raises(RuntimeError, match="max_pages"):
        _broker(ex).fetch_ledger_entries(max_pages=2)


def test_upstream_error_propagates_not_swallowed():
    entries = [_entry(f"L{i}", 1_700_000_000_000 + i * 1000, "deposit", "ZUSD", "10.0") for i in range(120)]
    ex = _LedgerExchange(entries, fail_on_page=1)
    with pytest.raises(RuntimeError, match="kraken exploded"):
        _broker(ex).fetch_ledger_entries()


def test_since_is_passed_as_ms_epoch():
    ex = _LedgerExchange([_entry("L1", 1_700_000_000_000, "deposit", "ZUSD", "10.0")])
    _broker(ex).fetch_ledger_entries(since="2026-02-07")
    assert ex.calls[0]["since"] == 1770422400000


def test_unparseable_since_raises():
    ex = _LedgerExchange([])
    with pytest.raises(ValueError, match="Unparseable"):
        _broker(ex).fetch_ledger_entries(since="not-a-date")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_cashflows_filter_and_sign():
    entries = [
        _entry("L1", 1_770_422_400_000, "deposit", "ZUSD", "500.0", fee="0.0"),
        _entry("L2", 1_770_508_800_000, "trade", "XXBT", "-0.01"),
        _entry("L3", 1_770_595_200_000, "withdrawal", "ZUSD", "-100.0", fee="5.0"),
        _entry("L4", 1_770_681_600_000, "margin", "ZUSD", "1.0"),
    ]
    df = _broker(_LedgerExchange(entries)).fetch_cashflows()
    assert list(df["id"]) == ["L1", "L3"]
    assert list(df["type"]) == ["deposit", "withdrawal"]
    assert list(df["amount"]) == [500.0, -100.0]
    # Fee is booked separately: the real balance delta is amount - fee.
    assert list(df["amount_net"]) == [500.0, -105.0]
    assert list(df["currency"]) == ["USD", "USD"]
    assert list(df["date"]) == ["2026-02-07", "2026-02-09"]


def test_transfer_is_returned_not_classified():
    entries = [_entry("L1", 1_770_422_400_000, "transfer", "ZUSD", "250.0")]
    df = _broker(_LedgerExchange(entries)).fetch_cashflows()
    assert list(df["type"]) == ["transfer"]


def test_non_quote_asset_flow_kept_in_native_units():
    """A BTC deposit must NOT be silently valued in USD — it comes back as BTC."""
    entries = [_entry("L1", 1_770_422_400_000, "deposit", "XXBT", "0.5")]
    df = _broker(_LedgerExchange(entries)).fetch_cashflows()
    assert df.loc[0, "currency"] == "BTC"
    assert df.loc[0, "amount"] == 0.5


def test_empty_ledger_returns_empty_frame_with_columns():
    df = _broker(_LedgerExchange([])).fetch_cashflows()
    assert df.empty
    assert "amount_net" in df.columns


def test_results_sorted_by_time():
    entries = [
        _entry("L2", 1_770_595_200_000, "deposit", "ZUSD", "10.0"),
        _entry("L1", 1_770_422_400_000, "deposit", "ZUSD", "20.0"),
    ]
    df = _broker(_LedgerExchange(entries)).fetch_cashflows()
    assert list(df["id"]) == ["L1", "L2"]


def test_missing_timestamp_raises():
    e = _entry("L1", 1_770_422_400_000, "deposit", "ZUSD", "10.0")
    e["timestamp"] = None
    with pytest.raises(ValueError, match="no timestamp"):
        _broker(_LedgerExchange([e])).fetch_cashflows()
