"""Broker fill-status normalization for the recon ledger (issue #93, item 2).

`_EXEC_STATUS_TO_LEDGER` maps each broker's execution status vocabulary to the
ledger's canonical `RESULT_STATUSES`. The #92 review flagged that statuses
beyond FILLED/PARTIAL/FAILED/SKIPPED (e.g. CANCELED, EXPIRED) must be classified
as NON-fills for the failure/missed-fill streak rather than silently ignored or
logged as "unknown". These tests lock that mapping down and prove every mapped
status the ledger will see is a recognised RESULT_STATUS (no spurious warning).
"""

from __future__ import annotations

from quantbox.plugins.pipeline.trading_pipeline import _EXEC_STATUS_TO_LEDGER
from quantbox.reconciliation.ledger import RESULT_STATUSES


def _to_ledger(status: str) -> str:
    """Mirror the pipeline mapping: known → canonical, else lowercased passthrough."""
    return _EXEC_STATUS_TO_LEDGER.get(status.strip().upper(), status.strip().lower())


def test_known_fill_statuses_map_to_fills():
    assert _to_ledger("FILLED") == "filled"
    assert _to_ledger("PARTIAL") == "partial"


def test_terminal_non_fills_map_to_recognised_non_fill_statuses():
    # CANCELED / CANCELLED / EXPIRED are terminal non-fills a live venue can emit.
    for status in ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED", "FAILED", "SKIPPED"):
        mapped = _to_ledger(status)
        assert mapped in RESULT_STATUSES, f"{status} -> {mapped} not a known RESULT_STATUS"
        # And none of them is a fill: they must feed the failure streak.
        assert mapped not in ("filled", "partial"), status


def test_timeout_maps_to_timeout_missed_fill():
    # TIMEOUT is the missed-fill signal (submitted, no confirmed result), distinct
    # from an outright failure; it must map to the ledger's `timeout`.
    mapped = _to_ledger("TIMEOUT")
    assert mapped == "timeout"
    assert mapped in RESULT_STATUSES


def test_every_mapped_target_is_a_recognised_ledger_status():
    # No entry in the map may point at a status the ledger doesn't recognise,
    # else record_result would warn "unknown status" on a status we explicitly
    # normalise.
    for src, dst in _EXEC_STATUS_TO_LEDGER.items():
        assert dst in RESULT_STATUSES, f"{src} -> {dst} not in RESULT_STATUSES"


def test_case_and_whitespace_insensitive():
    assert _to_ledger("  canceled ") == "failed"
    assert _to_ledger("Expired") == "failed"
