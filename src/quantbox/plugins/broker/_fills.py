"""Honest fill-status classification for live ccxt brokers (issue #68).

An accepted ccxt order result is **not** a guarantee of a fill. Reading
``status == FILLED`` off any returned order — and falling back to the
*requested* qty when ccxt returns ``filled = 0 / absent`` — can make the book
believe a live order filled when it is actually still open, partial, or
rejected. That is a silent wrong-state on real capital: the rebalancer thinks
it is positioned when it is not.

This module centralises the classification used by the Kraken and Hyperliquid
brokers. The philosophy mirrors the rebalancer dead-man: only report a fill we
can affirmatively see; never assume ``requested == filled``; and when a result
is ambiguous, do one follow-up fetch and otherwise **fail safe to not-filled**
(a false FAILED triggers an alert and a re-attempt next cycle — the safe
direction — whereas a false FILLED is a silent loss).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Internal verdicts produced by classify_fill().
# ---------------------------------------------------------------------------
FILL_FILLED = "FILLED"
FILL_PARTIAL = "PARTIAL"
FILL_UNFILLED = "UNFILLED"
FILL_UNKNOWN = "UNKNOWN"  # no evidence either way — caller should re-fetch

# ccxt unified order statuses.
_CLOSED = "closed"  # fully filled / no longer working
_DEAD = {"canceled", "cancelled", "rejected", "expired"}
_OPEN = "open"

# Below this many base units, a reported fill is treated as zero (float noise).
_EPS = 1e-12


def _to_float(value: Any) -> float | None:
    """Best-effort float; returns None for None / unparseable / NaN."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _fill_price(order: dict) -> float:
    price = _to_float(order.get("average"))
    if price is None or price <= 0:
        price = _to_float(order.get("price")) or 0.0
    return price


def classify_fill(order: dict | None, requested_qty: float) -> tuple[str, float, float]:
    """Classify a ccxt order result into ``(verdict, filled_qty, fill_price)``.

    ``verdict`` is one of FILL_FILLED / FILL_PARTIAL / FILL_UNFILLED /
    FILL_UNKNOWN. ``filled_qty`` is the *actual* filled base quantity (never the
    requested qty unless the venue explicitly reports the order closed without a
    ``filled`` field). FILL_UNKNOWN means the result carried no status and no
    ``filled`` field — the caller should re-fetch the order before deciding.
    """
    req = _to_float(requested_qty) or 0.0
    if not order:
        return FILL_UNKNOWN, 0.0, 0.0

    status = str(order.get("status") or "").strip().lower()
    filled = _to_float(order.get("filled"))
    remaining = _to_float(order.get("remaining"))
    price = _fill_price(order)
    has_fill = filled is not None and filled > _EPS

    if status == _CLOSED:
        # Venue says the order is done. Trust the reported filled amount; only
        # when the venue *omits* ``filled`` entirely do we treat a closed order
        # as a full fill. A closed order that explicitly reports filled==0 is
        # contradictory — do NOT claim a fill.
        if filled is None:
            return FILL_FILLED, req, price
        if has_fill:
            # A closed order that filled LESS than requested (e.g. an IOC that
            # partially filled then canceled the remainder) is a PARTIAL, not a
            # full fill — the unfilled remainder is real residual exposure, which
            # is critical on a close-out SELL. Only filled >= requested is FILLED.
            if req > _EPS and filled < req - _EPS:
                return FILL_PARTIAL, filled, price
            return FILL_FILLED, filled, price
        return FILL_UNFILLED, 0.0, price

    if status in _DEAD:
        return (FILL_PARTIAL, filled, price) if has_fill else (FILL_UNFILLED, 0.0, price)

    if status == _OPEN:
        return (FILL_PARTIAL, filled, price) if has_fill else (FILL_UNFILLED, 0.0, price)

    # Status missing / unrecognised: decide on the numeric fill evidence.
    if filled is not None:
        if not has_fill:
            return FILL_UNFILLED, 0.0, price
        if remaining is not None and remaining > _EPS:
            return FILL_PARTIAL, filled, price
        # filled > 0 and remainder 0 / unknown: take the visible filled amount.
        return FILL_FILLED, filled, price

    # No status AND no ``filled`` field: genuinely unknown — caller must verify.
    return FILL_UNKNOWN, 0.0, price


def resolve_fill(
    order: dict | None,
    requested_qty: float,
    *,
    refetch: Callable[[], dict | None] | None = None,
) -> tuple[str, float, float, str]:
    """Resolve an order into an *emitted* ``(status, qty, price, reason)`` row.

    ``status`` is one of ``"FILLED"`` / ``"PARTIAL"`` / ``"FAILED"`` — the
    vocabulary the pipeline understands. When the first classification is
    ambiguous (FILL_UNKNOWN) and a ``refetch`` callable is supplied, the order
    is re-read once; if it is *still* unconfirmable, the result fails safe to
    ``"FAILED"`` (never an unconfirmed FILLED).
    """
    verdict, filled_qty, price = classify_fill(order, requested_qty)

    if verdict == FILL_UNKNOWN and refetch is not None:
        try:
            refetched = refetch()
        except Exception:  # noqa: BLE001 - never let confirmation crash execution
            refetched = None
        if refetched:
            v2, q2, p2 = classify_fill(refetched, requested_qty)
            if v2 != FILL_UNKNOWN:
                verdict, filled_qty, price = v2, q2, p2

    if verdict == FILL_FILLED:
        return "FILLED", filled_qty, price, ""
    if verdict == FILL_PARTIAL:
        return (
            "PARTIAL",
            filled_qty,
            price,
            f"partial fill: {filled_qty:g}/{float(requested_qty):g} filled",
        )

    # FILL_UNFILLED or still-FILL_UNKNOWN: not confirmed filled. Report honestly as
    # FAILED. This is safe against double-placement: the pipeline reconciles from the
    # broker's ACTUAL positions every cycle (get_positions -> target-vs-current diff),
    # so if this order in fact filled, the next cycle sees the resulting position and
    # places no duplicate; a genuine miss is simply re-attempted. A false FAILED costs
    # an alert + one re-check, never a double order.
    raw_status = "unknown"
    if order:
        raw_status = str(order.get("status") or "unknown")
    return (
        "FAILED",
        0.0,
        price,
        f"order not confirmed filled (venue status={raw_status})",
    )
