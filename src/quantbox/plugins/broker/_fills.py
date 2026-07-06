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
is non-terminal — ambiguous OR accepted-but-still-settling — do a bounded
confirmation re-poll and otherwise **fail safe to not-filled** (a false FAILED
triggers an alert and a re-attempt next cycle — the safe direction — whereas a
false FILLED is a silent loss).

Kraken spot settles a marketable order *asynchronously*: ``create_order``
returns ``status='open', filled=0`` and the fill lands milliseconds later.
Classifying that first snapshot as a terminal miss (the pre-#97 behaviour)
mis-reported real fills as FAILED, poisoning fill accounting and firing false
alerts. So an ``open``/zero-fill order is FILL_PENDING (re-polled with a short
wait), distinct from a terminal FILL_UNFILLED (a dead/rejected order, reported
FAILED with no wait).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Internal verdicts produced by classify_fill().
# ---------------------------------------------------------------------------
FILL_FILLED = "FILLED"
FILL_PARTIAL = "PARTIAL"
FILL_UNFILLED = "UNFILLED"  # terminal not-filled (dead/closed-with-zero) — a real miss
FILL_PENDING = "PENDING"  # accepted + still working (open, filled=0) — async-settling, re-poll
FILL_UNKNOWN = "UNKNOWN"  # no evidence either way — caller should re-fetch

# Bounded confirmation wait for an async-settling order (Kraken spot settles a
# marketable order milliseconds after create_order returns status='open',
# filled=0 — issue #97). resolve_fill re-polls up to this many times, sleeping
# between polls, before falling back to a fail-safe FAILED.
_CONFIRM_ATTEMPTS = 3
_CONFIRM_DELAY_S = 0.5

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

    # Reference qty for "did it fully fill": the amount ACTUALLY submitted to the
    # venue (order["amount"], which the broker floors to lot/precision before
    # placing), NOT the pre-rounding strategy target `requested_qty`. Otherwise a
    # fully-filled floored order (filled == submitted < requested) reads as a false
    # PARTIAL and spams residual alerts. A genuine partial is still filled < ref.
    ordered = _to_float(order.get("amount"))
    ref = ordered if (ordered is not None and ordered > _EPS) else req

    if status == _CLOSED:
        # Venue says the order is done. Trust the reported filled amount; only
        # when the venue *omits* ``filled`` entirely do we treat a closed order
        # as a full fill. A closed order that explicitly reports filled==0 is
        # contradictory — do NOT claim a fill.
        if filled is None:
            # Venue omitted ``filled``. Treat a closed order as fully filled ONLY
            # when nothing contradicts it. A closed order that still reports a
            # positive ``remaining`` (issue #68 hardening) did NOT fully fill —
            # the unfilled remainder is real residual exposure, critical on a
            # close-out SELL. Never claim a full fill against that evidence.
            if remaining is not None and remaining > _EPS:
                implied = (ref - remaining) if ref > _EPS else 0.0
                if implied > _EPS:
                    return FILL_PARTIAL, implied, price
                return FILL_UNFILLED, 0.0, price
            return FILL_FILLED, req, price
        if has_fill:
            # A closed order that filled LESS than it SUBMITTED (e.g. an IOC that
            # partially filled then canceled the remainder) is a PARTIAL, not a
            # full fill — the unfilled remainder is real residual exposure, which
            # is critical on a close-out SELL. Compared against `ref` (the submitted
            # amount) so lot/precision flooring isn't mistaken for a partial.
            if ref > _EPS and filled < ref - _EPS:
                return FILL_PARTIAL, filled, price
            return FILL_FILLED, filled, price
        return FILL_UNFILLED, 0.0, price

    if status in _DEAD:
        # Terminal reject/cancel/expiry. A dead order will NEVER fill — it stays
        # FILL_UNFILLED so resolve_fill reports FAILED with no re-poll wait. This
        # is the guard that keeps a genuine reject reporting failed.
        return (FILL_PARTIAL, filled, price) if has_fill else (FILL_UNFILLED, 0.0, price)

    if status == _OPEN:
        # An accepted order still working the book. With a fill it's a PARTIAL;
        # with ZERO fill it is NOT a failure — Kraken spot returns status='open',
        # filled=0 for a marketable order that settles async milliseconds later
        # (issue #97). Report FILL_PENDING so resolve_fill does a bounded
        # confirmation re-poll before declaring FAILED, instead of the old
        # zero-wait FILL_UNFILLED that mis-reported real fills as failures.
        return (FILL_PARTIAL, filled, price) if has_fill else (FILL_PENDING, 0.0, price)

    # Status missing / unrecognised: decide on the numeric fill evidence.
    if filled is not None:
        if not has_fill:
            return FILL_UNFILLED, 0.0, price
        if remaining is not None and remaining > _EPS:
            return FILL_PARTIAL, filled, price
        # filled > 0 and remainder 0 / unknown: FILLED only if it reached the
        # requested qty. A visible underfill (filled < requested) is a PARTIAL even
        # without an explicit ``remaining`` field — same rule as the CLOSED branch,
        # so a real partial can't slip through the status-less path as a clean fill.
        if ref > _EPS and filled < ref - _EPS:
            return FILL_PARTIAL, filled, price
        return FILL_FILLED, filled, price

    # No status AND no ``filled`` field: genuinely unknown — caller must verify.
    return FILL_UNKNOWN, 0.0, price


def resolve_fill(
    order: dict | None,
    requested_qty: float,
    *,
    refetch: Callable[[], dict | None] | None = None,
    confirm_attempts: int = _CONFIRM_ATTEMPTS,
    confirm_delay: float = _CONFIRM_DELAY_S,
) -> tuple[str, float, float, str]:
    """Resolve an order into an *emitted* ``(status, qty, price, reason)`` row.

    ``status`` is one of ``"FILLED"`` / ``"PARTIAL"`` / ``"FAILED"`` — the
    vocabulary the pipeline understands. When the first classification is
    non-terminal — either ambiguous (FILL_UNKNOWN) or accepted-but-still-settling
    (FILL_PENDING) — and a ``refetch`` callable is supplied, the order is re-read
    (bounded ``confirm_attempts`` re-polls, sleeping ``confirm_delay`` s between
    polls for a PENDING order so Kraken's async settlement can land — issue #97).
    A terminal reject classifies as FILL_UNFILLED, never FILL_PENDING, so it is
    NOT re-polled and reports FAILED immediately. If the order is *still*
    unconfirmed after the wait, the result fails safe to ``"FAILED"`` (never an
    unconfirmed FILLED).
    """
    verdict, filled_qty, price = classify_fill(order, requested_qty)

    # Non-terminal (unknown or async-settling) + we can re-read the venue:
    # bounded confirmation poll. Re-classification is driven purely by the
    # venue's actual reported state, so a real reject can never be masked into a
    # fill — it simply re-reads as dead/unfilled and still reports FAILED.
    if verdict in (FILL_UNKNOWN, FILL_PENDING) and refetch is not None:
        for _ in range(max(1, confirm_attempts)):
            # Give async settlement time BEFORE re-reading a still-working order.
            # (UNKNOWN is a parse-ambiguity, not a settlement delay — don't sleep.)
            if verdict == FILL_PENDING and confirm_delay > 0:
                time.sleep(confirm_delay)
            try:
                refetched = refetch()
            except Exception:  # noqa: BLE001 - never let confirmation crash execution
                refetched = None
            if not refetched:
                break  # can't confirm — keep current verdict, fail safe below
            verdict, filled_qty, price = classify_fill(refetched, requested_qty)
            if verdict not in (FILL_UNKNOWN, FILL_PENDING):
                break  # reached a terminal state (filled / partial / dead)

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
