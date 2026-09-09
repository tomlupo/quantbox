"""Honest realised-funding accounting for perpetuals brokers (issue #92).

Funding is a *first-order* cost on a perps book — it is charged every hour on
the full notional, independent of how much you trade — and for 165 days every
daily report said ``Funding charge $0.00``. Nothing was measured: no broker
ever fetched the venue's funding history, so the pipeline's ``funding_charge``
fell through to a default and printed as free.

This module is the funding-side twin of :mod:`._fills`'s fee extraction, and it
carries the same rule: **absence is UNKNOWN, never 0.0.** A funding number we
could not read must be representable as such and must reach the report as
``None``, so it renders UNKNOWN rather than a fabricated zero.

Sign convention
---------------
An entry's ``amount`` is a *signed cash delta on the account*: **negative when
the account PAID funding, positive when it RECEIVED it.** That is the venue's
own convention, not ours — Hyperliquid's ``userFunding`` reports ``delta.usdc``
and ccxt passes it through un-negated (``parse_income``: ``amount =
safe_string(delta, 'usdc')``). Both documented examples agree:

  * long 49.1477 ETH, fundingRate +0.0000417 → the long pays → ``usdc``
    ``-3.625312``;
  * short 7375.9 SOL, fundingRate +0.00004381 → the short receives → ``usdc``
    ``+75.635093``.

It is also the convention the simulated book already uses:
``futures_paper.apply_funding`` returns ``-qty * price * rate`` and *adds* it to
the margin balance. So a live and a simulated ``funding_charge`` mean the same
thing and can sit in the same report column without a sign flip hiding in one
of the two paths.
"""

from __future__ import annotations

from typing import Any

# The settlement currency a perps funding payment is expected to be denominated
# in. An entry that states something else is not summable into the same total.
DEFAULT_QUOTE_CURRENCY = "USDC"


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


def funding_payment(entry: dict | None) -> float | None:
    """Signed funding cash flow from one ccxt funding-history entry.

    ``None`` when the entry carries no readable ``amount`` — an entry we cannot
    parse is not a zero-value entry, and treating it as one is how a real cost
    disappears into a total that still looks like a measurement.
    """
    if not entry:
        return None
    return _to_float(entry.get("amount"))


def select_window(
    entries: list[dict] | None,
    start_ms: int,
    end_ms: int,
) -> list[dict] | None:
    """The entries that fall in ``[start_ms, end_ms)``, or ``None`` if any cannot
    be placed in time at all.

    A funding total is only a measurement of the window it claims. The first cut
    of this code asked the venue for everything ``since`` the window start and
    never sent an end bound, so a report for a historical ``asof`` summed
    payments through wall-clock *now* and printed the result as a 24h figure —
    a plausible number for the wrong interval, which is the #92 defect wearing a
    different disguise. The bound is now sent to the venue AND enforced here, so
    a venue (or ccxt build) that ignores ``endTime`` cannot silently widen the
    window behind the caller's back.

    Half-open on purpose: a payment landing exactly at a window's end belongs to
    the next window, so consecutive daily reports neither double-count it nor
    drop it.

    An entry with no readable timestamp returns ``None`` for the whole list. It
    cannot be attributed to the window, and neither keeping it nor dropping it
    would be a fact — the same rule this module applies to an unreadable amount.
    """
    if entries is None:
        return None

    kept: list[dict] = []
    for entry in entries:
        ts = _to_float((entry or {}).get("timestamp"))
        if ts is None:
            return None
        if start_ms <= ts < end_ms:
            kept.append(entry)
    return kept


def net_funding(
    entries: list[dict] | None,
    *,
    quote_currency: str = DEFAULT_QUOTE_CURRENCY,
) -> float | None:
    """Net signed funding across ``entries``, or ``None`` when UNKNOWN.

    Returns ``None`` — never a partial sum — when:

    * ``entries`` is ``None``. That is the caller saying *the venue could not be
      read*, which is exactly the case this module exists to keep visible.
    * any entry has no readable ``amount``. A sum missing an unknown number of
      its terms is a fabricated number wearing a plausible value, which is the
      #92 defect one level up from the fabricated zero.
    * any entry states a currency other than ``quote_currency``. -3.6 USDC and
      -0.002 BNB are not -3.602 of anything (mirrors ``_fills.trade_fee``).

    An **empty list is 0.0, not None**: the venue was asked and affirmatively
    answered that no funding was charged in the window — a book that held no
    perps overnight genuinely pays nothing. That is data, and the distinction
    only holds because the caller is required to pass ``None`` (not ``[]``) when
    the request itself failed.
    """
    if entries is None:
        return None

    total = 0.0
    for entry in entries:
        amount = funding_payment(entry)
        if amount is None:
            return None
        code = (entry or {}).get("code")
        if code is not None and str(code) != quote_currency:
            return None
        total += amount
    return total
