"""Venue lot rules: the one place an order quantity is rounded to what a venue accepts (TOM-1458).

Before this module the rounding existed eight times: in the trading pipeline's
private rebalancer, the standard rebalancer, three brokers and the
allocations-to-orders pipeline, each a copy of the last with small drifts. Every
caller now rounds here; WHICH rule a caller uses is still its own, because the
rule is a fact about its venue.

**Per-venue behaviour is today's** (Tom, 2026-10-10, P5 grilling Q5): Binance
spot, Binance futures, the spot rebalancer and Kraken's fallback floor;
Hyperliquid rounds to the NEAREST step (:func:`round_to_decimals`). The switch
of Hyperliquid to round-down is its own slice, TOM-1721.

Only quantities are rounded here. Whether an order clears a venue minimum
(``min_qty``, ``min_notional``) is decided by the caller, because the exemptions
differ by intent: a reduce-only close is exempt on a perps venue, a partial
reduce is not (TOM-1778).
"""

from __future__ import annotations

import math
from decimal import ROUND_DOWN, Decimal
from typing import Any

#: Digits a step size is written with when its decimal places are counted.
_STEP_DIGITS = 8


def step_decimals(step: float, digits: int = _STEP_DIGITS) -> int:
    """Decimal places of *step* written with *digits* digits: ``0.001`` → 3, ``1`` → 0, ``0.25`` → 2."""
    text = f"{step:.{digits}f}".rstrip("0")
    return len(text.split(".")[-1]) if "." in text else 0


def floor_to_decimals(qty: float, decimals: int) -> float:
    """*qty* rounded toward zero to *decimals* decimal places, in decimal arithmetic.

    The float is read through its shortest repr (``str``), so ``0.3`` floors to
    ``0.3`` and not to ``0.2``. The rounding mode is passed to ``quantize``; the
    process-wide decimal context is never changed.
    """
    exponent = Decimal(1).scaleb(-int(decimals))
    return float(Decimal(str(qty)).quantize(exponent, rounding=ROUND_DOWN))


def floor_to_step(qty: float, step: float) -> float:
    """*qty* rounded toward zero to the decimal places of *step* (the Binance LOT_SIZE rule).

    A step of 0 or less means "no step known": *qty* is returned as is. The rule
    counts the step's decimal places; it does not round to multiples of the step
    (``0.5`` floors to one decimal). That is the rule every Binance-style caller
    applied before this module.
    """
    if step <= 0:
        return qty
    return floor_to_decimals(qty, step_decimals(step))


def round_to_decimals(qty: float, decimals: int) -> float:
    """*qty* rounded to the NEAREST *decimals* decimal places (Hyperliquid today; TOM-1721 changes it)."""
    return round(qty, int(decimals))


def amount_decimals(precision: Any) -> int:
    """A ccxt ``precision["amount"]`` as a number of decimal places.

    ccxt gives either a step (a float below 1, e.g. ``0.001``) or a number of
    decimals (e.g. ``3``). A step becomes its decimal places; anything else is
    read as a number of decimals.
    """
    if isinstance(precision, float) and 0 < precision < 1:
        return max(0, -int(math.floor(math.log10(precision))))
    return int(precision)


def floor_to_multiple(qty: float, step: float) -> float:
    """*qty* floored to a whole multiple of *step*; a step of 0 or less returns *qty*."""
    if step <= 0:
        return qty
    return math.floor(qty / step) * step


def round_to_multiple(qty: float, lot: float) -> float:
    """*qty* rounded to the NEAREST whole multiple of *lot*; a lot of 0 or less returns *qty*."""
    if lot <= 0:
        return qty
    return round(qty / lot) * lot


def binance_filters(symbol_info: dict | None) -> tuple[float, float, float]:
    """``(min_qty, step_size, min_notional)`` from a Binance symbol-info payload; zeros when absent."""
    min_qty, step_size, min_notional = 0.0, 0.0, 0.0
    if not symbol_info:
        return min_qty, step_size, min_notional
    for f in symbol_info.get("filters", []) or []:
        kind = f.get("filterType")
        if kind == "LOT_SIZE":
            min_qty = float(f.get("minQty", 0))
            step_size = float(f.get("stepSize", 0))
        elif kind == "NOTIONAL":
            min_notional = float(f.get("minNotional", 0))
    return min_qty, step_size, min_notional
