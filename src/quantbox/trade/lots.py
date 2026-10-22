"""Venue lot rules: the one place an order quantity is rounded to what a venue accepts (TOM-1458).

Before this module the rounding existed eight times: in the trading pipeline's
private rebalancer, the standard rebalancer, three brokers and the
allocations-to-orders pipeline, each a copy of the last with small drifts. Every
caller now rounds here; WHICH step or precision a caller uses is still its own,
because that is a fact about its venue.

**An order quantity is rounded DOWN on every venue** (Tom, 2026-10-10, P5
grilling Q5; TOM-1721): Binance spot, Binance futures, the spot rebalancer,
Kraken and Hyperliquid. A rounded order never exceeds its target or the cash
behind it. A full close is not cut short: a position size the venue reports is
on the venue's grid, and flooring an on-grid quantity returns it unchanged.

**A target position is rounded toward zero too** (Tom, 2026-10-22; TOM-1979):
the allocations-to-orders pipeline rounds a target to its ``lot_size`` and its
``qty_step`` with :func:`floor_to_multiple`, longs and shorts alike. A rounded
target never exceeds the size the weight asked for.

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


def amount_decimals(precision: Any) -> int:
    """A ccxt ``precision["amount"]`` as a number of decimal places.

    ccxt gives either a step (a float, e.g. ``0.001``, or ``1.0`` for a whole-unit
    coin) or a number of decimals (an int, e.g. ``3``). A step up to 1 becomes
    its decimal places (``1.0`` → 0); anything else is read as a number of
    decimals. Reading ``1.0`` as one decimal left a whole-unit order off the
    grid, and ccxt then rounded it to the nearest unit (TOM-1721).
    """
    if isinstance(precision, float) and 0 < precision <= 1:
        return max(0, -int(math.floor(math.log10(precision))))
    return int(precision)


def floor_to_multiple(qty: float, step: float) -> float:
    """*qty* rounded toward zero to a whole multiple of *step*, in decimal arithmetic.

    Toward zero for a short too: ``-0.25`` on a ``0.1`` step is ``-0.2``, never
    ``-0.3`` (TOM-1979). Both floats are read through ``str``, as
    :func:`floor_to_decimals` does, so an on-grid quantity is returned unchanged:
    ``0.3`` on a ``0.1`` step stays ``0.3`` (float division gives ``2.9999…``).
    A step of 0 or less, or a quantity that is not finite, returns *qty*.
    """
    if step <= 0 or not math.isfinite(qty):
        return qty
    d_step = Decimal(str(step))
    lots = (Decimal(str(qty)) / d_step).to_integral_value(rounding=ROUND_DOWN)
    return float(lots * d_step) + 0.0  # + 0.0 turns -0.0 into 0.0


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
