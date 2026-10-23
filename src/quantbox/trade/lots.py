"""Lot rules: what order quantity a venue accepts (TOM-1458, TOM-2206).

A venue's **lot rules** are its quantity step, its minimum quantity, its
minimum notional and whether a reduce-only order is exempt from both minimums.
:class:`LotRules` is the one module that knows them. A caller builds the rules
from the venue's market data and asks one question::

    rules = LotRules.from_market(market, "hyperliquid")
    lot = rules.accept(qty, price, reduce_only=False)
    if lot.skip:            # the venue would not accept an order: lot.skip says why
        ...
    send(lot.qty)           # the signed quantity the venue accepts

What :meth:`LotRules.accept` does, in this order, for every venue:

1. **Round toward zero** to a whole multiple of the step, longs and shorts
   alike (Tom, 2026-10-10, TOM-1721; targets too, Tom, 2026-10-22, TOM-1979).
   An answer never exceeds the quantity asked. A quantity already on the grid
   (a full close of a position the venue reports) is returned whole.
2. **Minimum notional**, on the ROUNDED quantity. Before TOM-2206 Hyperliquid
   judged the notional of the unrounded quantity, so an order just above the
   minimum was rounded below it and sent.
3. **Minimum quantity.**
4. A quantity that rounds to zero is not an order.

A reduce-only order skips steps 2 and 3 on a venue whose rules exempt it (the
perps venues): a position-closing order is never blocked by a minimum-size
filter (quantbox#87). Rounding still applies, so it never flips a position.

Venue facts that are not in the market data live in ``_VENUES`` here and
nowhere else: the default minimum notional when the venue reports none, the
reduce-only exemption, and the step used when no step is known. A broker names
its venue with the class attribute ``lot_venue``; :func:`lot_venue` reads it, so
an order generator that holds only the broker and its market snapshot builds the
same rules the broker sends with (TOM-2207).

The free functions below the class are the arithmetic the class uses, kept for
the callers not yet moved onto :class:`LotRules` (``binance_live``, TOM-2208).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any

logger = logging.getLogger(__name__)

#: Digits a step size is written with when its decimal places are counted.
_STEP_DIGITS = 8

#: Skip reasons :meth:`LotRules.accept` answers with.
SKIP_NOT_FINITE = "not a finite quantity"
SKIP_MIN_NOTIONAL = "below min notional"
SKIP_MIN_QTY = "below min qty"
SKIP_ZERO = "rounds to zero"


@dataclass(frozen=True)
class _Venue:
    default_min_notional: float
    reduce_only_exempt: bool
    unknown_step: Decimal | None


#: The venue facts the market data does not carry. ccxt gives no step for an
#: unknown market; the brokers then floored to 8 decimals.
_VENUES: dict[str, _Venue] = {
    "hyperliquid": _Venue(default_min_notional=10.0, reduce_only_exempt=True, unknown_step=Decimal("1E-8")),
    "binance_futures": _Venue(default_min_notional=5.0, reduce_only_exempt=True, unknown_step=Decimal("1E-8")),
    "kraken": _Venue(default_min_notional=0.0, reduce_only_exempt=False, unknown_step=Decimal("1E-8")),
}


def lot_venue(broker: object | None) -> str | None:
    """The venue *broker* declares in ``lot_venue``, or ``None`` when it declares none we know.

    ``None`` means: no venue facts. :meth:`LotRules.from_snapshot` then gives no
    reduce-only exemption and no fallback step.
    """
    venue = getattr(broker, "lot_venue", None)
    if venue is None:
        return None
    if venue not in _VENUES:
        logger.warning(
            "Broker %s declares lot_venue=%r, which is not one of %s", type(broker).__name__, venue, sorted(_VENUES)
        )
        return None
    return str(venue)


@dataclass(frozen=True)
class Lot:
    """The answer of :meth:`LotRules.accept`.

    ``qty`` is the signed quantity the venue accepts, ``0.0`` when skipped.
    ``rounded`` is the signed quantity after rounding, before the minimums: a
    caller that reports what a skipped order would have been reads it.
    ``skip`` is ``None`` for an order, else one of the ``SKIP_*`` reasons.
    """

    qty: float
    rounded: float
    skip: str | None = None


@dataclass(frozen=True)
class LotRules:
    """One venue's lot rules for one instrument. Build with :meth:`from_market` or :meth:`from_snapshot`."""

    step: Decimal | None = None
    min_qty: float = 0.0
    min_notional: float = 0.0
    reduce_only_exempt: bool = False
    #: The step used when ``step`` is None. Not reported as the instrument's step.
    fallback_step: Decimal | None = None
    #: The venue's own rounding of a positive quantity, when the caller keeps it
    #: (Kraken: ccxt ``amount_to_precision``). It replaces step rounding; if it
    #: raises, the step rounding is used.
    quantize: Callable[[float], Any] | None = None

    # -- construction ------------------------------------------------------

    @classmethod
    def from_market(
        cls,
        market: Mapping[str, Any] | None,
        venue: str,
        *,
        quantize: Callable[[float], Any] | None = None,
    ) -> LotRules:
        """The rules in a ccxt market dict, with *venue*'s facts from ``_VENUES``.

        ccxt gives ``precision["amount"]`` as a step (a float: ``0.001``, or
        ``1.0`` for a whole-unit coin, ``TICK_SIZE`` mode) or as a number of
        decimals (an int: ``3``). Missing ``limits.cost.min`` falls back to the
        venue's default minimum notional.
        """
        v = _VENUES[venue]
        market = market or {}
        limits = market.get("limits") or {}
        precision = market.get("precision") or {}
        cost_min = _positive((limits.get("cost") or {}).get("min"))
        return cls(
            step=_ccxt_step(precision.get("amount")),
            min_qty=_positive((limits.get("amount") or {}).get("min")),
            min_notional=cost_min if cost_min > 0 else v.default_min_notional,
            reduce_only_exempt=v.reduce_only_exempt,
            fallback_step=v.unknown_step,
            quantize=quantize,
        )

    @classmethod
    def from_snapshot(
        cls,
        row: Mapping[str, Any],
        *,
        venue: str | None = None,
        default_min_notional: float | None = None,
    ) -> LotRules:
        """The rules in a market-snapshot row: ``step_size`` (a step), ``min_qty``, ``min_notional``.

        A row without a positive ``min_notional`` takes *default_min_notional*,
        else the *venue*'s default. Without a *venue* there is no reduce-only
        exemption and no step when the row has none (the quantity is not rounded).
        """
        v = _VENUES[venue] if venue else None
        step = _positive(row.get("step_size"))
        mn = _positive(row.get("min_notional"))
        if mn <= 0:
            if default_min_notional is not None:
                mn = float(default_min_notional)
            elif v is not None:
                mn = v.default_min_notional
        return cls(
            step=Decimal(str(step)) if step > 0 else None,
            min_qty=_positive(row.get("min_qty")),
            min_notional=mn,
            reduce_only_exempt=bool(v and v.reduce_only_exempt),
            fallback_step=v.unknown_step if v else None,
        )

    # -- the question ------------------------------------------------------

    @property
    def step_size(self) -> float:
        """The instrument's step as a float for a snapshot row; ``0.0`` when none is known."""
        return float(self.step) if self.step is not None else 0.0

    def accept(self, qty: float, price: float | None = None, *, reduce_only: bool = False) -> Lot:
        """The signed quantity the venue accepts for *qty*, or the reason it accepts none.

        *price* prices the minimum notional. ``None`` means the caller sends the
        order unpriced (a Kraken market order): the minimum notional is then left
        to the venue. *reduce_only* exempts the order from both minimums on a
        venue whose rules say so. See the module docstring for the order of steps.
        """
        if qty is None or not math.isfinite(qty):
            return Lot(0.0, 0.0, SKIP_NOT_FINITE)
        sign = -1.0 if qty < 0 else 1.0
        size = self._round(abs(qty))
        rounded = sign * size + 0.0  # + 0.0 turns -0.0 into 0.0
        if not (reduce_only and self.reduce_only_exempt):
            if self.min_notional > 0 and price is not None and size * price < self.min_notional:
                return Lot(0.0, rounded, SKIP_MIN_NOTIONAL)
            if self.min_qty > 0 and size < self.min_qty:
                return Lot(0.0, rounded, SKIP_MIN_QTY)
        if size <= 0:
            return Lot(0.0, rounded, SKIP_ZERO)
        return Lot(rounded, rounded)

    def _round(self, size: float) -> float:
        step = self.step if self.step is not None else self.fallback_step
        if self.quantize is not None:
            try:
                return float(self.quantize(size))
            except Exception as e:  # noqa: BLE001 - the venue refused to round: fall back to the step
                logger.warning("Venue rounding failed for %s (%s); flooring to the step", size, e)
        if step is None:
            return size
        return _floor_to(size, step)


def _positive(value: Any) -> float:
    """*value* as a float when it is a positive finite number, else ``0.0``."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) and f > 0 else 0.0


def _ccxt_step(precision: Any) -> Decimal | None:
    """A ccxt ``precision["amount"]`` as a step: an int is a number of decimals, a float is the step."""
    if precision is None or isinstance(precision, bool):
        return None
    if isinstance(precision, int):
        return Decimal(1).scaleb(-precision) if precision >= 0 else None
    p = _positive(precision)
    return Decimal(str(p)) if p > 0 else None


def _floor_to(qty: float, step: Decimal) -> float:
    """*qty* rounded toward zero to a whole multiple of *step*, in decimal arithmetic."""
    if step <= 0 or not math.isfinite(qty):
        return qty
    lots = (Decimal(str(qty)) / step).to_integral_value(rounding=ROUND_DOWN)
    return float(lots * step) + 0.0  # + 0.0 turns -0.0 into 0.0


# ---------------------------------------------------------------------------
# Arithmetic for callers not yet on LotRules (binance_live, TOM-2208)
# ---------------------------------------------------------------------------


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
    (``0.5`` floors to one decimal). :class:`LotRules` rounds to multiples.
    """
    if step <= 0:
        return qty
    return floor_to_decimals(qty, step_decimals(step))


def floor_to_multiple(qty: float, step: float) -> float:
    """*qty* rounded toward zero to a whole multiple of *step*, in decimal arithmetic.

    Toward zero for a short too: ``-0.25`` on a ``0.1`` step is ``-0.2``, never
    ``-0.3`` (TOM-1979). Both floats are read through ``str``, so an on-grid
    quantity is returned unchanged: ``0.3`` on a ``0.1`` step stays ``0.3``.
    A step of 0 or less, or a quantity that is not finite, returns *qty*.
    """
    if step <= 0 or not math.isfinite(qty):
        return qty
    return _floor_to(qty, Decimal(str(step)))


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
