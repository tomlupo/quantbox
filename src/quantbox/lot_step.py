"""A venue's quantity step, read from its precision (TOM-2206).

A venue states its quantity step in one of two forms. ccxt calls both
``precision["amount"]``:

- a **number of decimals**, an int: ``3`` means a step of ``0.001``. Hyperliquid
  ``szDecimals`` is this form.
- a **step**, a float: ``0.001``, or ``1.0`` for a whole-unit coin (ccxt
  ``TICK_SIZE`` mode).

:func:`lot_step` reads either form. It is core. The plugin-layer universe guard
(:mod:`quantbox.universe`) and the trade-layer lot rules
(:mod:`quantbox.trade.lots`) both call it. So a plugin never imports
:mod:`quantbox.trade`.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any


def lot_step(precision: Any) -> Decimal | None:
    """The quantity step in *precision*: an int is a number of decimals, a float is the step.

    ``None`` when *precision* states no step: ``None``, a bool, a negative
    number of decimals, or a step that is not a positive finite number.

    The step is exact. A number of decimals ``n`` gives ``Decimal("1E-n")``; a
    float step is read through its shortest repr (``str``), so ``0.001`` gives
    ``Decimal("0.001")`` and not the binary value of the float.
    """
    if precision is None or isinstance(precision, bool):
        return None
    if isinstance(precision, int):
        return Decimal(1).scaleb(-precision) if precision >= 0 else None
    try:
        step = float(precision)
    except (TypeError, ValueError):
        return None
    return Decimal(str(step)) if math.isfinite(step) and step > 0 else None
