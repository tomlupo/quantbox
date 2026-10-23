"""A rounded quantity never exceeds the requested size, long or short (TOM-1979).

Tom, 2026-10-22: targets round toward zero, as orders do since TOM-1721. This is
a property test of pure logic that real books rarely reach: a short target, a
quantity just under a lot boundary, and a step that float division gets wrong
(``0.3 / 0.1 == 2.9999999999999996``). A seeded randomized loop stands in for
hypothesis, which is not a dev dependency.

The order rounding is :class:`quantbox.trade.lots.LotRules` (TOM-2206). Until
TOM-2208 these properties were asserted on the free function
``floor_to_multiple``; it had no caller left when ``binance_live`` was deleted
(Tom, 2026-10-23), so they are asserted on ``LotRules`` itself. ``test_lot_rules.py``
asserts the same property per venue on power-of-ten steps; the steps here also
include 0.25, 0.5, 5 and 100.
"""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from quantbox.plugins.pipeline.alloc2orders import _apply_qty_rules
from quantbox.trade.lots import LotRules

STEPS = (0.00001, 0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 5.0, 100.0)
N_CASES = 2000


def _floor(qty: float, step: float) -> float:
    """*qty* rounded by the lot rules of an instrument with *step* and no minimums."""
    return LotRules(step=Decimal(str(step))).accept(qty).rounded


def _quantities(seed: int) -> list[float]:
    """Random sizes of both signs and several magnitudes, plus the edges a loop rarely draws."""
    rng = random.Random(seed)
    qtys = [0.0, 0.3, -0.3, 0.7, -0.7, 1.6, -1.6, 2.5, -2.5, 0.25, -0.25, 1e-9, -1e-9]
    for _ in range(N_CASES):
        magnitude = 10 ** rng.uniform(-4, 4)
        qtys.append(rng.choice((1.0, -1.0)) * round(magnitude, rng.randint(0, 9)))
    return qtys


def _within(rounded: float, requested: float) -> bool:
    """``rounded`` is on ``requested``'s side of zero and no larger in absolute value."""
    if rounded == 0:
        return True
    return (rounded > 0) == (requested > 0) and abs(rounded) <= abs(requested)


@pytest.mark.parametrize("step", STEPS)
def test_the_lot_rules_never_exceed_the_request(step):
    bad = [(q, _floor(q, step)) for q in _quantities(1979) if not _within(_floor(q, step), q)]
    assert not bad, f"step {step}: {bad[:5]}"


@pytest.mark.parametrize("lot_size", (0.0, 0.1, 1.0, 5.0))
@pytest.mark.parametrize("qty_step", (0.0, 0.001, 0.25))
def test_a_target_never_exceeds_the_requested_size(lot_size, qty_step):
    """``_apply_qty_rules`` rounds a TARGET position: toward zero for longs and shorts."""
    bad = []
    for q in _quantities(22):
        rounded = _apply_qty_rules(q, lot_size, 0.0, qty_step)
        if not _within(rounded, q):
            bad.append((q, rounded))
    assert not bad, f"lot {lot_size}, step {qty_step}: {bad[:5]}"


@pytest.mark.parametrize("step", STEPS)
def test_an_on_grid_quantity_is_unchanged(step):
    """A quantity already on the grid survives, so a full close or an exact target is never cut short."""
    rng = random.Random(10)
    d_step = Decimal(str(step))
    for _ in range(500):
        n = rng.randint(-100_000, 100_000)
        on_grid = float(n * d_step)
        assert _floor(on_grid, step) == on_grid, (n, step)


def test_the_lot_rules_read_the_decimal_not_the_binary_float():
    """``0.3 / 0.1`` is ``2.999...`` in binary floats; the lot rules keep ``0.3`` (was asserted on floor_to_step)."""
    assert _floor(0.3, 0.1) == 0.3
    assert _floor(0.29, 0.01) == 0.29


def test_a_rounded_quantity_is_a_whole_multiple_of_the_step():
    for step in STEPS:
        d_step = Decimal(str(step))
        for q in _quantities(3):
            rounded = Decimal(str(_floor(q, step)))
            assert rounded % d_step == 0, (q, step, rounded)
