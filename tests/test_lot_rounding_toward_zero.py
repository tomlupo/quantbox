"""A rounded quantity never exceeds the requested size, long or short (TOM-1979).

Tom, 2026-10-22: targets round toward zero, as orders do since TOM-1721. This is
a property test of pure logic that real books rarely reach: a short target, a
quantity just under a lot boundary, and a step that float division gets wrong
(``0.3 / 0.1 == 2.9999999999999996``). A seeded randomized loop stands in for
hypothesis, which is not a dev dependency.
"""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from quantbox.plugins.pipeline.alloc2orders import _apply_qty_rules
from quantbox.trade.lots import floor_to_decimals, floor_to_multiple, floor_to_step

STEPS = (0.00001, 0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 5.0, 100.0)
N_CASES = 2000


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
def test_floor_to_multiple_never_exceeds_the_request(step):
    bad = [(q, floor_to_multiple(q, step)) for q in _quantities(1979) if not _within(floor_to_multiple(q, step), q)]
    assert not bad, f"step {step}: {bad[:5]}"


@pytest.mark.parametrize("step", STEPS)
def test_floor_to_step_never_exceeds_the_request(step):
    bad = [(q, floor_to_step(q, step)) for q in _quantities(1721) if not _within(floor_to_step(q, step), q)]
    assert not bad, f"step {step}: {bad[:5]}"


@pytest.mark.parametrize("decimals", (0, 1, 3, 8))
def test_floor_to_decimals_never_exceeds_the_request(decimals):
    bad = [
        (q, floor_to_decimals(q, decimals)) for q in _quantities(8) if not _within(floor_to_decimals(q, decimals), q)
    ]
    assert not bad, f"decimals {decimals}: {bad[:5]}"


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
        assert floor_to_multiple(on_grid, step) == on_grid, (n, step)


def test_a_rounded_quantity_is_a_whole_multiple_of_the_step():
    for step in STEPS:
        d_step = Decimal(str(step))
        for q in _quantities(3):
            rounded = Decimal(str(floor_to_multiple(q, step)))
            assert rounded % d_step == 0, (q, step, rounded)
