"""One lot rule: a venue order quantity is rounded DOWN, never up (TOM-1721).

Tom, 2026-10-10 (P5 grilling Q5): round down on every venue. Hyperliquid used to
round to the NEAREST step, so an order could exceed its target or the cash that
funds it. The property runs through the real ``HyperliquidBroker.place_order``;
only the venue boundary (ccxt ``create_order`` and the price fetch) is faked.

Two properties, for any step and precision:

* the quantity sent never exceeds the quantity requested, and it is less than one
  step below it;
* a quantity already on the venue grid (a full close of a venue position, whose
  size the venue reports on that grid) is sent whole: flooring leaves no dust.
"""

from __future__ import annotations

import random
from decimal import Decimal
from types import SimpleNamespace

import pytest

from quantbox.plugins.broker.binance_futures import adjust_quantity
from quantbox.plugins.broker.hyperliquid import HyperliquidBroker

SEED = 1721
N_RANDOM = 400

# ccxt gives precision["amount"] either as a step (float < 1) or as a number of
# decimals (int). Both forms, with the edges: 0 decimals, 1, a fine step.
PRECISIONS = [0, 1, 2, 3, 4, 5, 8, 0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-08]


def _decimals(precision) -> int:
    """Independent of quantbox: a step's decimal places, or the int as given."""
    if isinstance(precision, float):
        return -Decimal(repr(precision)).normalize().as_tuple().exponent
    return int(precision)


class _Venue:
    """The ccxt boundary: records the amount each order is sent with."""

    def __init__(self):
        self.sent: list[float] = []

    def create_order(self, symbol, type, side, amount, price=None, params=None):  # noqa: A002 - ccxt's name
        self.sent.append(amount)
        return {"id": str(len(self.sent)), "status": "closed", "filled": amount, "amount": amount}


def _broker(precision) -> tuple[HyperliquidBroker, _Venue]:
    venue = _Venue()
    b = object.__new__(HyperliquidBroker)
    b._exchange = venue
    b._markets = {"X/USDC:USDC": {"precision": {"amount": precision}, "limits": {"amount": {"min": None}}}}
    b._get_market_symbol = lambda s: "X/USDC:USDC"
    b.get_price = lambda s: 1.0
    b.risk = SimpleNamespace(min_order_notional=0.0)
    return b, venue


def _sent(precision, qty, reduce_only=False):
    b, venue = _broker(precision)
    b.place_order("X", "sell" if reduce_only else "buy", qty, reduce_only=reduce_only)
    return venue.sent[-1] if venue.sent else None


def _cases():
    rng = random.Random(SEED)
    cases = [
        # Edges: the nearest step is ABOVE the request (the old round-up), exactly
        # on the grid, half a step, just below a grid point, large, tiny.
        (3, 0.0156),
        (0.001, 0.0156),
        (0, 2.6),
        (1, 0.25),
        (2, 0.29),
        (1, 0.3),
        (3, 0.0159999),
        (0.0001, 12345.67895),
        (5, 1.234565),
        (8, 0.123456789),
    ]
    for _ in range(N_RANDOM):
        precision = rng.choice(PRECISIONS)
        qty = rng.uniform(0, 10) * 10 ** rng.randint(-4, 4)
        cases.append((precision, qty))
    return cases


CASES = _cases()


def test_the_property_ran_on_every_case():
    print(f"seed={SEED} cases={len(CASES)}")
    assert len(CASES) == 10 + N_RANDOM


@pytest.mark.parametrize(("precision", "qty"), CASES)
def test_hyperliquid_never_sends_more_than_requested(precision, qty):
    sent = _sent(precision, qty)
    step = 10.0 ** -_decimals(precision)
    if sent is None:  # floored to zero: refused, never rounded up to one step
        assert qty < step
        return
    assert sent <= qty
    assert qty - sent < step + 1e-12


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("units", [1, 7, 153, 9999, 123456])
def test_a_full_close_on_the_grid_is_sent_whole(precision, units):
    # A venue position is reported on the venue's grid: k steps exactly.
    size = float(Decimal(units).scaleb(-_decimals(precision)))
    assert _sent(precision, size, reduce_only=True) == size


@pytest.mark.parametrize(("precision", "qty"), CASES)
def test_binance_futures_with_no_known_step_never_rounds_up(precision, qty):
    assert adjust_quantity(qty, 0.0) <= qty


@pytest.mark.parametrize("units", [1, 2, 7, 12, 153, 9999])
@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.7, 0.999])
def test_a_whole_unit_step_sends_whole_units(units, fraction):
    # ccxt (TICK_SIZE) gives a Hyperliquid coin with szDecimals 0 a step of 1.0,
    # a float. Read as "1 decimal" the order leaves quantbox off the grid, and
    # ccxt rounds it to the NEAREST unit on send: 12.7 goes out as 13.
    qty = units + fraction
    sent = _sent(1.0, qty)
    assert sent == float(units)
