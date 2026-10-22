"""One property of a venue's lot rules, on every venue, longs and shorts (TOM-2206).

The property, for an order of signed quantity ``q`` at price ``p``:

* the quantity answered is on ``q``'s side of zero, never larger than ``|q|``,
  less than one step below it, and a whole multiple of the venue's step;
* it is never below the venue's minimum notional or minimum quantity, unless
  the order is reduce-only on a venue that exempts it;
* otherwise the answer is a skip, and a skip carries a reason;
* an on-grid quantity (a full close of a position the venue reports) is sent
  whole when it clears the minimums, and always when it is an exempt
  reduce-only order.

The property runs twice. ``test_the_property_through_the_lot_rules`` asks
:class:`quantbox.trade.lots.LotRules` directly. ``test_the_property_at_each_venue_entry``
asks each venue's real entry point (``place_order``, Kraken's ``_place_one``,
the standard rebalancer's order frame) with only the venue boundary faked; it
imports nothing new, so it also runs on the base commit and shows where the
base disagrees. Kraken's rounding stays ccxt's own ``amount_to_precision``,
called on a real ``ccxt.kraken`` object; ``test_kraken_rounding_equals_the_lot_rules_floor``
proves the lot rules' decimal floor gives the same quantity.

Kraken sends a market order unpriced, so its minimum notional is not judged at
placement (the rebalancer judges it, the snapshot form below). The property
asserts no notional minimum for Kraken. A broker entry that skips returns
``None``; its reason is in the log only, so the reason clause is asserted for
the lot rules, Kraken and the rebalancer.

A seeded randomized loop stands in for hypothesis, which is not a dev dependency.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

SEED = 2206
N_RANDOM = 500
STEPS = ("0.00000001", "0.00001", "0.001", "0.1", "1")
VENUES = ("hyperliquid", "binance_futures", "kraken", "snapshot")
#: The minimum notional each venue's entry enforces in these markets.
MIN_NOTIONAL = {"hyperliquid": 10.0, "binance_futures": 5.0, "kraken": 0.0, "snapshot": 1.0}
#: Kraken's market carries a cost.min; the venue judges it, not the broker.
KRAKEN_COST_MIN = 0.5
PERPS = ("hyperliquid", "binance_futures")


@dataclass(frozen=True)
class Case:
    venue: str
    step: Decimal
    min_qty: float
    price: float
    qty: float  # signed
    reduce_only: bool
    on_grid: bool

    @property
    def exempt(self) -> bool:
        return self.reduce_only and self.venue in PERPS

    @property
    def min_notional(self) -> float:
        return MIN_NOTIONAL[self.venue]

    def clears_minimums(self, size: float) -> bool:
        return size * self.price >= self.min_notional and size >= self.min_qty


def _cases(venue: str) -> list[Case]:
    rng = random.Random(f"{SEED}-{venue}")
    out: list[Case] = []
    for i in range(N_RANDOM):
        step = Decimal(rng.choice(STEPS))
        price = 10 ** rng.uniform(-3, 5)
        min_qty = float(step * rng.choice((0, 1, 3)))
        kind = ("random", "edge", "grid", "qty_edge")[i % 4]
        if kind == "grid":
            size = float(step * rng.randint(1, 20_000))
        elif kind == "edge" and MIN_NOTIONAL[venue] > 0:
            # Just above the minimum notional: the round-down can take it below.
            size = MIN_NOTIONAL[venue] * (1 + rng.uniform(0, 0.05)) / price
        elif kind == "qty_edge":
            # Below a minimum quantity of three steps, priced above the minimum notional.
            min_qty = float(step * 3)
            size = float(step) * rng.uniform(1, 2.99)
            price = 2 * max(MIN_NOTIONAL[venue], 1.0) / size
        else:
            size = 10 ** rng.uniform(-1, 3.5) / price
        sign = rng.choice((1.0, -1.0))
        reduce_only = venue in PERPS and rng.random() < 0.5
        out.append(Case(venue, step, min_qty, price, sign * size, reduce_only, kind == "grid"))
    return out


CASES = [c for v in VENUES for c in _cases(v)]


def _violations(case: Case, answer: float | None, reason: str | None, *, reasons_known: bool) -> list[str]:
    q, size = case.qty, abs(case.qty)
    if answer is None or answer == 0:
        bad = []
        if reasons_known and not reason:
            bad.append("skipped without a reason")
        if case.on_grid and (case.exempt or case.clears_minimums(size)):
            bad.append("an on-grid order that the venue accepts was skipped")
        return bad
    bad = []
    a = abs(answer)
    if (answer > 0) != (q > 0):
        bad.append(f"sign flipped: {answer}")
    if a > size:
        bad.append(f"answer {a} larger than asked {size}")
    if size - a >= float(case.step) * (1 + 1e-9):
        bad.append(f"answer {a} is a step or more below {size}")
    if Decimal(str(a)) % case.step != 0:
        bad.append(f"answer {a} is off the {case.step} grid")
    if not case.exempt and not case.clears_minimums(a):
        bad.append(f"answer {a} (${a * case.price:.4f}) below the minimums")
    if case.on_grid and a != size:
        bad.append(f"on-grid {size} not sent whole: {a}")
    return bad


# ---------------------------------------------------------------------------
# The market data each venue is built from
# ---------------------------------------------------------------------------


def _ccxt_market(case: Case, symbol: str, cost_min: float) -> dict:
    return {
        "id": symbol.replace("/", ""),
        "symbol": symbol,
        "base": "X",
        "quote": symbol.split("/")[1].split(":")[0],
        "type": "spot" if case.venue == "kraken" else "swap",
        "spot": case.venue == "kraken",
        "active": True,
        "precision": {"amount": float(case.step), "price": 0.01},
        "limits": {"amount": {"min": case.min_qty or None}, "cost": {"min": cost_min}},
    }


def _snapshot_row(case: Case) -> dict:
    return {"symbol": "X", "step_size": float(case.step), "min_qty": case.min_qty, "min_notional": case.min_notional}


# ---------------------------------------------------------------------------
# Answer 1: the lot rules, asked directly
# ---------------------------------------------------------------------------


def _rules_answer(case: Case) -> tuple[float | None, str | None]:
    from quantbox.trade.lots import LotRules

    if case.venue == "snapshot":
        rules = LotRules.from_snapshot(_snapshot_row(case), default_min_notional=1.0)
    elif case.venue == "kraken":
        rules = LotRules.from_market(_ccxt_market(case, "X/USD", KRAKEN_COST_MIN), "kraken")
    else:
        rules = LotRules.from_market(_ccxt_market(case, "X/USDC:USDC", case.min_notional), case.venue)
    price = None if case.venue == "kraken" else case.price
    lot = rules.accept(case.qty, price, reduce_only=case.reduce_only)
    return (None, lot.skip) if lot.skip else (lot.qty, None)


# ---------------------------------------------------------------------------
# Answer 2: each venue's real entry point, the venue boundary faked
# ---------------------------------------------------------------------------


class _Venue:
    """The ccxt boundary: records the amount an order is sent with."""

    def __init__(self, kraken=None):
        self.sent: list[float] = []
        self._kraken = kraken
        if kraken is not None:
            self.amount_to_precision = kraken.amount_to_precision

    def create_order(self, symbol=None, type=None, side=None, amount=None, price=None, params=None):  # noqa: A002
        self.sent.append(amount)
        return {"id": str(len(self.sent)), "status": "closed", "filled": amount, "amount": amount}


def _side(case: Case) -> str:
    return "buy" if case.qty > 0 else "sell"


def _signed(case: Case, amount: float | None) -> float | None:
    return None if amount is None else (amount if case.qty > 0 else -amount)


def _hyperliquid(case: Case):
    from quantbox.plugins.broker.hyperliquid import HyperliquidBroker

    venue = _Venue()
    b = object.__new__(HyperliquidBroker)
    b._exchange = venue
    b._markets = {"X/USDC:USDC": _ccxt_market(case, "X/USDC:USDC", case.min_notional)}
    b._get_market_symbol = lambda s: "X/USDC:USDC"
    b.get_price = lambda s: case.price
    b.risk = SimpleNamespace(min_order_notional=case.min_notional)
    b.place_order("X", _side(case), abs(case.qty), reduce_only=case.reduce_only)
    return _signed(case, venue.sent[-1] if venue.sent else None), None


def _binance_futures(case: Case):
    from quantbox.plugins.broker.binance_futures import BinanceFuturesBroker

    venue = _Venue()
    b = BinanceFuturesBroker.__new__(BinanceFuturesBroker)
    b._exchange = venue
    b._markets = {"X/USDT:USDT": _ccxt_market(case, "X/USDT:USDT", case.min_notional)}
    b.quote_currency = "USDT"
    b.target_leverage = 1
    b.set_leverage = lambda s, lev: True
    b.get_price = lambda s: case.price
    b.place_order("X", _side(case), abs(case.qty), reduce_only=case.reduce_only)
    return _signed(case, venue.sent[-1] if venue.sent else None), None


def _kraken_exchange(case: Case):
    ccxt = pytest.importorskip("ccxt")
    ex = ccxt.kraken()
    ex.set_markets([_ccxt_market(case, "X/USD", KRAKEN_COST_MIN)])
    return ex


def _kraken(case: Case):
    from quantbox.plugins.broker.kraken import KrakenBroker

    venue = _Venue(kraken=_kraken_exchange(case))
    b = object.__new__(KrakenBroker)
    b._exchange = venue
    b._markets = {"X/USD": _ccxt_market(case, "X/USD", KRAKEN_COST_MIN)}
    b._market_symbol = lambda s: "X/USD"
    result = b._place_one("X", _side(case), abs(case.qty), None)
    if venue.sent:
        return _signed(case, venue.sent[-1]), None
    return None, "below exchange minimum (skipped)" if result is not None else None


class _SnapshotBroker:
    def __init__(self, row):
        self.row = row

    def get_market_snapshot(self, symbols):
        return pd.DataFrame([self.row])


def _snapshot(case: Case):
    from quantbox.plugins.rebalancing.standard_rebalancer import StandardRebalancer

    buy = case.qty > 0
    frame = pd.DataFrame(
        [
            {
                "Asset": "X",
                "Trade Action": "Buy" if buy else "Sell",
                "Delta Quantity": case.qty,
                "Price": case.price,
                "Weight Delta": 1.0 if buy else -1.0,
                "Target Weight": 1.0 if buy else 0.0,
                "Current Quantity": 0.0 if buy else abs(case.qty),
                "Target Quantity": abs(case.qty) if buy else 0.0,
            }
        ]
    )
    orders = StandardRebalancer()._create_executable_orders(
        rebalancing_df=frame,
        broker=_SnapshotBroker(_snapshot_row(case)),
        stable_coin="USDC",
        min_notional_default=1.0,
        min_trade_size=0.0,
        cash_available=1e18,
        scaling_factor_min=0.9,
    )
    row = orders.iloc[0]
    if bool(row["Executable"]):
        return _signed(case, float(row["Adjusted Quantity"])), None
    return None, str(row["Order Status"])


ENTRIES = {"hyperliquid": _hyperliquid, "binance_futures": _binance_futures, "kraken": _kraken, "snapshot": _snapshot}


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------


def test_the_property_ran_on_every_venue():
    print(f"seed={SEED} cases={len(CASES)}")
    assert len(CASES) == N_RANDOM * len(VENUES)
    for v in VENUES:
        mine = [c for c in CASES if c.venue == v]
        assert any(c.qty < 0 for c in mine) and any(c.qty > 0 for c in mine), v
        assert any(c.on_grid for c in mine), v
    assert any(c.reduce_only for c in CASES if c.venue in PERPS)


def _report(venue: str, answer) -> list:
    bad = []
    for case in (c for c in CASES if c.venue == venue):
        a, reason = answer(case)
        for v in _violations(case, a, reason, reasons_known=answer is _rules_answer or venue in ("kraken", "snapshot")):
            bad.append((v, case))
    return bad


@pytest.mark.parametrize("venue", VENUES)
def test_the_property_through_the_lot_rules(venue):
    bad = _report(venue, _rules_answer)
    assert not bad, f"{len(bad)} violation(s), first: {bad[:3]}"


@pytest.mark.parametrize("venue", VENUES)
def test_the_property_at_each_venue_entry(venue):
    bad = _report(venue, ENTRIES[venue])
    assert not bad, f"{len(bad)} violation(s), first: {bad[:3]}"


@pytest.mark.parametrize("venue", VENUES)
def test_the_entry_answers_what_the_lot_rules_answer(venue):
    """Every venue entry sends exactly the quantity its lot rules accept, or nothing."""
    diff = []
    for case in (c for c in CASES if c.venue == venue):
        entry, _ = ENTRIES[venue](case)
        rules, _ = _rules_answer(case)
        if (entry or 0.0) != (rules or 0.0):
            diff.append((entry, rules, case))
    assert not diff, f"{len(diff)} difference(s), first: {diff[:3]}"


def test_kraken_rounding_equals_the_lot_rules_floor():
    """ccxt's amount_to_precision and the lot rules' decimal floor agree on every Kraken case."""
    from quantbox.trade.lots import LotRules

    diff = []
    for case in (c for c in CASES if c.venue == "kraken"):
        ex = _kraken_exchange(case)
        try:
            ccxt_qty = float(ex.amount_to_precision("X/USD", abs(case.qty)))
        except Exception:  # noqa: BLE001 - ccxt refuses a quantity that truncates to zero
            ccxt_qty = 0.0
        floor_qty = LotRules(step=case.step).accept(abs(case.qty)).rounded
        if ccxt_qty != floor_qty:
            diff.append((case.qty, case.step, ccxt_qty, floor_qty))
    assert not diff, f"{len(diff)} difference(s), first: {diff[:5]}"
