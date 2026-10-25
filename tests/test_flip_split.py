"""A flip through zero is a reduce-only CLOSE leg, then an OPEN leg (TOM-2396).

Incident replay: carver-HL paged REBALANCER FROZEN on 2026-10-24 and
2026-10-25. Prod run ``2026-10-25__trade_full_pipeline_v1__1e17c1129a76__20261025T050028Z``
held an ETH long of 0.0032 ($7.57) with a short target of -0.000587 ($1.39).
The rebalancer wrote ONE sell of the whole delta, 0.003787 ($8.76). That row
was neither a full close nor a partial reduce, so it was not reduce-only, and
the $10 floor suppressed it. The executor read a suppressed declared exit and
froze the book. Hyperliquid accepts a reduce-only full close below $10 (SOL,
$2.33, 2026-07-22), so the exit was available.

The guard below is that run's own rebalancing frame and lot rules. It imports
only names the base commit has, so at the base it fails on its assertions
(frozen), not on an import.

The property test walks flips of both signs, legs above and below the floor,
and dust closes under a Kraken-like lot rule (no reduce-only exemption). A
seeded randomized loop stands in for hypothesis, which is not a dev dependency.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest

from quantbox.plugins.rebalancing.futures_rebalancer import RESIDUAL_STATUS, FuturesRebalancer
from quantbox.trade.executor import execute_orders
from quantbox.trade.lots import LotRules

# --- the real 2026-10-25 carver-HL run ------------------------------------------

#: rebalancing.parquet of the prod run, verbatim (3 rows).
CARVER_1025_REBALANCING = [
    {
        "Asset": "BTC",
        "Current Quantity": 0.00033,
        "Current Value": 26.691555,
        "Current Weight": 0.378854,
        "Target Quantity": 0.000251,
        "Target Value": 20.322155,
        "Target Weight": 0.288448,
        "Weight Delta": -0.090406,
        "Delta Quantity": -0.000079,
        "Price": 80883.5,
        "Trade Action": "Sell",
    },
    {
        "Asset": "ETH",
        "Current Quantity": 0.0032,
        "Current Value": 7.57392,
        "Current Weight": 0.107503,
        "Target Quantity": -0.000587,
        "Target Value": -1.39015,
        "Target Weight": -0.019731,
        "Weight Delta": -0.127234,
        "Delta Quantity": -0.003787,
        "Price": 2366.85,
        "Trade Action": "Sell",
    },
    {
        "Asset": "HYPE",
        "Current Quantity": 0.0,
        "Current Value": 0.0,
        "Current Weight": 0.0,
        "Target Quantity": -0.00354,
        "Target Value": -0.271358,
        "Target Weight": -0.003852,
        "Weight Delta": -0.003852,
        "Delta Quantity": -0.00354,
        "Price": 76.6505,
        "Trade Action": "Sell",
    },
]
#: Step Size per asset from the run's orders.parquet; Hyperliquid's floor is $10.
CARVER_1025_STEPS = {"BTC": 0.00001, "ETH": 0.0001, "HYPE": 0.01}


class _Broker:
    """Records what the venue would get, in order, and fills it. Holds the given positions."""

    def __init__(self, positions: dict[str, float]):
        self.positions = positions
        self.sent: list[dict] = []

    def get_positions(self) -> pd.DataFrame:
        return pd.DataFrame([{"symbol": k, "qty": v} for k, v in self.positions.items() if v])

    def place_orders(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for i, (_, o) in enumerate(frame.iterrows()):
            self.sent.append(
                {"symbol": o["symbol"], "side": o["side"], "qty": o["qty"], "reduce_only": o["reduce_only"]}
            )
            rows.append(
                {
                    "symbol": o["symbol"],
                    "side": o["side"],
                    "qty": o["qty"],
                    "price": o["price"],
                    "order_id": str(i),
                    "status": "FILLED",
                    "error": "",
                }
            )
        return pd.DataFrame(rows)


def _plan(rows, *, lot_rules, min_qty_map=None, min_notional_map=None, venue, min_trade_size=0.01):
    return FuturesRebalancer()._create_executable_orders(
        rebalancing_df=pd.DataFrame(rows),
        min_trade_size=min_trade_size,
        min_notional=10.0,
        min_notional_configured=None,
        min_notional_map=min_notional_map or {},
        min_qty_map=min_qty_map or {},
        lot_rules=lot_rules,
        default_lot_rules=LotRules.from_snapshot({}, venue=venue),
    )


def _execute(orders: pd.DataFrame, positions: dict[str, float]):
    broker = _Broker(positions)
    alerts: list = []
    report = execute_orders(
        broker=broker,
        orders_df=orders,
        trading_enabled=True,
        alert=lambda m, *, site=None: alerts.append(site) or True,
    )
    return report, broker.sent, alerts


def test_guard_carver_hl_2026_10_25_eth_flip_sends_the_close_and_does_not_freeze():
    """The prod ETH flip becomes a sent reduce-only close of 0.0032 and a suppressed sub-min short entry."""
    rules = {
        a: LotRules.from_snapshot({"step_size": s, "min_notional": 10.0}, venue="hyperliquid")
        for a, s in CARVER_1025_STEPS.items()
    }
    orders = _plan(
        CARVER_1025_REBALANCING,
        lot_rules=rules,
        min_notional_map=dict.fromkeys(CARVER_1025_STEPS, 10.0),
        venue="hyperliquid",
    )
    eth = orders[orders["Asset"] == "ETH"]
    assert len(eth) == 2, eth.to_string()
    close, opening = eth.iloc[0], eth.iloc[1]

    assert close["Action"] == "Sell"
    assert close["Raw Quantity"] == pytest.approx(0.0032)
    assert close["Adjusted Quantity"] == pytest.approx(0.0032)
    assert close["Order Status"] == "To be placed"
    assert bool(close["reduce_only"]) and bool(close["full_close"]) and not bool(close["opens_exposure"])
    assert bool(close["Executable"])

    assert opening["Action"] == "Sell"
    assert opening["Raw Quantity"] == pytest.approx(0.000587)
    assert opening["Order Status"] == "Below min notional"
    assert bool(opening["opens_exposure"]) and not bool(opening["reduce_only"]) and not bool(opening["full_close"])
    assert not bool(opening["Executable"])

    # BTC and HYPE keep their prod statuses.
    by_asset = orders.drop_duplicates("Asset").set_index("Asset")["Order Status"]
    assert by_asset["BTC"] == RESIDUAL_STATUS
    assert by_asset["HYPE"] == "Below threshold"

    report, sent, alerts = _execute(orders, {"BTC": 0.00033, "ETH": 0.0032})
    assert not report.get("frozen"), report.get("frozen_orders")
    assert "frozen" not in alerts
    assert sent == [{"symbol": "ETH", "side": "sell", "qty": pytest.approx(0.0032), "reduce_only": True}]
    assert [r["symbol"] for r in report.get("residuals", [])] == ["BTC"]


# --- property: every flip shape --------------------------------------------------

HL = dict(venue="hyperliquid", reduce_only_exempt=True)
KRAKEN = dict(venue="kraken", reduce_only_exempt=False)


#: The 2026-10-25 carver-HL book value, so a leg's weight reads as it did on that day.
NAV = 70.453358


def _flip_row(asset, cur, tgt, price, nav=NAV):
    cw, tw = cur * price / nav, tgt * price / nav
    delta = tgt - cur
    return {
        "Asset": asset,
        "Current Quantity": cur,
        "Current Value": cur * price,
        "Current Weight": cw,
        "Target Quantity": tgt,
        "Target Value": tgt * price,
        "Target Weight": tw,
        "Weight Delta": tw - cw,
        "Delta Quantity": delta,
        "Price": price,
        "Trade Action": "Buy" if delta > 0 else "Sell",
    }


def _check_flip(cur, tgt, price, *, step, min_qty, venue, min_trade_size=0.01, min_notional=10.0):
    rules = LotRules.from_snapshot({"step_size": step, "min_qty": min_qty, "min_notional": min_notional}, venue=venue)
    row = _flip_row("X", cur, tgt, price)
    orders = _plan(
        [row],
        lot_rules={"X": rules},
        min_qty_map={"X": min_qty} if min_qty else {},
        min_notional_map={"X": min_notional},
        venue=venue,
        min_trade_size=min_trade_size,
    )
    side = "Buy" if tgt > cur else "Sell"
    assert list(orders["Asset"]) == ["X", "X"], orders.to_string()
    close, opening = orders.iloc[0], orders.iloc[1]

    # Two legs of the same side; their quantities add up to the delta.
    assert close["Action"] == side and opening["Action"] == side
    assert close["Raw Quantity"] == pytest.approx(abs(cur))
    assert opening["Raw Quantity"] == pytest.approx(abs(tgt))
    assert close["Raw Quantity"] + opening["Raw Quantity"] == pytest.approx(abs(tgt - cur))

    # The close leg is a full close, sent reduce-only, never an open.
    assert bool(close["reduce_only"]) and bool(close["full_close"]) and not bool(close["opens_exposure"])
    close_lot = rules.accept(abs(cur), None, reduce_only=True)
    if close_lot.skip and rules.never_accepts_close(close_lot):
        assert close["Order Status"] == RESIDUAL_STATUS  # dust the venue can never take
    else:
        # Exempt from the floor and the churn band at any size.
        assert close["Order Status"] == "To be placed", close["Reason"]
        assert close["Adjusted Quantity"] == pytest.approx(abs(close_lot.rounded))

    # The open leg is an open from flat, judged by the normal gate chain on its own weight.
    assert bool(opening["opens_exposure"]) and not bool(opening["reduce_only"]) and not bool(opening["full_close"])
    open_qty = abs(rules.accept(abs(tgt), None).rounded)
    if abs(tgt * price / NAV) < min_trade_size:
        assert opening["Order Status"] == "Below threshold"
    elif open_qty * price < max(min_notional, min_qty * price):
        assert opening["Order Status"] == "Below min notional"
    else:
        assert opening["Order Status"] == "To be placed", opening["Reason"]

    # The executor sends the close before the open, and a flip never freezes the book.
    report, sent, _ = _execute(orders, {"X": cur})
    assert not report.get("frozen"), (report.get("frozen_orders"), orders.to_string())
    if sent:
        if close["Order Status"] == "To be placed":
            assert bool(sent[0]["reduce_only"]) and sent[0]["qty"] == pytest.approx(close["Adjusted Quantity"])
        assert all(s["side"] == side.lower() for s in sent)
        assert len(sent) == int(orders["Executable"].sum())
    return close, opening


@pytest.mark.parametrize("sign", [1, -1], ids=["long_to_short", "short_to_long"])
@pytest.mark.parametrize(
    ("cur", "tgt", "close_status", "open_status"),
    [
        (0.0032, -0.000587, "To be placed", "Below min notional"),  # the 25 Oct shape: both legs sub-min
        (0.0032, -0.02, "To be placed", "To be placed"),  # sub-min close, above-min open
        (0.02, -0.000587, "To be placed", "Below min notional"),  # above-min close, sub-min open
        (0.02, -0.03, "To be placed", "To be placed"),  # both legs above min
        (0.02, -0.0002, "To be placed", "Below threshold"),  # open leg inside the churn band ($0.47, 0.7%)
    ],
)
def test_flip_legs_on_a_reduce_only_exempt_venue(sign, cur, tgt, close_status, open_status):
    close, opening = _check_flip(sign * cur, sign * tgt, 2366.85, step=0.0001, min_qty=0.0, venue="hyperliquid")
    assert close["Order Status"] == close_status
    assert opening["Order Status"] == open_status


@pytest.mark.parametrize("sign", [1, -1], ids=["long_to_short", "short_to_long"])
def test_flip_dust_close_on_a_kraken_like_lot_rule_is_residual(sign):
    """A close below min_qty on a venue with no reduce-only exemption can never be sent: a residual."""
    close, _ = _check_flip(sign * 0.0088, sign * -60.0, 0.2076, step=1e-8, min_qty=20.0, venue="kraken")
    assert close["Order Status"] == RESIDUAL_STATUS
    assert not bool(close["Executable"])


def test_flip_close_that_rounds_to_zero_is_residual():
    close, _ = _check_flip(0.00004, -0.02, 2366.85, step=0.0001, min_qty=0.0, venue="hyperliquid")
    assert close["Order Status"] == RESIDUAL_STATUS


def test_flip_property_randomized():
    rng = random.Random(2396)
    for _ in range(400):
        price = 10 ** rng.uniform(-5, 5)
        cur = 10 ** rng.uniform(-3, 2.5) / price * rng.choice([1, -1])
        tgt = -cur / abs(cur) * 10 ** rng.uniform(-3, 2.5) / price
        venue = rng.choice(["hyperliquid", "kraken"])
        step = 10 ** rng.randint(-8, 0)
        min_qty = rng.choice([0.0, 10 ** rng.uniform(-6, 2)])
        _check_flip(cur, tgt, price, step=step, min_qty=min_qty, venue=venue, min_notional=rng.choice([0.5, 5.0, 10.0]))


def test_non_flips_are_one_row_each():
    """Opens, adds, partial reduces and full closes are untouched: one row each."""
    rows = [
        _flip_row("OPEN", 0.0, 0.01, 2000.0),
        _flip_row("ADD", 0.01, 0.02, 2000.0),
        _flip_row("TRIM", 0.02, 0.01, 2000.0),
        _flip_row("CLOSE", 0.02, 0.0, 2000.0),
    ]
    rules = {r["Asset"]: LotRules.from_snapshot({"step_size": 0.0001}, venue="hyperliquid") for r in rows}
    orders = _plan(rows, lot_rules=rules, venue="hyperliquid")
    assert list(orders["Asset"]) == ["OPEN", "ADD", "TRIM", "CLOSE"]


def test_flatten_gate_never_reduces_a_flip_past_the_position():
    """Under the FLATTEN gate both legs oppose the position; the open leg is dropped once the close takes it to zero."""
    from quantbox.trade.reconciliation.cycle import filter_reduce_only

    rules = {"X": LotRules.from_snapshot({"step_size": 0.0001}, venue="hyperliquid")}
    orders = _plan([_flip_row("X", 0.02, -0.03, 2366.85)], lot_rules=rules, venue="hyperliquid")
    assert int(orders["Executable"].sum()) == 2
    kept = filter_reduce_only(orders[orders["Executable"]], _Broker({"X": 0.02}))
    assert list(kept["Adjusted Quantity"]) == [pytest.approx(0.02)]
    assert bool(kept.iloc[0]["full_close"])
