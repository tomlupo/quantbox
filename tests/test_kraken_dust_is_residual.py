"""Dust a venue can never accept is residual, not a freeze (TOM-2212).

Decision (Tom, 2026-10-23): "Residual is not a freeze" (TOM-1778, partial
reduces) extends to full closes the venue can never accept. Churn-band trims on
held names do not trap a book either. The book freezes only when a suppressed
row may hold something the venue could accept.

Incident: the live crypto-trend Kraken book paged REBALANCER FROZEN on 7 of 10
runs (2026-10-13 .. 10-20), each with ``total_executed = 0``. The book holds dust
far below Kraken's minimum quantity (ADA 0.0088 vs min 20; AVAX, BNB, DOT, TAO,
TRX, WLD). It can never sell it, so the close came back every day.

Source of the fixtures, copied read-only from core (``quantbox-live/artifacts``):

* ``2026-10-19__trade_full_pipeline_v1__76f62d2c8092__20261019T041505Z``: the
  current quantity, price and target weight of every row of
  ``rebalancing.parquet``; cash 224.4803 USD (run metrics). No order reached
  the venue that day.
* ``2026-10-20__trade_full_pipeline_v1__1d7c1d7487ce__20261020T041504Z``: the
  same columns and cash. One LINK buy rested at the venue (``total_working = 1``).
* Kraken's lot data for these pairs (``AssetPairs``: ``ordermin``,
  ``lot_decimals``, ``costmin`` 0.5), from the TOM-2207 shadow-run tape.

The path is the real one: ``FuturesRebalancer.generate_orders`` -> real
``KrakenBroker`` (ccxt faked at the boundary) -> real ``execute_orders``. At
6138af9 the 2026-10-19 replay is FROZEN (the dust is ``Below min qty``, and the
churn-band trims count as suppressed sells), and the 2026-10-20 replay lists no
residual.

The guard tests below pin what must still page. Each one goes red if the fix
reads too much as residual or untrapped.
"""

from __future__ import annotations

import ccxt
import pandas as pd
import pytest

from quantbox.plugins.broker import _fills
from quantbox.plugins.broker.kraken import KrakenBroker
from quantbox.plugins.rebalancing.futures_rebalancer import RESIDUAL_STATUS, FuturesRebalancer
from quantbox.trade.executor import execute_orders
from quantbox.trade.lots import LotRules

CASH_USD = 224.4803

#: Kraken AssetPairs: (ordermin, lot_decimals). costmin is 0.5 USD on every pair.
KRAKEN_LOTS = {
    "ADA": (20.0, 8),
    "AVAX": (0.5, 8),
    "BNB": (0.007, 5),
    "BTC": (0.00006, 8),
    "DOT": (3.9, 8),
    "ETH": (0.002, 8),
    "LINK": (0.35, 8),
    "NEAR": (1.0, 8),
    "QNT": (0.02, 8),
    "SOL": (0.05, 8),
    "SUI": (5.0, 5),
    "TAO": (0.015, 5),
    "TRX": (16.0, 8),
    "WLD": (9.0, 5),
    "XRP": (4.0, 8),
}

#: asset: (current quantity, price, target weight) — rebalancing.parquet, 2026-10-19.
DAY_2026_10_19 = {
    "ADA": (0.00878402, 0.2203, 0.009199234028907904),
    "AVAX": (0.0001208256, 9.734, 0.0),
    "BNB": (1.07e-06, 725.52, 0.0),
    "BTC": (0.00016805, 82390.0, 0.04688749519244878),
    "DOT": (6.71251e-05, 1.1162, 0.0),
    "ETH": (0.0034194224, 2418.47, 0.023671775497128233),
    "NEAR": (1.55087185, 5.0144, 0.01893259870802534),
    "QNT": (0.0, 193.51, 0.009893123095971405),
    "SOL": (0.0643845395, 105.06, 0.020473112211853904),
    "SUI": (5.73925, 0.9993, 0.01263506432659219),
    "TAO": (9.89e-06, 271.6441, 0.0),
    "TRX": (0.00259157, 0.330814, 0.0),
    "WLD": (9e-05, 0.4919, 0.0),
    "XRP": (4.0473011, 1.30912, 0.016708109613118253),
}

#: asset: (current quantity, price, target weight) — rebalancing.parquet, 2026-10-20.
DAY_2026_10_20 = {
    "ADA": (0.00878402, 0.218473, 0.0),
    "AVAX": (0.0001208256, 9.605, 0.0),
    "BNB": (1.07e-06, 729.28, 0.0),
    "BTC": (0.00016805, 82039.2, 0.046828997746002944),
    "DOT": (6.71251e-05, 1.151, 0.0),
    "ETH": (0.0034194224, 2410.75, 0.02380053131252028),
    "LINK": (0.0, 12.06476, 0.02170508732204679),
    "NEAR": (1.55087185, 4.8407, 0.01816193851706956),
    "QNT": (0.0, 197.85, 0.009892558607033987),
    "SOL": (0.0643845395, 105.25, 0.021105885023349764),
    "SUI": (5.73925, 1.0011, 0.011756206859850796),
    "TAO": (9.89e-06, 267.4862, 0.014388428769363576),
    "TRX": (0.00259157, 0.331233, 0.0),
    "WLD": (9e-05, 0.5502, 0.012165795840496343),
    "XRP": (4.0473011, 1.3044, 0.016362681168037792),
}

#: The live config's rebalancing params (configs/crypto_trend_kraken.yaml), plus
#: the mode trade.full_pipeline.v1 threads in.
PARAMS = {
    "capital_at_risk": 1.0,
    "stable_coin_symbol": "USD",
    "min_trade_size": 0.01,
    "max_leverage": 1,
    "mode": "live",
}


class _Kraken:
    """ccxt.kraken at the boundary: balances, tickers, markets, and one order outcome.

    ``outcome`` is what ``create_order`` does: ``"working"`` (accepted, still on
    the book — what prod saw for LINK on 2026-10-20), ``"reject"`` (ccxt
    InvalidOrder), or ``"unconfirmed"`` (a reply with no status and no id).
    """

    def __init__(self, day: dict, outcome: str = "working", lots: dict | None = None):
        self.day = day
        self.outcome = outcome
        self.lots = lots or KRAKEN_LOTS
        self.created: list[tuple[str, str, float]] = []

    def load_markets(self):
        return {
            f"{b}/USD": {
                "spot": True,
                "base": b,
                "quote": "USD",
                "precision": {"amount": d},
                "limits": {"amount": {"min": m}, "cost": {"min": 0.5}},
            }
            for b, (m, d) in self.lots.items()
        }

    def fetch_balance(self):
        held = {a: q for a, (q, _, _) in self.day.items() if q > 0}
        return {"total": {"USD": CASH_USD, **held}}

    def fetch_ticker(self, ms):
        return {"last": self.day[ms.split("/")[0]][1]}

    def amount_to_precision(self, ms, amount):
        d = self.lots[ms.split("/")[0]][1]
        return f"{int(float(amount) * 10**d) / 10**d:.{d}f}"

    def create_order(self, symbol, type, side, amount, price=None):
        self.created.append((symbol, side, float(amount)))
        if self.outcome == "reject":
            raise ccxt.InvalidOrder("EOrder:Invalid order")
        if self.outcome == "unconfirmed":
            return {}
        return {
            "id": f"oid-{len(self.created)}",
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "status": "open",
            "filled": 0.0,
        }

    def fetch_order(self, oid, ms):
        return {"id": oid, "status": "open", "filled": 0.0, "amount": 1.0}


@pytest.fixture(autouse=True)
def _no_confirmation_wait(monkeypatch):
    """The fill-confirmation poll sleeps between re-reads; the fake venue answers at once."""
    monkeypatch.setattr(_fills.time, "sleep", lambda s: None)


def _run(day: dict, *, broker: KrakenBroker | None = None, weights: dict | None = None, outcome: str = "working"):
    broker = broker or KrakenBroker(quote_asset="USD", _exchange=_Kraken(day, outcome))
    if weights is None:
        weights = {a: w for a, (_, _, w) in day.items() if w > 0}
    orders = FuturesRebalancer().generate_orders(weights=weights, broker=broker, params=PARAMS)["orders"]
    alerts: list[tuple[str, str]] = []
    report = execute_orders(
        broker=broker,
        orders_df=orders,
        trading_enabled=True,
        alert=lambda message, *, site=None: alerts.append((site, message)) or True,
    )
    return orders, report, alerts, broker._exchange.created


def _status(orders: pd.DataFrame, asset: str) -> str:
    return orders.loc[orders["Asset"] == asset, "Order Status"].iloc[0]


# ---------------------------------------------------------------------------
# The incident: whole path, red at 6138af9
# ---------------------------------------------------------------------------


def test_kraken_2026_10_19_dust_and_band_day_is_a_residual_day_not_a_freeze():
    orders, report, alerts, venue_calls = _run(DAY_2026_10_19)

    dust = ["AVAX", "BNB", "DOT", "TAO", "TRX", "WLD"]
    for asset in dust:
        assert _status(orders, asset) == RESIDUAL_STATUS, asset
    assert venue_calls == []
    assert report.get("residual_day") is True
    assert not report.get("frozen"), report.get("freeze_reasons")
    assert alerts == []
    assert [r["symbol"] for r in report["residuals"]] == dust
    assert all("residual retained" in r["reason"] for r in report["residuals"])


def test_kraken_2026_10_20_dust_is_listed_as_residual_while_link_is_sent():
    orders, report, alerts, venue_calls = _run(DAY_2026_10_20)

    assert [c[:2] for c in venue_calls] == [("LINK/USD", "buy")]
    assert report["summary"]["total_working"] == 1
    assert not report.get("frozen")
    assert alerts == []
    assert [r["symbol"] for r in report["residuals"]] == ["ADA", "AVAX", "BNB", "DOT", "TRX"]
    # An order went out, so this is neither a residual day nor a quiet day.
    assert not report.get("residual_day")
    assert not report.get("quiet_day")


def test_kraken_held_book_inside_the_churn_band_is_a_quiet_day():
    """2026-10-19 without the dust: every held name inside the band, two sub-min entries."""
    clean = {a: v for a, v in DAY_2026_10_19.items() if v[2] > 0}
    _, report, alerts, venue_calls = _run(clean)

    assert venue_calls == []
    assert report.get("quiet_day") is True
    assert not report.get("frozen")
    assert alerts == []
    assert "residuals" not in report


# ---------------------------------------------------------------------------
# Guards: what must still page. Red if the fix over-reaches.
# ---------------------------------------------------------------------------


def _with_btc_close(day: dict) -> dict:
    """The day with BTC's target at 0: a real $13.8 full close Kraken accepts (min 0.00006 BTC)."""
    out = dict(day)
    q, p, _ = out["BTC"]
    out["BTC"] = (q, p, 0.0)
    return out


def test_guard_a_rejected_close_the_venue_would_accept_still_alerts():
    day = _with_btc_close(DAY_2026_10_19)
    orders, report, alerts, venue_calls = _run(day, outcome="reject")

    assert _status(orders, "BTC") == "To be placed"
    assert [c[:2] for c in venue_calls] == [("BTC/USD", "sell")]
    assert report["summary"]["total_failed"] == 1
    assert [s for s, _ in alerts] == ["order_failures"]
    assert not report.get("residual_day")


def test_guard_an_unconfirmed_fill_still_alerts():
    day = _with_btc_close(DAY_2026_10_19)
    _, report, alerts, _ = _run(day, outcome="unconfirmed")

    assert report["summary"]["total_executed"] == 0
    assert report["summary"]["total_failed"] == 1
    assert [s for s, _ in alerts] == ["order_failures"]
    assert not report.get("residual_day")


class _SnapshotDriftBroker(KrakenBroker):
    """The broker's snapshot says BTC has no minimum; its markets say 0.001 BTC.

    So the rebalancer plans the BTC close as placeable, and the broker skips it
    at placement: the #81 shape, an executable close skipped with 0 fills.
    """

    def get_market_snapshot(self, symbols):
        snap = super().get_market_snapshot(symbols)
        snap.loc[snap["symbol"] == "BTC", "min_qty"] = 0.0
        return snap


def test_guard_81_an_executable_close_the_broker_skipped_still_freezes():
    day = _with_btc_close(DAY_2026_10_19)
    lots = {**KRAKEN_LOTS, "BTC": (0.001, 8)}
    broker = _SnapshotDriftBroker(quote_asset="USD", _exchange=_Kraken(day, lots=lots))
    orders, report, alerts, venue_calls = _run(day, broker=broker)

    assert _status(orders, "BTC") == "To be placed"
    assert venue_calls == []
    assert report.get("frozen") is True
    assert report["freeze_reasons"]["skipped_sell"] == 1
    assert [s for s, _ in alerts] == ["trapped_residual"]


def _frame(*rows: dict) -> pd.DataFrame:
    base = {"Adjusted Quantity": 0.0, "Price": 1.0, "Min Notional": 5.0, "Executable": False}
    return pd.DataFrame([{**base, **r} for r in rows])


class _Held:
    def get_positions(self):
        return pd.DataFrame([{"symbol": "BTC", "qty": 0.0002}, {"symbol": "ARB", "qty": 100.0}])


BAND_TRIM = {
    "Asset": "BTC",
    "Action": "Sell",
    "Notional Value": 1.08,
    "Order Status": "Below threshold",
    "opens_exposure": False,
}
DUST = {
    "Asset": "TRX",
    "Action": "Sell",
    "Notional Value": 0.0009,
    "Order Status": RESIDUAL_STATUS,
    "opens_exposure": False,
}


@pytest.mark.parametrize("with_dust", [False, True])
def test_guard_a_declared_exit_a_floor_blocked_beside_band_trims_still_freezes(with_dust):
    """A flip below the floor exits the held side: it is not an entry, not a band trim."""
    flip = {
        "Asset": "ARB",
        "Action": "Sell",
        "Notional Value": 7.0,
        "Order Status": "Below min notional",
        "opens_exposure": False,
    }
    rows = [BAND_TRIM, flip] + ([DUST] if with_dust else [])
    alerts: list = []
    report = execute_orders(
        broker=_Held(),
        orders_df=_frame(*rows),
        trading_enabled=True,
        alert=lambda m, *, site=None: alerts.append(site) or True,
    )
    assert report.get("frozen") is True
    assert not report.get("residual_day") and not report.get("quiet_day")
    assert alerts == ["frozen"]


def test_a_freeze_page_names_the_trapped_row_not_the_band_trims():
    flip = {
        "Asset": "ARB",
        "Action": "Sell",
        "Notional Value": 7.0,
        "Order Status": "Below min notional",
        "opens_exposure": False,
    }
    report = execute_orders(
        broker=_Held(), orders_df=_frame(BAND_TRIM, flip), trading_enabled=True, alert=lambda m, *, site=None: True
    )
    assert [r["symbol"] for r in report["frozen_orders"]] == ["ARB"]


@pytest.mark.parametrize("status", ["Zero price", "Invalid (NaN)"])
def test_guard_a_data_fault_beside_dust_and_band_trims_still_freezes(status):
    fault = {"Asset": "SOL", "Action": "Buy", "Notional Value": 0.0, "Order Status": status, "opens_exposure": True}
    report = execute_orders(
        broker=_Held(),
        orders_df=_frame(BAND_TRIM, DUST, fault),
        trading_enabled=True,
        alert=lambda m, *, site=None: True,
    )
    assert report.get("frozen") is True
    assert not report.get("residual_day")


def test_guard_a_band_row_that_declares_no_intent_still_freezes_a_held_book():
    """A third-party rebalancer's row proves nothing: the #82 fail-safe still judges it."""
    undeclared = {k: v for k, v in BAND_TRIM.items() if k != "opens_exposure"}
    report = execute_orders(
        broker=_Held(), orders_df=_frame(undeclared), trading_enabled=True, alert=lambda m, *, site=None: True
    )
    assert report.get("frozen") is True


@pytest.mark.parametrize("band", [False, True], ids=["guard", "with-band-trim"])
def test_guard_tom_1778_partial_reduce_still_reads_residual_beside_dust(band):
    """carver-HL 2026-10-12's partial reduce (ETH $6.33 < $10) and a dust close, one residual day.

    ``guard`` is the TOM-1778 shape, green at 6138af9 too. ``with-band-trim``
    adds a held name inside the churn band: residual only since TOM-2212.
    """
    partial = {
        "Asset": "ETH",
        "Action": "Sell",
        "Notional Value": 6.33,
        "Order Status": RESIDUAL_STATUS,
        "opens_exposure": False,
        "Reason": "Partial reduce 6.33 < 10.00: the venue rejects it, residual retained",
    }
    entry = {
        "Asset": "BTC",
        "Action": "Buy",
        "Notional Value": 1.63,
        "Order Status": "Below min notional",
        "opens_exposure": True,
    }
    report = execute_orders(
        broker=_Held(),
        orders_df=_frame(partial, DUST, entry, *([BAND_TRIM | {"Asset": "SOL"}] if band else [])),
        trading_enabled=True,
        alert=lambda m, *, site=None: True,
    )
    assert report.get("residual_day") is True
    assert not report.get("frozen")
    assert [r["symbol"] for r in report["residuals"]] == ["ETH", "TRX"]


# ---------------------------------------------------------------------------
# Which closes are residual: the venue's facts decide
# ---------------------------------------------------------------------------


def _close_status(rules: LotRules, qty: float, price: float = 1.0, min_qty: float = 0.0) -> pd.Series:
    rebalancing = pd.DataFrame(
        [
            {
                "Asset": "X",
                "Trade Action": "Sell",
                "Delta Quantity": -qty,
                "Price": price,
                "Weight Delta": -0.001,
                "Target Weight": 0.0,
                "Current Quantity": qty,
                "Target Quantity": 0.0,
            }
        ]
    )
    orders = FuturesRebalancer()._create_executable_orders(
        rebalancing_df=rebalancing,
        min_trade_size=0.01,
        min_qty_map={"X": min_qty} if min_qty else {},
        lot_rules={"X": rules},
    )
    return orders.iloc[0]


def test_a_close_below_min_qty_on_an_undeclared_venue_is_not_residual():
    """Without the venue's facts, the missing reduce-only exemption is not proven: the close still pages."""
    row = _close_status(
        LotRules.from_snapshot({"min_qty": 20.0, "step_size": 1e-8}), qty=0.0088, price=0.22, min_qty=20.0
    )
    assert row["Order Status"] == "Below min qty"


def test_a_close_below_min_qty_on_a_reduce_only_exempt_venue_is_sent():
    """Hyperliquid exempts a reduce-only close from its minimums: it is an order, not a residual."""
    rules = LotRules.from_snapshot({"min_qty": 1.0, "step_size": 0.01}, venue="hyperliquid")
    row = _close_status(rules, qty=0.5, price=3.0, min_qty=1.0)
    assert row["Order Status"] == "To be placed"
    assert bool(row["Executable"])


def test_a_close_that_rounds_to_zero_is_residual_on_any_venue():
    rules = LotRules.from_snapshot({"step_size": 0.01}, venue="hyperliquid")
    row = _close_status(rules, qty=0.004, price=3.0)
    assert row["Order Status"] == RESIDUAL_STATUS
    assert "rounds to zero" in row["Reason"]
