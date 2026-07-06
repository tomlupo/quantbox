"""Regression for the frozen Kraken-USD ``crypto_trend`` book (issue #105).

Incident (diagnosed on obsidian-vaults#111): the live ~$278, 10-name Kraken-USD
book was FROZEN. Every daily rebalance produced $4-8 per-name deltas, but the
rebalancer gated them against a FLAT ``min_notional`` config of $10, so nothing
ever traded ("Below min notional" on every leg).

Kraken's REAL minimums are per-pair and far lower than $10:
  * ``costmin`` (limits.cost.min) = $0.50 for all USD pairs,
  * ``ordermin`` (limits.amount.min) is a base-unit floor: BTC 0.00005 (~$4),
    ETH 0.001 (~$4), SOL 0.06 (~$5), XRP 1.65 (~$4), ADA 20 units (~$11),
    DOGE 50 units (~$8).

The true binding minimum per pair is ``max(costmin, ordermin * price)``.

Fix pinned here (``FuturesRebalancer``):
  * ``_generate_orders`` now captures per-pair ``min_notional``/``min_qty`` from
    the broker snapshot (previously it read only ``mid`` and discarded them).
  * ``_create_executable_orders`` gates each order against the per-pair floor,
    falling back to the flat config ``min_notional`` only as a safety backstop,
    and additionally enforces the base-unit ``ordermin`` so ADA/DOGE-style
    sub-base-unit orders are suppressed rather than sent-and-rejected.
"""

from __future__ import annotations

import pandas as pd

from quantbox.plugins.rebalancing.futures_rebalancer import FuturesRebalancer

# Prices chosen so ordermin * price matches the task's stated per-pair notionals.
_PRICES = {
    "BTC": 80000.0,  # ordermin 0.00005 -> ~$4
    "ETH": 4000.0,  # ordermin 0.001   -> ~$4
    "SOL": 83.0,  # ordermin 0.06    -> ~$5
    "XRP": 2.40,  # ordermin 1.65    -> ~$4
    "ADA": 0.55,  # ordermin 20      -> ~$11
    "DOGE": 0.16,  # ordermin 50      -> ~$8
}

# Kraken per-pair base-unit floors (limits.amount.min).
_ORDERMIN = {
    "BTC": 0.00005,
    "ETH": 0.001,
    "SOL": 0.06,
    "XRP": 1.65,
    "ADA": 20.0,
    "DOGE": 50.0,
}

_COSTMIN = 0.50  # limits.cost.min, all Kraken USD pairs


class _FakeBroker:
    """Minimal broker exposing the snapshot columns the rebalancer consumes.

    ``get_market_snapshot`` returns the same schema as ``KrakenBroker``:
    ``symbol, mid, min_qty, step_size, min_notional``.
    """

    def __init__(self, cash: float, holdings: dict[str, float]):
        self._cash = cash
        self._holdings = holdings

    def get_cash(self):
        return {"USD": self._cash}

    def get_positions(self):
        if not self._holdings:
            return pd.DataFrame(columns=["symbol", "qty"])
        return pd.DataFrame([{"symbol": s, "qty": q} for s, q in self._holdings.items()])

    def get_market_snapshot(self, symbols):
        rows = []
        for sym in symbols:
            rows.append(
                {
                    "symbol": sym,
                    "mid": _PRICES.get(sym, 0.0),
                    "min_qty": _ORDERMIN.get(sym, 0.0),
                    "step_size": 0.0,
                    "min_notional": _COSTMIN,
                }
            )
        return pd.DataFrame(rows)


def _status(orders: pd.DataFrame, asset: str) -> str:
    return orders.loc[orders["Asset"] == asset, "Order Status"].iloc[0]


def test_small_deltas_clear_perpair_min_not_flat_ten():
    """The frozen case: a $278 book already near target, $4-8 rebalance deltas.

    BTC/ETH/SOL/XRP legs clear their true per-pair floor (~$4-5) and are
    executable, even though each is below the flat $10 config min_notional that
    previously froze the whole book.
    """
    total = 278.0
    # Six equal-weight names -> ~$46.33 target each.
    weights = {s: 1.0 / 6 for s in _PRICES}
    target_val = total / 6

    # Current holdings sit ~$5 BELOW target for the 4 low-min names (a $5 buy
    # delta) and AT target for ADA/DOGE (no delta), mirroring a near-balanced
    # book that only needs small top-ups.
    holdings = {}
    for s in _PRICES:
        cur_val = target_val if s in ("ADA", "DOGE") else target_val - 5.0
        holdings[s] = cur_val / _PRICES[s]

    broker = _FakeBroker(cash=total, holdings=holdings)
    reb = FuturesRebalancer()
    result = reb.generate_orders(
        weights=weights,
        broker=broker,
        params={
            "capital_at_risk": 1.0,
            "stable_coin_symbol": "USD",
            "min_trade_size": 0.0,  # isolate the min-notional gate
            "min_notional": 10.0,  # the flat floor that froze the book
        },
    )
    orders = result["orders"]

    # The ~$5 top-ups on the low-min names are now EXECUTABLE (were frozen at $10).
    for sym in ("BTC", "ETH", "SOL", "XRP"):
        row = orders[orders["Asset"] == sym].iloc[0]
        assert row["Order Status"] == "To be placed", (sym, row["Reason"])
        assert row["Executable"], sym
        assert 0 < row["Notional Value"] < 10.0  # below the old flat floor


def test_subordermin_dust_suppressed_by_perpair_floor():
    """Genuine dust: a sub-ordermin ADA/DOGE order is suppressed, not sent.

    A tiny buy (below the base-unit ordermin) must be gated — otherwise Kraken
    rejects it (the actual 2026-07-05 reject). It is suppressed as "Below min
    notional" or "Below min qty", never marked executable.
    """
    # Flat book, weights that produce tiny sub-ordermin ADA/DOGE targets.
    total = 100.0
    # Weight so ADA target notional ~ $2 (< $11 per-pair floor) and DOGE ~ $1.
    weights = {"ADA": 0.02, "DOGE": 0.01, "BTC": 0.5}
    broker = _FakeBroker(cash=total, holdings={})
    reb = FuturesRebalancer()
    result = reb.generate_orders(
        weights=weights,
        broker=broker,
        params={
            "capital_at_risk": 1.0,
            "stable_coin_symbol": "USD",
            "min_trade_size": 0.0,
            "min_notional": 10.0,
        },
    )
    orders = result["orders"]

    # ADA ~$2 and DOGE ~$1 are below their per-pair floors ($11 / $8) -> gated.
    for sym in ("ADA", "DOGE"):
        row = orders[orders["Asset"] == sym].iloc[0]
        assert not row["Executable"], (sym, row["Order Status"])
        assert row["Order Status"] in ("Below min notional", "Below min qty")

    # BTC ~$50 clears easily and stays executable.
    btc = orders[orders["Asset"] == "BTC"].iloc[0]
    assert btc["Executable"]


def test_backstop_used_when_snapshot_has_no_perpair_floor():
    """When the snapshot carries no per-pair floor, the flat config min_notional
    is used as the safety backstop (unchanged legacy behaviour)."""

    class _NoLimitsBroker(_FakeBroker):
        def get_market_snapshot(self, symbols):
            snap = super().get_market_snapshot(symbols)
            snap["min_notional"] = 0.0
            snap["min_qty"] = 0.0
            return snap

    broker = _NoLimitsBroker(cash=278.0, holdings={})
    reb = FuturesRebalancer()
    # Weight producing a ~$6 BTC target -> below the $10 backstop -> suppressed.
    result = reb.generate_orders(
        weights={"BTC": 6.0 / 278.0, "ETH": 0.5},
        broker=broker,
        params={
            "capital_at_risk": 1.0,
            "stable_coin_symbol": "USD",
            "min_trade_size": 0.0,
            "min_notional": 10.0,
        },
    )
    orders = result["orders"]
    btc = orders[orders["Asset"] == "BTC"].iloc[0]
    assert not btc["Executable"]
    assert btc["Order Status"] == "Below min notional"
    assert "10.00" in btc["Reason"]  # flat backstop applied
