"""Regression for the 2026-06-30 Kraken-USD first-armed-run dust failure.

Incident: the freshly-armed $278.80 Kraken-USD ``crypto_trend`` book placed
ZERO fills on its first live run — ``n_orders=8, n_fills=0, total_failed=1`` —
with the broker logging ``Quantity 0.001442 below Kraken minimum 5.0 for USDC``
then ``Kraken orders failed (1/1)``.

Root cause (traced end-to-end through FuturesRebalancer -> KrakenBroker):
  * A ~0.0014 USDC residue on the account was reported by
    ``KrakenBroker.get_positions()`` as a liquidatable position. The rebalancer
    built a close-out sell for it; close-outs are min-notional-exempt, so it was
    the book's only ``Executable`` order on that (low-trend-exposure) day.
  * That dust sell reached Kraken at 0.001442 USDC, below the 5.0 USDC minimum,
    and was counted as a hard FAILURE (``total_failed=1``).
  * The 7 strategy buys did NOT reach the broker because the trend exposure was
    near-flat that day (target legs sub-$10), so they were correctly suppressed
    "Below min notional" upstream — NOT gated on the dust sell. (Only 1 order
    ever reached ``place_orders`` -> the "1/1" in the log.)

Fixes pinned here:
  (a) Quote-equivalent stablecoin dust is excluded from ``get_positions`` — a
      USDC residue alongside a USD quote is cash-equivalent, never liquidated.
  (b) A genuinely sub-minimum order is a clean SKIP (status SKIPPED), never a
      FAILURE, and never blocks the rest of the batch.

The headline test mirrors the task's spec: account = {USD cash + tiny USDC
dust} + 7-name target -> the 7 buys are produced and executable, the dust is
skipped cleanly, and there are zero failures.
"""

from __future__ import annotations

import pandas as pd

from quantbox.plugins.broker.kraken import KrakenBroker
from quantbox.plugins.rebalancing.futures_rebalancer import FuturesRebalancer

# Seven liquid large-caps with USD books, mirroring the armed Kraken universe.
_PRICES = {
    "BTC": 60000.0,
    "ETH": 3000.0,
    "SOL": 150.0,
    "XRP": 0.50,
    "AAVE": 100.0,
    "SUI": 1.20,
    "LINK": 15.0,
}


class _FakeExchange:
    """Minimal ccxt.kraken stand-in covering the 7-name USD universe."""

    def __init__(self, balance):
        self._balance = balance
        self.created_orders: list[dict] = []
        self.markets = {
            f"{base}/USD": {
                "spot": True,
                "base": base,
                "quote": "USD",
                "precision": {"amount": 6},
                "limits": {"amount": {"min": 0.0}, "cost": {"min": 5.0}},
            }
            for base in (*_PRICES, "USDC")
        }
        self.markets["USDC/USD"]["limits"]["amount"]["min"] = 5.0  # Kraken's real USDC floor

    def load_markets(self):
        return self.markets

    def fetch_balance(self):
        return self._balance

    def fetch_ticker(self, symbol):
        base = symbol.split("/")[0]
        return {"last": 1.0 if base == "USDC" else _PRICES.get(base, 0.0)}

    def amount_to_precision(self, symbol, amount):
        return f"{float(amount):.6f}"

    def create_order(self, symbol, type, side, amount, price=None):
        base = symbol.split("/")[0]
        order = {
            "id": f"oid-{len(self.created_orders)}",
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "average": price or (1.0 if base == "USDC" else _PRICES.get(base, 0.0)),
            "filled": amount,
        }
        self.created_orders.append(order)
        return order


def _broker(balance):
    return KrakenBroker(quote_asset="USD", _exchange=_FakeExchange(balance))


def _equal_weights() -> dict[str, float]:
    return {sym: 1.0 / len(_PRICES) for sym in _PRICES}


# Account: $278.80 USD cash + the exact 0.001442 USDC dust from the incident.
_ACCOUNT = {"total": {"ZUSD": 278.80, "USDC": 0.001442}}


def test_dust_excluded_from_positions():
    """The 0.0014 USDC residue is NOT reported as a liquidatable position."""
    pos = _broker(_ACCOUNT).get_positions()
    assert "USDC" not in pos["symbol"].tolist()
    assert pos.empty  # the account holds only cash + dust


def test_armed_run_produces_seven_buys_skips_dust_zero_failures():
    """End-to-end: 7-name target on a {USD cash + USDC dust} account ->
    7 executable buys, dust skipped cleanly, zero failures."""
    broker = _broker(_ACCOUNT)
    reb = FuturesRebalancer()

    result = reb.generate_orders(
        weights=_equal_weights(),
        broker=broker,
        params={
            "capital_at_risk": 1.0,
            "stable_coin_symbol": "USD",
            "min_trade_size": 0.01,
            "min_notional": 10.0,
        },
    )
    orders = result["orders"]

    # No USDC row at all — the dust never entered the rebalancing universe.
    assert "USDC" not in orders["Asset"].tolist()

    executable = orders[orders["Executable"]]
    assert len(executable) == 7  # all 7 buys (~$39.8 each, well above $10 min)
    assert set(executable["Action"]) == {"Buy"}

    # Submit exactly what the pipeline would submit.
    broker_orders = pd.DataFrame(
        [
            {"symbol": r["Asset"], "side": "buy", "qty": float(r["Adjusted Quantity"]), "price": float(r["Price"])}
            for _, r in executable.iterrows()
        ]
    )
    fills = broker.place_orders(broker_orders)

    assert (fills["status"] == "FILLED").sum() == 7
    assert (fills["status"] == "FAILED").sum() == 0
    assert (fills["status"] == "SKIPPED").sum() == 0
    assert len(broker._exchange.created_orders) == 7


def test_sub_min_dust_sell_skips_cleanly_at_broker():
    """Defense-in-depth (b): even if a sub-min dust sell DID reach the broker
    (e.g. a non-stablecoin residue), it is a clean SKIP, not a FAILURE, and the
    legitimate buys in the same batch still fill."""
    broker = _broker(_ACCOUNT)
    orders = pd.DataFrame(
        [
            {"symbol": "USDC", "side": "sell", "qty": 0.001442, "price": 1.0},  # below 5.0 min
            {"symbol": "BTC", "side": "buy", "qty": 0.0006, "price": 60000.0},  # ~$36, real
        ]
    )
    fills = broker.place_orders(orders)
    by_sym = dict(zip(fills["symbol"], fills["status"], strict=False))
    assert by_sym["USDC"] == "SKIPPED"
    assert by_sym["BTC"] == "FILLED"


# ---------------------------------------------------------------------------
# Nit 1: peg-scoped stablecoin exclusion (PR #80 review)
# ---------------------------------------------------------------------------


def test_eur_stable_not_excluded_on_usd_book():
    """A EUR-pegged stable (EURT) on a USD book is a genuine FX position, NOT
    USD-cash-equivalent dust — it must remain liquidatable."""
    balance = {"total": {"ZUSD": 278.80, "EURT": 50.0, "USDC": 0.001442}}
    pos = _broker(balance).get_positions()
    symbols = pos["symbol"].tolist()
    assert "EURT" in symbols  # FX position retained on the exit path
    assert "USDC" not in symbols  # USD-pegged dust still excluded


def test_depegged_token_not_excluded_on_usd_book():
    """A depegged token (USTC) is not cash-equivalent and must stay liquidatable."""
    balance = {"total": {"ZUSD": 100.0, "USTC": 10000.0}}
    pos = _broker(balance).get_positions()
    assert "USTC" in pos["symbol"].tolist()


def test_usd_stables_excluded_on_usd_book():
    """USD-pegged stables (USDT/DAI) remain cash-equivalent dust on a USD book."""
    balance = {"total": {"ZUSD": 100.0, "USDT": 0.002, "DAI": 0.003}}
    pos = _broker(balance).get_positions()
    assert pos.empty


# ---------------------------------------------------------------------------
# Nit 2: skipped close-out SELL surfaces a residual-exposure note (PR #80)
# ---------------------------------------------------------------------------


def test_skipped_closeout_sell_surfaces_residual(caplog):
    """A sub-min close-out SELL is SKIPPED (not FAILED) but must be surfaced as a
    residual-exposure note so a trapped residual stays visible."""
    import logging

    broker = _broker(_ACCOUNT)
    orders = pd.DataFrame([{"symbol": "USDC", "side": "sell", "qty": 0.001442, "price": 1.0}])
    with caplog.at_level(logging.WARNING):
        fills = broker.place_orders(orders)
    assert (fills["status"] == "SKIPPED").all()
    assert "residual exposure" in fills.iloc[0]["error"].lower()
    assert any("residual exposure RETAINED" in r.message for r in caplog.records)


def test_skipped_buy_does_not_surface_residual(caplog):
    """A sub-min BUY skip is a quiet no-op — it must NOT emit a residual note."""
    import logging

    broker = _broker(_ACCOUNT)
    orders = pd.DataFrame([{"symbol": "BTC", "side": "buy", "qty": 1e-9, "price": 60000.0}])
    with caplog.at_level(logging.WARNING):
        broker.place_orders(orders)
    assert not any("residual exposure RETAINED" in r.message for r in caplog.records)
