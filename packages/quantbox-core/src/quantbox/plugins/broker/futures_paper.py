"""Futures paper trading broker.

Simulates a perpetual-futures account with:
- Signed positions (positive = long, negative = short)
- Margin accounting (initial capital is the portfolio value)
- Optional funding rate simulation
- Configurable leverage
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from quantbox.contracts import PluginMeta

logger = logging.getLogger(__name__)


@dataclass
class FuturesPaperBroker:
    """Paper-trading broker for perpetual futures.

    Positions are stored as signed floats (negative = short).
    Portfolio value equals ``margin_balance`` (the account equity);
    positions are leveraged and do not add to the equity calculation.
    """

    meta = PluginMeta(
        name="sim.futures_paper.v1",
        kind="broker",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Futures paper broker with margin accounting and short support",
        tags=("paper", "futures"),
        capabilities=("paper", "futures", "shorts", "leverage"),
        schema_version="v1",
    )

    # Account
    margin_balance: float = 100_000.0
    quote_currency: str = "USDT"
    leverage: int = 1

    # Positions: symbol -> signed qty (+ long, - short)
    positions: Dict[str, float] = field(default_factory=dict)
    # Entry prices: symbol -> avg entry price
    entry_prices: Dict[str, float] = field(default_factory=dict)
    # Current market prices: symbol -> price (injected by pipeline or seeded)
    prices: Dict[str, float] = field(default_factory=dict)
    # Funding rates: symbol -> rate per snapshot (default 0.01% / 8h)
    funding_rates: Dict[str, float] = field(default_factory=dict)
    default_funding_rate: float = 0.0001  # 0.01% per 8h

    # Internal fill log
    _fill_log: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Price / funding injection
    # ------------------------------------------------------------------

    def set_prices(self, prices: Dict[str, float]) -> None:
        """Inject latest market prices (called by pipeline before rebalancing)."""
        self.prices.update(prices)

    def set_funding_rates(self, rates: Dict[str, float]) -> None:
        """Inject per-symbol funding rates."""
        self.funding_rates.update(rates)

    # ------------------------------------------------------------------
    # BrokerPlugin protocol
    # ------------------------------------------------------------------

    def get_positions(self) -> pd.DataFrame:
        rows = []
        for symbol, qty in self.positions.items():
            if abs(qty) < 1e-12:
                continue
            price = self.prices.get(symbol, 0.0)
            entry = self.entry_prices.get(symbol, price)
            notional = abs(qty) * price
            unrealized_pnl = qty * (price - entry) if price and entry else 0.0
            rows.append({
                "symbol": symbol,
                "qty": qty,
                "notional": notional,
                "entry_price": entry,
                "unrealized_pnl": unrealized_pnl,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["symbol", "qty", "notional", "entry_price", "unrealized_pnl"]
        )

    def get_cash(self) -> Dict[str, float]:
        return {self.quote_currency: self.margin_balance}

    def get_market_snapshot(self, symbols: List[str]) -> pd.DataFrame:
        rows = []
        for s in symbols:
            rows.append({
                "symbol": s,
                "mid": self.prices.get(s),
                "min_qty": 0.0,
                "step_size": 0.0,
                "min_notional": 1.0,
            })
        return pd.DataFrame(rows)

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        fills: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc).isoformat()

        for _, o in orders.iterrows():
            sym = str(o["symbol"])
            side = str(o["side"]).lower()
            qty = float(o["qty"])
            price = self.prices.get(sym, float(o.get("price", 0.0) or 0.0))

            if price <= 0:
                logger.warning("No price for %s, skipping order", sym)
                continue

            signed = qty if side == "buy" else -qty

            # Update position with weighted-average entry price
            old_qty = self.positions.get(sym, 0.0)
            old_entry = self.entry_prices.get(sym, 0.0)
            new_qty = old_qty + signed

            if abs(new_qty) < 1e-12:
                # Flat — realise PnL
                realised = old_qty * (price - old_entry) if old_entry else 0.0
                self.margin_balance += realised
                self.positions.pop(sym, None)
                self.entry_prices.pop(sym, None)
            elif (old_qty >= 0 and signed > 0) or (old_qty <= 0 and signed < 0):
                # Adding to position — weighted average entry
                if abs(old_qty) + abs(signed) > 0:
                    self.entry_prices[sym] = (
                        (abs(old_qty) * old_entry + abs(signed) * price)
                        / (abs(old_qty) + abs(signed))
                    )
                self.positions[sym] = new_qty
            else:
                # Partial close or flip
                close_qty = min(abs(signed), abs(old_qty))
                realised = close_qty * (price - old_entry) * (1 if old_qty > 0 else -1)
                self.margin_balance += realised

                remainder = abs(signed) - close_qty
                if remainder < 1e-12:
                    self.positions[sym] = new_qty
                    # entry stays the same for the remaining original-direction position
                else:
                    # Flipped direction
                    self.positions[sym] = new_qty
                    self.entry_prices[sym] = price

            notional = abs(signed) * price
            fill = {
                "symbol": sym,
                "side": side,
                "qty": qty,
                "price": price,
                "notional": notional,
                "timestamp": now,
            }
            fills.append(fill)
            self._fill_log.append(fill)

        return pd.DataFrame(fills) if fills else pd.DataFrame(
            columns=["symbol", "side", "qty", "price", "notional", "timestamp"]
        )

    def fetch_fills(self, since: str) -> pd.DataFrame:
        matching = [f for f in self._fill_log if f.get("timestamp", "") >= since]
        return pd.DataFrame(matching) if matching else pd.DataFrame(
            columns=["symbol", "side", "qty", "price", "notional", "timestamp"]
        )

    # ------------------------------------------------------------------
    # Funding rate application
    # ------------------------------------------------------------------

    def apply_funding(self) -> float:
        """Apply funding rates to open positions. Returns total funding paid/received."""
        total = 0.0
        for symbol, qty in list(self.positions.items()):
            if abs(qty) < 1e-12:
                continue
            price = self.prices.get(symbol, 0.0)
            rate = self.funding_rates.get(symbol, self.default_funding_rate)
            # Longs pay funding when rate > 0, shorts receive (and vice versa)
            funding = -qty * price * rate
            self.margin_balance += funding
            total += funding
        return total
