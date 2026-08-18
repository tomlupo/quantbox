"""Futures paper trading broker.

Simulates a perpetual-futures account with:
- Signed positions (positive = long, negative = short)
- Margin accounting (initial capital is the portfolio value)
- Optional funding rate simulation
- Configurable leverage
- Volume-dependent price impact
- JSON state persistence across runs
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quantbox.contracts import PluginMeta

logger = logging.getLogger(__name__)


def _skipped(symbol: str, side: str, qty: float, reason: str) -> dict[str, Any]:
    """A row for an order the broker intentionally did not place.

    The pipeline distinguishes three outcomes — filled, FAILED, and an
    intentional broker-side no-op — and a broker signals the third by returning
    a row with ``status="SKIPPED"`` (``trading_pipeline.py`` treats it as
    "neither executed nor failed: record for visibility but do not count it").
    Returning nothing instead makes the order vanish from ``orders_details`` and
    leaves a dangling entry in the per-(symbol, side) intent FIFO, which reads
    downstream as a MISSED FILL rather than a deliberate decline.

    Shape mirrors the Kraken broker's skip row: the REQUESTED quantity is kept
    (it is what the book wanted), price is 0.0 because nothing traded.
    """
    return {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": 0.0,
        "notional": 0.0,
        "fee": 0.0,
        "status": "SKIPPED",
        "error": reason,
    }


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

    # Slippage model
    spread_bps: float = 2.0  # half-spread in basis points (0.02%)
    slippage_bps: float = 5.0  # market impact in basis points (0.05%)

    # Volume-dependent price impact
    impact_factor: float = 0.01  # bps per $10k notional
    max_impact_bps: float = 20.0  # cap on price impact

    # Trading fees
    maker_fee_bps: float = 2.0  # 0.02%
    taker_fee_bps: float = 4.0  # 0.04%
    assume_taker: bool = True
    _cumulative_fees: float = field(default=0.0, repr=False)

    # State persistence
    state_file: str | None = None  # Path to JSON state file

    # Position limits: symbol -> max notional USD
    position_limits: dict[str, float] = field(default_factory=dict)
    default_max_notional: float = 1_000_000.0

    # Positions: symbol -> signed qty (+ long, - short)
    positions: dict[str, float] = field(default_factory=dict)
    # Entry prices: symbol -> avg entry price
    entry_prices: dict[str, float] = field(default_factory=dict)
    # Current market prices: symbol -> price (injected by pipeline or seeded)
    prices: dict[str, float] = field(default_factory=dict)
    # Funding rates: symbol -> rate per snapshot (default 0.01% / 8h)
    funding_rates: dict[str, float] = field(default_factory=dict)
    default_funding_rate: float = 0.0001  # 0.01% per 8h

    # Internal fill log
    _fill_log: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Load persisted state if state_file exists."""
        if self.state_file:
            self._load_state()

    # ------------------------------------------------------------------
    # Price / funding injection
    # ------------------------------------------------------------------

    def set_prices(self, prices: dict[str, float]) -> None:
        """Inject latest market prices (called by pipeline before rebalancing)."""
        self.prices.update(prices)

    def set_funding_rates(self, rates: dict[str, float]) -> None:
        """Inject per-symbol funding rates."""
        self.funding_rates.update(rates)

    def set_position_limits(self, limits: dict[str, float]) -> None:
        """Inject per-symbol max-notional position limits."""
        self.position_limits.update(limits)

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
            rows.append(
                {
                    "symbol": symbol,
                    "qty": qty,
                    "notional": notional,
                    "entry_price": entry,
                    "unrealized_pnl": unrealized_pnl,
                }
            )
        return (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(columns=["symbol", "qty", "notional", "entry_price", "unrealized_pnl"])
        )

    def get_cash(self) -> dict[str, float]:
        return {self.quote_currency: self.margin_balance}

    def get_market_snapshot(self, symbols: list[str]) -> pd.DataFrame:
        rows = []
        for s in symbols:
            rows.append(
                {
                    "symbol": s,
                    "mid": self.prices.get(s),
                    "min_qty": 0.0,
                    "step_size": 0.0,
                    "min_notional": 1.0,
                }
            )
        return pd.DataFrame(rows)

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        fills: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc).isoformat()

        for _, o in orders.iterrows():
            sym = str(o["symbol"])
            side = str(o["side"]).lower()
            qty = float(o["qty"])
            mid_price = self.prices.get(sym, float(o.get("price", 0.0) or 0.0))
            # Optional column, NaN-safe (a mixed-column frame fills it NaN;
            # bool(NaN) is True), mirroring the live Hyperliquid broker.
            _ro = o.get("reduce_only", False)
            reduce_only = bool(_ro) if pd.notna(_ro) else False

            if not (mid_price > 0):
                logger.warning("Invalid price (%r) for %s, skipping order", mid_price, sym)
                continue

            signed = qty if side == "buy" else -qty
            old_qty = self.positions.get(sym, 0.0)

            # Venue semantics: a reduceOnly order can never open a position and
            # can never flip one through zero. Clamp before the slippage model so
            # the impact/fee/fill quantities describe the order actually filled.
            if reduce_only:
                if abs(old_qty) < 1e-12:
                    logger.warning("reduce_only order on flat %s, skipping", sym)
                    fills.append(_skipped(sym, side, qty, "reduce_only on a flat position"))
                    continue
                if signed * old_qty > 0:
                    logger.warning("reduce_only order would increase %s exposure, skipping", sym)
                    fills.append(_skipped(sym, side, qty, "reduce_only would increase exposure"))
                    continue
                if abs(signed) > abs(old_qty):
                    logger.info(
                        "reduce_only: clamped %s qty from %.4f to %.4f",
                        sym,
                        abs(signed),
                        abs(old_qty),
                    )
                    signed = -old_qty
                    qty = abs(signed)

            # ── Position limit ─────────────────────────────────────────────
            # Runs BEFORE the slippage model, and against mid_price, for two
            # reasons: every downstream number (impact, fee, reported qty) must
            # describe the quantity actually filled, and a risk limit should not
            # move with slippage.
            #
            # TOM-886: the old form sized the fill *to* the limit from a ZERO
            # base — `allowed * sign(signed) - old_qty` — so for an order moving
            # toward zero the terms ADDED. old=+100, req=-30, limit=50 came out
            # as -150 and landed at -50: a 30-unit trim turned into a 150-unit
            # sell that reversed the book. Two invariants now hold by
            # construction, and they are the whole fix:
            #
            #   1. a limit may only ever SHRINK an order, never grow it;
            #   2. a limit may never change the direction the order was going.
            #
            # It also only engages on an order that INCREASES exposure. An order
            # already moving toward compliance is never "capped" — that was the
            # case the old algebra inverted.
            if not reduce_only:
                max_notional = self.position_limits.get(sym, self.default_max_notional)
                projected_notional = abs(old_qty + signed) * mid_price
                current_notional = abs(old_qty) * mid_price
                if projected_notional > max_notional and projected_notional > current_notional:
                    allowed_abs = max_notional / mid_price
                    # Where the book lands if it obeys the limit, on the side the
                    # order is heading for.
                    target_new = allowed_abs * (1.0 if (old_qty + signed) > 0 else -1.0)
                    capped_signed = target_new - old_qty

                    if capped_signed * signed <= 0:
                        # Obeying the limit would mean trading the OTHER way.
                        # That is not a cap, so refuse rather than invent a trade
                        # nobody asked for.
                        logger.warning(
                            "Position limit reached for %s (holding %.4f, limit %.4f units) — skipping",
                            sym,
                            old_qty,
                            allowed_abs,
                        )
                        fills.append(_skipped(sym, side, qty, "position limit reached"))
                        continue

                    # Invariant 1: shrink only. Invariant 2: keep the direction.
                    capped_signed = (1.0 if signed > 0 else -1.0) * min(abs(capped_signed), abs(signed))

                    if abs(capped_signed) < 1e-12:
                        logger.warning("Position limit reached for %s, skipping", sym)
                        fills.append(_skipped(sym, side, qty, "position limit reached"))
                        continue

                    logger.info(
                        "Position limit: capped %s qty from %.4f to %.4f",
                        sym,
                        abs(signed),
                        abs(capped_signed),
                    )
                    signed = capped_signed
                    qty = abs(signed)

            # Slippage model: spread + slippage + volume-dependent impact.
            # Sits AFTER both clamps so it prices the order actually filled.
            direction = 1 if side == "buy" else -1
            notional_est = qty * mid_price
            impact_bps = min(notional_est * self.impact_factor / 10_000, self.max_impact_bps)
            cost_bps = (self.spread_bps + self.slippage_bps + impact_bps) / 10_000
            fill_price = mid_price * (1 + direction * cost_bps)

            # Update position with weighted-average entry price
            old_entry = self.entry_prices.get(sym, 0.0)
            new_qty = old_qty + signed

            if abs(new_qty) < 1e-12:
                # Flat — realise PnL
                realised = old_qty * (fill_price - old_entry) if old_entry else 0.0
                self.margin_balance += realised
                self.positions.pop(sym, None)
                self.entry_prices.pop(sym, None)
            elif (old_qty >= 0 and signed > 0) or (old_qty <= 0 and signed < 0):
                # Adding to position — weighted average entry
                if abs(old_qty) + abs(signed) > 0:
                    self.entry_prices[sym] = (abs(old_qty) * old_entry + abs(signed) * fill_price) / (
                        abs(old_qty) + abs(signed)
                    )
                self.positions[sym] = new_qty
            else:
                # Partial close or flip
                close_qty = min(abs(signed), abs(old_qty))
                realised = close_qty * (fill_price - old_entry) * (1 if old_qty > 0 else -1)
                self.margin_balance += realised

                remainder = abs(signed) - close_qty
                if remainder < 1e-12:
                    self.positions[sym] = new_qty
                else:
                    # Flipped direction
                    self.positions[sym] = new_qty
                    self.entry_prices[sym] = fill_price

            notional = abs(signed) * fill_price

            # Trading fee
            fee_bps = self.taker_fee_bps if self.assume_taker else self.maker_fee_bps
            fee = notional * fee_bps / 10_000
            self.margin_balance -= fee
            self._cumulative_fees += fee

            fill = {
                "symbol": sym,
                "side": side,
                "qty": qty,
                "price": fill_price,
                "notional": notional,
                "fee": fee,
                "timestamp": now,
            }
            fills.append(fill)
            self._fill_log.append(fill)

        # Persist state after order execution
        self._save_state()

        return (
            pd.DataFrame(fills)
            if fills
            else pd.DataFrame(
                columns=["symbol", "side", "qty", "price", "notional", "fee", "timestamp", "status", "error"]
            )
        )

    def fetch_fills(self, since: str) -> pd.DataFrame:
        matching = [f for f in self._fill_log if f.get("timestamp", "") >= since]
        return (
            pd.DataFrame(matching)
            if matching
            else pd.DataFrame(columns=["symbol", "side", "qty", "price", "notional", "timestamp"])
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
        self._save_state()
        return total

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Persist broker state to JSON file."""
        if not self.state_file:
            return
        state = {
            "margin_balance": self.margin_balance,
            "quote_currency": self.quote_currency,
            "positions": self.positions,
            "entry_prices": self.entry_prices,
            "cumulative_fees": self._cumulative_fees,
            "fill_log": self._fill_log[-1000:],  # Keep last 1000 fills
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        path = Path(self.state_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(state, indent=2, default=str))
        except Exception as exc:
            logger.warning("Failed to save futures broker state: %s", exc)

    def _load_state(self) -> None:
        """Load broker state from JSON file if it exists."""
        if not self.state_file:
            return
        path = Path(self.state_file)
        if not path.exists():
            return
        try:
            state = json.loads(path.read_text())
            loaded_margin = float(state.get("margin_balance", self.margin_balance))
            if math.isfinite(loaded_margin):
                self.margin_balance = loaded_margin
            else:
                logger.warning(
                    "Persisted margin_balance is non-finite (%r); rejecting and keeping seed value %.2f",
                    loaded_margin,
                    self.margin_balance,
                )
            self.positions = {str(k): float(v) for k, v in state.get("positions", {}).items()}
            self.entry_prices = {str(k): float(v) for k, v in state.get("entry_prices", {}).items()}
            loaded_fees = float(state.get("cumulative_fees", 0.0))
            if math.isfinite(loaded_fees):
                self._cumulative_fees = loaded_fees
            else:
                logger.warning(
                    "Persisted cumulative_fees is non-finite (%r); rejecting and keeping 0.0",
                    loaded_fees,
                )
                self._cumulative_fees = 0.0
            self._fill_log = state.get("fill_log", [])
            logger.info(
                "Loaded futures broker state: margin=%.2f, %d positions, %d fills",
                self.margin_balance,
                len(self.positions),
                len(self._fill_log),
            )
        except Exception as exc:
            logger.warning("Failed to load futures broker state: %s", exc)
