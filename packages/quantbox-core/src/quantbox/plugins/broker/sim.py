from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from quantbox.contracts import PluginMeta

logger = logging.getLogger(__name__)


@dataclass
class SimPaperBroker:
    meta = PluginMeta(
        name="sim.paper.v1",
        kind="broker",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Simple paper broker simulator for spot trading",
        tags=("paper",),
        capabilities=("paper",),
        schema_version="v1",
    )
    cash: float = 100_000.0
    quote_currency: str = "USDT"

    # Slippage model
    spread_bps: float = 2.0     # half-spread in basis points (0.02%)
    slippage_bps: float = 5.0   # market impact in basis points (0.05%)

    # Volume-dependent price impact
    impact_factor: float = 0.01   # bps per $10k notional
    max_impact_bps: float = 20.0  # cap on price impact

    # Trading fees (spot defaults)
    maker_fee_bps: float = 10.0  # 0.10%
    taker_fee_bps: float = 10.0  # 0.10%
    assume_taker: bool = True
    _cumulative_fees: float = field(default=0.0, repr=False)

    # State persistence
    state_file: Optional[str] = None  # Path to JSON state file

    positions: Dict[str, float] = field(default_factory=dict)
    prices: Dict[str, float] = field(default_factory=dict)
    _fill_log: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Load persisted state if state_file exists."""
        if self.state_file:
            self._load_state()

    def set_prices(self, prices: Dict[str, float]) -> None:
        self.prices.update(prices)

    def get_positions(self) -> pd.DataFrame:
        return pd.DataFrame([{"symbol": s, "qty": q} for s, q in self.positions.items()])

    def get_cash(self) -> Dict[str, float]:
        return {self.quote_currency: float(self.cash)}

    def get_market_snapshot(self, symbols: List[str]) -> pd.DataFrame:
        rows = []
        for s in symbols:
            rows.append({"symbol": s, "mid": self.prices.get(s)})
        return pd.DataFrame(rows)

    def _compute_impact_bps(self, notional: float) -> float:
        """Volume-dependent price impact: scales with order notional."""
        return min(notional * self.impact_factor / 10_000, self.max_impact_bps)

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        fills = []
        now = datetime.now(timezone.utc).isoformat()

        for _, o in orders.iterrows():
            sym = str(o["symbol"])
            side = str(o["side"]).lower()
            qty = float(o["qty"])
            mid_price = float(o.get("price", 0.0) or 0.0)
            if mid_price == 0.0:
                mid_price = self.prices.get(sym, 0.0)
            if mid_price == 0.0:
                continue

            # Slippage model: spread + slippage + volume-dependent impact
            direction = 1 if side == "buy" else -1
            notional = qty * mid_price
            impact_bps = self._compute_impact_bps(notional)
            cost_bps = (self.spread_bps + self.slippage_bps + impact_bps) / 10_000
            fill_price = mid_price * (1 + direction * cost_bps)

            signed = qty if side == "buy" else -qty
            self.positions[sym] = self.positions.get(sym, 0.0) + signed
            self.cash -= signed * fill_price

            # Trading fee
            fill_notional = abs(signed) * fill_price
            fee_bps = self.taker_fee_bps if self.assume_taker else self.maker_fee_bps
            fee = fill_notional * fee_bps / 10_000
            self.cash -= fee
            self._cumulative_fees += fee

            fill = {
                "symbol": sym, "side": side, "qty": qty,
                "price": fill_price, "fee": fee, "timestamp": now,
            }
            fills.append(fill)
            self._fill_log.append(fill)

        # Persist state after order execution
        self._save_state()

        return pd.DataFrame(fills)

    def fetch_fills(self, since: str) -> pd.DataFrame:
        matching = [f for f in self._fill_log if f.get("timestamp", "") >= since]
        return pd.DataFrame(matching) if matching else pd.DataFrame(
            columns=["symbol", "side", "qty", "price", "fee", "timestamp"]
        )

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Persist broker state to JSON file."""
        if not self.state_file:
            return
        state = {
            "cash": self.cash,
            "quote_currency": self.quote_currency,
            "positions": self.positions,
            "cumulative_fees": self._cumulative_fees,
            "fill_log": self._fill_log[-1000:],  # Keep last 1000 fills
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        path = Path(self.state_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(state, indent=2, default=str))
        except Exception as exc:
            logger.warning("Failed to save broker state: %s", exc)

    def _load_state(self) -> None:
        """Load broker state from JSON file if it exists."""
        if not self.state_file:
            return
        path = Path(self.state_file)
        if not path.exists():
            return
        try:
            state = json.loads(path.read_text())
            self.cash = float(state.get("cash", self.cash))
            self.positions = {
                str(k): float(v) for k, v in state.get("positions", {}).items()
            }
            self._cumulative_fees = float(state.get("cumulative_fees", 0.0))
            self._fill_log = state.get("fill_log", [])
            logger.info(
                "Loaded broker state: cash=%.2f, %d positions, %d fills",
                self.cash, len(self.positions), len(self._fill_log),
            )
        except Exception as exc:
            logger.warning("Failed to load broker state: %s", exc)
