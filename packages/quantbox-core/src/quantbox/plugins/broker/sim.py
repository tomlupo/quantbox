from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd

from quantbox.contracts import PluginMeta

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

    # Trading fees (spot defaults)
    maker_fee_bps: float = 10.0  # 0.10%
    taker_fee_bps: float = 10.0  # 0.10%
    assume_taker: bool = True
    _cumulative_fees: float = field(default=0.0, repr=False)

    positions: Dict[str, float] = field(default_factory=dict)
    prices: Dict[str, float] = field(default_factory=dict)

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

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        fills = []
        for _, o in orders.iterrows():
            sym = str(o["symbol"])
            side = str(o["side"]).lower()
            qty = float(o["qty"])
            mid_price = float(o.get("price", 0.0) or 0.0)
            if mid_price == 0.0:
                mid_price = self.prices.get(sym, 0.0)
            if mid_price == 0.0:
                continue

            # Slippage model
            direction = 1 if side == "buy" else -1
            cost_bps = (self.spread_bps + self.slippage_bps) / 10_000
            fill_price = mid_price * (1 + direction * cost_bps)

            signed = qty if side == "buy" else -qty
            self.positions[sym] = self.positions.get(sym, 0.0) + signed
            self.cash -= signed * fill_price

            # Trading fee
            notional = abs(signed) * fill_price
            fee_bps = self.taker_fee_bps if self.assume_taker else self.maker_fee_bps
            fee = notional * fee_bps / 10_000
            self.cash -= fee
            self._cumulative_fees += fee

            fills.append({
                "symbol": sym, "side": side, "qty": qty,
                "price": fill_price, "fee": fee,
            })
        return pd.DataFrame(fills)

    def fetch_fills(self, since: str) -> pd.DataFrame:
        return pd.DataFrame([])
