from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantbox.contracts import PluginMeta


@dataclass
class PaperBrokerStub:
    """
    Binance

    This is a **paper-mode stub** for marketplace scaffolding.
    It behaves like a minimal broker with in-memory state.

    Replace this with real API integration later.
    """

    meta = PluginMeta(
        name="binance.paper.stub.v1",
        kind="broker",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Binance (paper stub)",
        tags=("paper", "stub"),
        capabilities=("paper",),
        schema_version="v1",
        params_schema={"type": "object", "properties": {"starting_cash_usd": {"type": "number", "default": 100000}}},
        examples=(
            "plugins:\n  broker:\n    name: binance.paper.stub.v1\n    params_init:\n      starting_cash_usd: 100000",
        ),
    )
    cash_usd: float = 100000.0
    positions: dict[str, float] = field(default_factory=dict)

    def __init__(self, starting_cash_usd: float = 100000.0):
        self.cash_usd = float(starting_cash_usd)
        self.positions = {}

    def get_positions(self) -> pd.DataFrame:
        return pd.DataFrame([{"symbol": s, "qty": q} for s, q in self.positions.items()])

    def get_cash(self) -> dict[str, float]:
        return {"USD": float(self.cash_usd)}

    def get_market_snapshot(self, symbols: list[str]) -> pd.DataFrame:
        return pd.DataFrame([{"symbol": s, "mid": None} for s in symbols])

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        fills = []
        for _, o in orders.iterrows():
            sym = str(o["symbol"])
            side = str(o["side"]).lower()
            qty = float(o["qty"])
            price = float(o.get("price", 0.0) or 0.0)
            signed = qty if side == "buy" else -qty
            self.positions[sym] = self.positions.get(sym, 0.0) + signed
            self.cash_usd -= signed * price
            fills.append({"symbol": sym, "side": side, "qty": qty, "price": price})
        return pd.DataFrame(fills)

    def fetch_fills(self, since: str) -> pd.DataFrame:
        return pd.DataFrame([])
