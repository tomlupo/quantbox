from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import pandas as pd

from quantbox.contracts import PluginMeta

try:
    from binance.client import Client
except Exception:  # pragma: no cover
    Client = None  # type: ignore

@dataclass
class BinanceBroker:
    """Binance adapter scaffold using python-binance.

    Interface:
    - get_positions: spot balances as positions (free+locked)
    - get_cash: returns USDT as USD proxy (and others)
    - get_market_snapshot: mid from ticker price (best-effort)
    - place_orders: MARKET orders by default, LIMIT if price provided
    - fetch_fills: returns recent trades (best-effort)
    """

    api_key_env: str = "BINANCE_API_KEY"
    api_secret_env: str = "BINANCE_API_SECRET"
    testnet: bool = False
    readonly: bool = False

    meta = PluginMeta(
        name="binance.live.v1",
        kind="broker",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Binance broker adapter (python-binance)",
        tags=("binance","broker","crypto"),
        capabilities=("paper","live","crypto"),
        schema_version="v1",
        params_schema={
            "type":"object",
            "properties":{
                "api_key_env":{"type":"string","default":"BINANCE_API_KEY"},
                "api_secret_env":{"type":"string","default":"BINANCE_API_SECRET"},
                "testnet":{"type":"boolean","default":False},
                "readonly":{"type":"boolean","default":False}
            }
        },
        examples=(
            "plugins:\n  broker:\n    name: binance.live.v1\n    params_init:\n      api_key_env: BINANCE_API_KEY\n      api_secret_env: BINANCE_API_SECRET\n      testnet: false",
        )
    )

    def __post_init__(self):
        if Client is None:
            raise ImportError("python-binance is required. pip install python-binance")
        api_key = os.environ.get(self.api_key_env)
        api_secret = os.environ.get(self.api_secret_env)
        if not api_key or not api_secret:
            raise EnvironmentError(f"Missing env vars: {self.api_key_env} / {self.api_secret_env}")
        self.client = Client(api_key, api_secret, testnet=self.testnet)

    def get_positions(self) -> pd.DataFrame:
        acc = self.client.get_account()
        rows = []
        for b in acc.get("balances", []):
            asset = b["asset"]
            qty = float(b.get("free", 0.0)) + float(b.get("locked", 0.0))
            if qty != 0.0:
                rows.append({"symbol": asset, "qty": qty})
        return pd.DataFrame(rows)

    def get_cash(self) -> Dict[str, float]:
        # treat stablecoins as cash; return dict of balances
        pos = self.get_positions()
        cash = {}
        for _, r in pos.iterrows():
            sym = str(r["symbol"])
            if sym in ("USDT","USD","BUSD","USDC"):
                cash[sym] = float(r["qty"])
        return cash

    def get_market_snapshot(self, symbols: List[str]) -> pd.DataFrame:
        # Binance uses trading pairs (e.g., BTCUSDT). User must pass valid symbols.
        rows = []
        for sym in symbols:
            try:
                t = self.client.get_symbol_ticker(symbol=sym)
                mid = float(t["price"])
            except Exception:
                mid = None
            rows.append({"symbol": sym, "mid": mid})
        return pd.DataFrame(rows)

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        if self.readonly:
            raise PermissionError("readonly broker: order placement disabled")
        fills = []
        for _, o in orders.iterrows():
            sym = str(o["symbol"])
            side = str(o["side"]).upper()
            qty = float(o["qty"])
            price = o.get("price", None)

            try:
                if price is None or (isinstance(price, float) and price == 0.0):
                    res = self.client.create_order(symbol=sym, side=side.upper(), type="MARKET", quantity=qty)
                else:
                    res = self.client.create_order(symbol=sym, side=side.upper(), type="LIMIT", timeInForce="GTC", quantity=qty, price=str(price))
            except Exception:
                continue

            # best-effort: parse fills
            for f in res.get("fills", []) or []:
                fills.append({"symbol": sym, "side": side.lower(), "qty": float(f.get("qty", 0.0)), "price": float(f.get("price", 0.0))})

        return pd.DataFrame(fills)

    def fetch_fills(self, since: str) -> pd.DataFrame:
        # best-effort: no timestamp parsing here; user can extend.
        return pd.DataFrame([])
