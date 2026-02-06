from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

from quantbox.contracts import PluginMeta

try:
    from ib_insync import IB, Stock, Contract, MarketOrder, LimitOrder, util
except Exception:  # pragma: no cover
    IB = None  # type: ignore

@dataclass
class IBKRBroker:
    """Interactive Brokers adapter using ib_insync.

    Implements the QuantBox BrokerPlugin interface:
    - get_positions
    - get_cash
    - get_market_snapshot
    - place_orders
    - fetch_fills

    This is a *scaffold*: good enough to start, but you will likely extend:
    - futures contracts (multiplier/exchange/currency)
    - FX conversion
    - smarter market data handling
    - robust reconciliation
    """

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 7
    account: Optional[str] = None
    readonly: bool = False  # set True to block order placement

    meta = PluginMeta(
        name="ibkr.live.v1",
        kind="broker",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Interactive Brokers broker adapter (ib_insync)",
        tags=("ibkr","broker"),
        capabilities=("paper","live","stocks","etfs","futures"),
        schema_version="v1",
        params_schema={
            "type":"object",
            "properties":{
                "host":{"type":"string","default":"127.0.0.1"},
                "port":{"type":"integer","default":7497},
                "client_id":{"type":"integer","default":7},
                "account":{"type":["string","null"]},
                "readonly":{"type":"boolean","default":False}
            },
        },
        examples=(
            "plugins:\n  broker:\n    name: ibkr.live.v1\n    params_init:\n      host: 127.0.0.1\n      port: 7497\n      client_id: 7\n      account: DUXXXX",
        )
    )

    def __post_init__(self):
        if IB is None:
            raise ImportError("ib_insync is required. pip install ib_insync")
        self.ib = IB()
        self.ib.connect(self.host, int(self.port), clientId=int(self.client_id))

    def _ensure_account(self):
        # If account not specified, keep None and rely on default account in TWS/Gateway
        return

    def get_positions(self) -> pd.DataFrame:
        self._ensure_account()
        positions = self.ib.positions()
        rows = []
        for p in positions:
            if self.account and getattr(p, "account", None) != self.account:
                continue
            sym = getattr(p.contract, "symbol", None) or getattr(p.contract, "localSymbol", "")
            rows.append({"symbol": sym, "qty": float(p.position)})
        return pd.DataFrame(rows)

    def get_cash(self) -> Dict[str, float]:
        self._ensure_account()
        # accountSummary returns list of AccountValue
        vals = self.ib.accountSummary()
        cash = {}
        for v in vals:
            if self.account and getattr(v, "account", None) != self.account:
                continue
            if v.tag == "CashBalance":
                try:
                    cash[v.currency] = float(v.value)
                except Exception:
                    pass
        return cash

    def get_market_snapshot(self, symbols: List[str]) -> pd.DataFrame:
        # Snapshot mid using reqMktData; in practice you may need exchange/currency routing.
        rows = []
        for sym in symbols:
            c = Stock(sym, "SMART", "USD")
            self.ib.qualifyContracts(c)
            t = self.ib.reqMktData(c, "", snapshot=True, regulatorySnapshot=False)
            self.ib.sleep(0.2)
            mid = None
            try:
                bid = float(t.bid) if t.bid is not None else None
                ask = float(t.ask) if t.ask is not None else None
                last = float(t.last) if t.last is not None else None
                if bid is not None and ask is not None:
                    mid = (bid + ask) / 2.0
                elif last is not None:
                    mid = last
            except Exception:
                mid = None
            rows.append({"symbol": sym, "mid": mid})
            self.ib.cancelMktData(c)
        return pd.DataFrame(rows)

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        if self.readonly:
            raise PermissionError("readonly broker: order placement disabled")

        fills = []
        for _, o in orders.iterrows():
            sym = str(o["symbol"])
            side = str(o["side"]).lower()
            qty = float(o["qty"])
            price = o.get("price", None)

            contract = Stock(sym, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            action = "BUY" if side == "buy" else "SELL"
            if price is None or (isinstance(price, float) and (price == 0.0)):
                order = MarketOrder(action, qty)
            else:
                order = LimitOrder(action, qty, float(price))

            trade = self.ib.placeOrder(contract, order)
            # wait briefly for fill updates; for real trading you need robust tracking
            self.ib.sleep(0.5)

            # Collect executions (if any)
            for f in trade.fills:
                fills.append({
                    "symbol": sym,
                    "side": side,
                    "qty": float(f.execution.shares),
                    "price": float(f.execution.price),
                })

        return pd.DataFrame(fills)

    def fetch_fills(self, since: str) -> pd.DataFrame:
        # Best-effort: return executions; filtering by since is minimal here
        execs = self.ib.executions()
        rows = []
        for e in execs:
            sym = getattr(e.contract, "symbol", None) or getattr(e.contract, "localSymbol", "")
            side = "buy" if e.execution.side.upper() == "BOT" else "sell"
            rows.append({"symbol": sym, "side": side, "qty": float(e.execution.shares), "price": float(e.execution.price)})
        return pd.DataFrame(rows)
