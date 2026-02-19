# Broker Plugin API

## BrokerPlugin Protocol

Source: `packages/quantbox-core/src/quantbox/contracts.py`

```python
class BrokerPlugin(Protocol):
    meta: PluginMeta

    def get_positions(self) -> pd.DataFrame:
        """Current holdings: DataFrame[symbol, qty, value]."""
        ...

    def get_cash(self) -> dict[str, float]:
        """Cash balances: {"USDT": 100000.0}."""
        ...

    def get_market_snapshot(self, symbols: list[str]) -> pd.DataFrame:
        """Current prices/info for given symbols."""
        ...

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Execute orders. Input columns: [symbol, side, qty].
        Returns fills DataFrame: [symbol, side, qty, price, status]."""
        ...

    def fetch_fills(self, since: str) -> pd.DataFrame:
        """Historical fills since timestamp."""
        ...
```

### Optional Methods (checked via `hasattr`)

```python
def get_equity(self) -> float:
    """Total account value (margin + unrealized PnL).
    Preferred over cash + sum(qty * price) for derivatives."""
    ...

def describe(self) -> dict[str, Any]:
    """Structured state snapshot for LLM inspection.
    Returns: {plugin, paper_trading, quote_currency, positions, ...}"""
    ...
```

## Template

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
import os
import pandas as pd
from quantbox.contracts import PluginMeta

@dataclass
class MyBroker:
    """Broker for MyExchange.

    LLM Note: Requires MY_API_KEY and MY_API_SECRET env vars.
    Use paper_trading=True for testing.
    """

    meta = PluginMeta(
        name="myexchange.live.v1",
        kind="broker",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Live broker for MyExchange",
        tags=("spot", "live"),
        capabilities=("paper", "live"),
    )

    # Constructor params (via params_init)
    api_key: str = ""
    api_secret: str = ""
    paper_trading: bool = True
    quote_currency: str = "USDT"

    # Internal state
    _cash: float = field(default=100000.0, repr=False)
    _positions: Dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.api_key = self.api_key or os.environ.get("MY_API_KEY", "")
        self.api_secret = self.api_secret or os.environ.get("MY_API_SECRET", "")

    def get_positions(self) -> pd.DataFrame:
        rows = [{"symbol": s, "qty": q} for s, q in self._positions.items() if q != 0]
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["symbol", "qty"])

    def get_cash(self) -> Dict[str, float]:
        return {self.quote_currency: self._cash}

    def get_market_snapshot(self, symbols: List[str]) -> pd.DataFrame:
        # Fetch prices from exchange API
        return pd.DataFrame({"symbol": symbols, "price": [0.0] * len(symbols)})

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        fills = []
        for _, row in orders.iterrows():
            # Execute via exchange API (or simulate)
            fills.append({
                "symbol": row["symbol"],
                "side": row["side"],
                "qty": row["qty"],
                "price": 0.0,  # fill price
                "status": "filled",
            })
        return pd.DataFrame(fills)

    def fetch_fills(self, since: str) -> pd.DataFrame:
        return pd.DataFrame(columns=["symbol", "side", "qty", "price", "timestamp"])

    def describe(self) -> Dict[str, Any]:
        return {
            "plugin": self.meta.name,
            "paper_trading": self.paper_trading,
            "cash": self.get_cash(),
            "positions": self._positions,
        }
```

## Registration

1. Create: `packages/quantbox-core/src/quantbox/plugins/broker/my_exchange.py`
2. Export: Add to `plugins/broker/__init__.py`
3. Register in `plugins/builtins.py`:
   ```python
   from .broker import MyBroker
   "broker": _map(..., MyBroker),
   ```

### External

```toml
[project.entry-points."quantbox.brokers"]
"myexchange.live.v1" = "my_pkg.broker:MyBroker"
```
