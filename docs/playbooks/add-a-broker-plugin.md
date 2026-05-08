# Recipe: Add a broker plugin

Two options:

## Option A: Built-in broker (recommended for core plugins)

### 1. Create the module

Add a new file under `packages/quantbox-core/src/quantbox/plugins/broker/`.

```python
# packages/quantbox-core/src/quantbox/plugins/broker/my_exchange.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from quantbox.contracts import PluginMeta


@dataclass
class MyExchangeBroker:
    """Broker plugin for MyExchange.

    LLM Note: This broker requires API_KEY_MYEXCHANGE and API_SECRET_MYEXCHANGE
    environment variables. Use paper_trading=True for testing.
    """

    meta = PluginMeta(
        name="myexchange.live.v1",
        kind="broker",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Live broker for MyExchange spot trading",
        tags=("spot", "live"),
        capabilities=("paper", "live"),
        schema_version="v1",
    )

    # Constructor params (set via params_init in config)
    api_key: str = ""
    api_secret: str = ""
    paper_trading: bool = True
    quote_currency: str = "USDT"

    # Internal state
    positions: Dict[str, float] = field(default_factory=dict)
    _client: Any = field(default=None, repr=False)

    def __post_init__(self):
        import os
        self.api_key = self.api_key or os.environ.get("API_KEY_MYEXCHANGE", "")
        self.api_secret = self.api_secret or os.environ.get("API_SECRET_MYEXCHANGE", "")

    def get_positions(self) -> pd.DataFrame:
        """Return current positions as DataFrame with columns [symbol, qty, value]."""
        rows = [
            {"symbol": s, "qty": q, "value": q * self.prices.get(s, 0)}
            for s, q in self.positions.items()
            if q != 0
        ]
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["symbol", "qty", "value"])

    def get_cash(self) -> Dict[str, float]:
        """Return available cash balances."""
        return {self.quote_currency: self._cash}

    def get_market_snapshot(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch latest prices for given symbols."""
        # Replace with actual API call
        return {s: 100.0 for s in symbols}

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Execute orders. Input columns: [symbol, side, qty]. Returns fills DataFrame."""
        fills = []
        for _, row in orders.iterrows():
            price = self.get_market_snapshot([row["symbol"]])[row["symbol"]]
            fills.append({
                "symbol": row["symbol"],
                "side": row["side"],
                "qty": row["qty"],
                "price": price,
                "status": "filled",
            })
        return pd.DataFrame(fills)

    def fetch_fills(self, since: str) -> pd.DataFrame:
        """Fetch historical fills since the given timestamp."""
        return pd.DataFrame(columns=["symbol", "side", "qty", "price", "timestamp"])

    def describe(self) -> Dict[str, Any]:
        """Return a structured state snapshot (useful for LLM inspection)."""
        return {
            "plugin": self.meta.name,
            "paper_trading": self.paper_trading,
            "quote_currency": self.quote_currency,
            "positions": dict(self.positions),
        }
```

### 2. Export from `__init__.py`

```python
# In plugins/broker/__init__.py, add:
from .my_exchange import MyExchangeBroker
```

### 3. Register in builtins

```python
# In plugins/builtins.py, add to imports:
from .broker import MyExchangeBroker

# Add to the "broker" line in builtins():
"broker": _map(..., MyExchangeBroker),
```

### 4. Add example config

```yaml
# configs/run_myexchange_paper.yaml
run:
  mode: paper
  asof: "2026-02-01"
  pipeline: "trade.full_pipeline.v1"

plugins:
  broker:
    name: "myexchange.live.v1"
    params_init:
      paper_trading: true
      quote_currency: "USDT"
```

### 5. Verify

```bash
uv run quantbox plugins list          # Should show myexchange.live.v1
uv run quantbox plugins info --name myexchange.live.v1
uv run pytest -q                      # All tests still pass
```

## Option B: External broker (separate repo/package)

Entry point group in your package's `pyproject.toml`:

```toml
[project.entry-points."quantbox.brokers"]
"myexchange.live.v1" = "quantbox_plugin_myexchange.broker:MyExchangeBroker"
```

Implement the same methods as above. The plugin will be discovered automatically
when the package is installed in the same environment as quantbox.

Keep secrets out of configs: read from env vars / secret refs.
