# Data Plugin API

## DataPlugin Protocol

Source: `packages/quantbox-core/src/quantbox/contracts.py`

```python
class DataPlugin(Protocol):
    meta: PluginMeta

    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        """Returns DataFrame with 'symbol' column listing tradable assets."""
        ...

    def load_market_data(
        self, universe: pd.DataFrame, asof: str, params: dict[str, Any]
    ) -> dict[str, pd.DataFrame]:
        """Returns dict of wide DataFrames.
        Required: "prices" (close prices).
        Optional: "volume", "market_cap", "funding_rates".
        All DataFrames: DatetimeIndex x symbol columns."""
        ...

    def load_fx(self, asof: str, params: dict[str, Any]) -> pd.DataFrame | None:
        """FX rates DataFrame, or None for crypto (all quoted in same currency)."""
        ...
```

## Template

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pandas as pd
from quantbox.contracts import PluginMeta

@dataclass
class MyDataPlugin:
    """Loads market data from MySource.

    LLM Note: Requires MY_API_KEY environment variable for authentication.
    """

    meta = PluginMeta(
        name="my.data_source.v1",
        kind="data",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Market data from MySource API",
        tags=("equities", "live"),
    )

    # Constructor params (set via params_init in config)
    api_key_env: str = "MY_API_KEY"
    quote_currency: str = "USD"

    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        symbols = params.get("universe", {}).get("symbols", [])
        return pd.DataFrame({"symbol": symbols})

    def load_market_data(
        self, universe: pd.DataFrame, asof: str, params: dict[str, Any]
    ) -> dict[str, pd.DataFrame]:
        symbols = universe["symbol"].tolist()
        lookback = params.get("lookback_days", 365)

        # Fetch data from your source
        prices = ...  # Must be wide: DatetimeIndex x symbol columns
        volume = ...

        return {
            "prices": prices,
            "volume": volume,
            "market_cap": pd.DataFrame(),  # empty if not available
        }

    def load_fx(self, asof: str, params: dict[str, Any]) -> pd.DataFrame | None:
        return None  # Not needed for single-currency
```

## Registration

1. Create: `packages/quantbox-core/src/quantbox/plugins/datasources/my_source.py`
2. Export: Add to `plugins/datasources/__init__.py`
3. Register in `plugins/builtins.py`:
   ```python
   from .datasources import MyDataPlugin
   "data": _map(..., MyDataPlugin),
   ```

### External

```toml
[project.entry-points."quantbox.data"]
"my.data_source.v1" = "my_pkg.data:MyDataPlugin"
```
