# Strategy Plugin API

## StrategyPlugin Protocol

Source: `packages/quantbox-core/src/quantbox/contracts.py`

```python
class StrategyPlugin(Protocol):
    meta: PluginMeta

    def run(
        self,
        data: dict[str, Any],        # {"prices": df, "volume": df, "market_cap": df, "universe": df}
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Returns dict with at minimum "weights" (wide DataFrame: date x symbols)."""
        ...
```

### Input: `data` dict

Wide-format DataFrames (DatetimeIndex x symbol columns):
- `"prices"` (required) - close prices
- `"volume"` (optional) - trading volume
- `"market_cap"` (optional) - market capitalization
- `"funding_rates"` (optional) - funding rates for perps
- `"universe"` (optional) - boolean mask of tradable symbols

### Output: dict

Must contain `"weights"` key with a wide-format DataFrame (date index x symbol columns).
Optionally include `"simple_weights"` (latest weights as dict) and `"details"` (intermediate data).

## PluginMeta

```python
@dataclass(frozen=True)
class PluginMeta:
    name: str                              # e.g. "strategy.my_strategy.v1"
    kind: PluginKind                       # Must be "strategy"
    version: str                           # Semver, e.g. "0.1.0"
    core_compat: str                       # e.g. ">=0.1,<0.2"
    description: str = ""                  # Human/LLM-readable description
    tags: tuple[str, ...] = ()             # Searchable tags
    capabilities: tuple[str, ...] = ()     # Supported modes/features
    schema_version: str = "v1"
    params_schema: dict | None = None      # JSON Schema for params
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
```

## Template

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from quantbox.contracts import PluginMeta

@dataclass
class MyStrategy:
    """Short description of strategy logic.

    LLM Note: Explain the core algorithm and key parameters.
    """

    meta = PluginMeta(
        name="strategy.my_strategy.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="One-line description of strategy",
        tags=("crypto", "trend"),
    )

    # Constructor params (set via params or params_init in config)
    lookback_days: int = 90
    vol_target: float = 0.25

    def run(
        self,
        data: dict[str, pd.DataFrame],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prices = data["prices"]

        # Apply param overrides
        if params:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)

        # --- Your strategy logic here ---
        # Must produce a wide DataFrame: date index x symbol columns
        # Values are portfolio weights (typically sum to 1.0)
        weights = pd.DataFrame(...)

        return {
            "weights": weights,
            "simple_weights": weights.iloc[-1].dropna().to_dict(),
        }

    def describe(self) -> dict[str, Any]:
        """LLM-friendly state snapshot."""
        return {
            "name": self.meta.name,
            "parameters": {"lookback_days": self.lookback_days},
        }
```

## Registration

### Built-in

1. Create: `packages/quantbox-core/src/quantbox/plugins/strategies/my_strategy.py`
2. Export: Add `from .my_strategy import MyStrategy` to `plugins/strategies/__init__.py`
3. Register: In `plugins/builtins.py`:
   ```python
   from .strategies import MyStrategy
   # In builtins():
   "strategy": _map(..., MyStrategy),
   ```

### External (separate package)

```toml
[project.entry-points."quantbox.strategies"]
"strategy.my_strategy.v1" = "my_pkg.strategy:MyStrategy"
```

## Naming Convention

Format: `strategy.<descriptive_name>.v<version>`

Examples: `strategy.crypto_trend.v1`, `strategy.momentum_long_short.v1`
