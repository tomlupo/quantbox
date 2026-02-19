# Risk Plugin API

## RiskPlugin Protocol

Source: `packages/quantbox-core/src/quantbox/contracts.py`

```python
class RiskPlugin(Protocol):
    meta: PluginMeta

    def check_targets(
        self, targets: pd.DataFrame, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Validate weight targets before order generation.
        Returns list of findings (empty = all clear).

        Finding format:
        {"severity": "warning"|"error", "message": "...", "symbol": "...", "value": ...}
        """
        ...

    def check_orders(
        self, orders: pd.DataFrame, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Validate generated orders before execution.
        Returns list of findings (empty = all clear)."""
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
class MyRiskManager:
    """Custom risk validation.

    LLM Note: Add risk checks specific to your trading strategy.
    """

    meta = PluginMeta(
        name="risk.my_checks.v1",
        kind="risk",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Custom risk validation for my strategy",
        tags=("risk",),
    )

    def check_targets(
        self, targets: pd.DataFrame, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        findings = []
        max_pct = params.get("max_position_pct", 0.25)

        # Check each position weight
        latest = targets.iloc[-1] if not targets.empty else pd.Series()
        for symbol, weight in latest.items():
            if abs(weight) > max_pct:
                findings.append({
                    "severity": "error",
                    "message": f"{symbol} weight {weight:.2%} exceeds {max_pct:.0%} limit",
                    "symbol": symbol,
                    "value": weight,
                })

        return findings

    def check_orders(
        self, orders: pd.DataFrame, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        findings = []
        max_notional = params.get("max_notional", 50000)

        for _, order in orders.iterrows():
            notional = order.get("qty", 0) * order.get("price", 0)
            if notional > max_notional:
                findings.append({
                    "severity": "warning",
                    "message": f"{order['symbol']} notional {notional:.0f} exceeds limit",
                    "symbol": order["symbol"],
                    "value": notional,
                })

        return findings
```

## Registration

1. Create: `packages/quantbox-core/src/quantbox/plugins/risk/my_risk.py`
2. Export: Add to `plugins/risk/__init__.py`
3. Register in `plugins/builtins.py`:
   ```python
   from .risk import MyRiskManager
   "risk": _map(..., MyRiskManager),
   ```

### External

```toml
[project.entry-points."quantbox.risk"]
"risk.my_checks.v1" = "my_pkg.risk:MyRiskManager"
```
