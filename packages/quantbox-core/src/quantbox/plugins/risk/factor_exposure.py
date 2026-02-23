"""Factor exposure risk manager plugin.

Validates portfolio targets against single-position weight limits
and sector concentration constraints.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from quantbox.contracts import PluginMeta


@dataclass
class FactorExposureRiskManager:
    """Validates targets against position weight and sector concentration limits.

    Params:
        max_single_weight: Maximum absolute weight for any single position (default 0.5).
        sectors: Dict mapping symbol to sector name (optional).
        max_sector_weight: Maximum total absolute weight per sector (default 0.5).
    """

    meta = PluginMeta(
        name="risk.factor_exposure.v1",
        kind="risk",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Factor exposure risk plugin. Checks single-position weight limits "
            "and sector concentration constraints."
        ),
        tags=("risk", "factor", "concentration"),
        capabilities=("backtest", "paper", "live"),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "max_single_weight": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.5,
                    "description": "Maximum absolute weight for any single position.",
                },
                "sectors": {
                    "type": "object",
                    "description": "Mapping of symbol to sector name.",
                },
                "max_sector_weight": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.5,
                    "description": "Maximum total absolute weight per sector.",
                },
            },
        },
    )

    def check_targets(
        self,
        targets: pd.DataFrame,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Validate target weights against position and sector limits."""
        findings: list[dict[str, Any]] = []
        if targets is None or targets.empty:
            return findings

        max_single = float(params.get("max_single_weight", 0.5))

        weights = targets["weight"].astype(float) if "weight" in targets.columns else pd.Series(dtype=float)
        if weights.empty:
            return findings

        symbols = (
            targets["symbol"].tolist()
            if "symbol" in targets.columns
            else [str(i) for i in range(len(targets))]
        )

        for symbol, weight in zip(symbols, weights, strict=False):
            if abs(weight) > max_single:
                findings.append({
                    "level": "warn",
                    "rule": "single_weight_exceeded",
                    "detail": (
                        f"{symbol} weight {weight:.4f} exceeds "
                        f"max_single_weight {max_single}."
                    ),
                })

        sectors = params.get("sectors")
        if sectors:
            max_sector = float(params.get("max_sector_weight", 0.5))
            sector_totals: dict[str, float] = defaultdict(float)

            for symbol, weight in zip(symbols, weights, strict=False):
                sector = sectors.get(symbol)
                if sector:
                    sector_totals[sector] += abs(weight)

            for sector, total in sector_totals.items():
                if total > max_sector:
                    findings.append({
                        "level": "warn",
                        "rule": "sector_concentration_exceeded",
                        "detail": (
                            f"Sector '{sector}' total weight {total:.4f} exceeds "
                            f"max_sector_weight {max_sector}."
                        ),
                    })

        return findings

    def check_orders(
        self,
        orders: pd.DataFrame,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Order checks delegated to trading_basic; always returns empty."""
        return []
