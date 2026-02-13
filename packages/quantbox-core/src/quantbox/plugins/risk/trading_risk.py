"""Trading risk manager plugin.

Validates targets (weight-level checks) and orders (order-level checks)
before execution.  Ported from quantlab risk logic embedded in trading.py
and orders.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quantbox.contracts import PluginMeta


@dataclass
class TradingRiskManager:
    meta = PluginMeta(
        name="risk.trading_basic.v1",
        kind="risk",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="Basic trading risk checks: leverage, concentration, negative weights, min notional, max order size.",
        tags=("trading", "risk"),
        capabilities=("paper", "live", "crypto", "etf", "stocks"),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "max_leverage": {
                    "type": "number",
                    "minimum": 0,
                    "default": 1.0,
                    "description": "Maximum sum(abs(weight)). Exceeding triggers a warning.",
                },
                "max_concentration": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.30,
                    "description": "Max absolute weight per single asset.",
                },
                "allow_negative_weights": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether negative (short) weights are allowed.",
                },
                "min_notional": {
                    "type": "number",
                    "minimum": 0,
                    "default": 1.0,
                    "description": "Minimum order notional value (qty * price).",
                },
                "max_order_notional": {
                    "type": "number",
                    "minimum": 0,
                    "default": 0,
                    "description": "Maximum single-order notional. 0 = unlimited.",
                },
            },
        },
        examples=(
            "plugins:\n  risk:\n    - name: risk.trading_basic.v1\n      params:\n        max_leverage: 1.0\n        max_concentration: 0.30\n        allow_negative_weights: false",
        ),
    )

    # ------------------------------------------------------------------
    # check_targets: weight-level validation
    # ------------------------------------------------------------------
    def check_targets(self, targets: pd.DataFrame, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Validate target weights before order generation.

        Expected *targets* columns: ``symbol``, ``weight``.
        Additional columns are ignored.

        Returns a list of finding dicts, each with keys:
            level  – "warn" | "error"
            rule   – short machine-readable rule id
            detail – human-readable message
        """
        findings: list[dict[str, Any]] = []
        if targets is None or targets.empty:
            return findings

        max_leverage = float(params.get("max_leverage", 1.0))
        max_conc = float(params.get("max_concentration", 0.30))
        allow_neg = bool(params.get("allow_negative_weights", False))

        weights = targets["weight"].astype(float) if "weight" in targets.columns else pd.Series(dtype=float)
        if weights.empty:
            return findings

        # --- leverage ---
        gross = weights.abs().sum()
        if gross > max_leverage:
            findings.append(
                {
                    "level": "warn",
                    "rule": "max_leverage_exceeded",
                    "detail": f"Gross leverage {gross:.4f} exceeds limit {max_leverage}.",
                }
            )

        # --- concentration ---
        for idx, w in weights.items():
            if abs(w) > max_conc:
                symbol = targets.at[idx, "symbol"] if "symbol" in targets.columns else str(idx)
                findings.append(
                    {
                        "level": "warn",
                        "rule": "concentration_exceeded",
                        "detail": f"{symbol} weight {w:.4f} exceeds max_concentration {max_conc}.",
                    }
                )

        # --- negative weights ---
        if not allow_neg and (weights < 0).any():
            neg_syms = (
                targets.loc[weights < 0, "symbol"].tolist()
                if "symbol" in targets.columns
                else weights[weights < 0].index.tolist()
            )
            findings.append(
                {
                    "level": "error",
                    "rule": "negative_weight_disallowed",
                    "detail": f"Negative weights found for {neg_syms} but allow_negative_weights=False.",
                }
            )

        return findings

    # ------------------------------------------------------------------
    # check_orders: order-level validation
    # ------------------------------------------------------------------
    def check_orders(self, orders: pd.DataFrame, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Validate orders before execution.

        Expected *orders* columns: ``symbol``, ``side``, ``qty``, ``price``.

        Returns a list of finding dicts (same schema as check_targets).
        """
        findings: list[dict[str, Any]] = []
        if orders is None or orders.empty:
            return findings

        min_notional = float(params.get("min_notional", 1.0))
        max_order_notional = float(params.get("max_order_notional", 0))

        for _, row in orders.iterrows():
            sym = str(row.get("symbol", ""))
            qty = float(row.get("qty", 0))
            price = float(row.get("price", 0))
            notional = qty * price

            if notional < min_notional:
                findings.append(
                    {
                        "level": "warn",
                        "rule": "below_min_notional",
                        "detail": f"{sym} notional {notional:.4f} < min_notional {min_notional}.",
                    }
                )

            if max_order_notional > 0 and notional > max_order_notional:
                findings.append(
                    {
                        "level": "error",
                        "rule": "exceeds_max_order_notional",
                        "detail": f"{sym} notional {notional:.2f} exceeds max_order_notional {max_order_notional}.",
                    }
                )

        return findings
