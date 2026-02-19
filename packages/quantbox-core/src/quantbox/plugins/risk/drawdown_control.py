"""Drawdown control risk manager plugin.

Checks current portfolio drawdown against thresholds and can signal
halt or position scaling based on severity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quantbox.contracts import PluginMeta


@dataclass
class DrawdownControlRiskManager:
    """Risk plugin that enforces drawdown limits on portfolio targets.

    The caller passes ``current_drawdown`` in params (typically computed
    from recent history). The plugin checks this against thresholds and
    emits halt or scale findings.

    Params:
        max_drawdown: Worst acceptable drawdown (negative, default -0.20).
        current_drawdown: Current portfolio drawdown (required; negative).
        action: "warn" (default) or "halt".
        scale_threshold: Drawdown level triggering position scaling
            (default same as max_drawdown).
        scale_factor: Factor to scale positions by when scaling (default 0.5).
    """

    meta = PluginMeta(
        name="risk.drawdown_control.v1",
        kind="risk",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Drawdown control risk plugin. Halts or scales positions "
            "when portfolio drawdown exceeds configured thresholds."
        ),
        tags=("risk", "drawdown", "control"),
        capabilities=("backtest", "paper", "live"),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "max_drawdown": {
                    "type": "number",
                    "default": -0.20,
                    "description": "Worst acceptable drawdown (negative number).",
                },
                "current_drawdown": {
                    "type": "number",
                    "description": "Current portfolio drawdown (negative number, required).",
                },
                "action": {
                    "type": "string",
                    "enum": ["warn", "halt"],
                    "default": "warn",
                    "description": "Action on breach: 'warn' or 'halt'.",
                },
                "scale_threshold": {
                    "type": "number",
                    "description": "Drawdown level triggering position scaling (defaults to max_drawdown).",
                },
                "scale_factor": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.5,
                    "description": "Factor to scale positions by when scaling is triggered.",
                },
            },
            "required": ["current_drawdown"],
        },
    )

    def check_targets(
        self,
        targets: pd.DataFrame,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Check current drawdown against halt and scale thresholds."""
        findings: list[dict[str, Any]] = []

        current_drawdown = params.get("current_drawdown")
        if current_drawdown is None:
            return findings

        current_drawdown = float(current_drawdown)
        max_drawdown = float(params.get("max_drawdown", -0.20))
        action = params.get("action", "warn")
        scale_threshold = float(params.get("scale_threshold", max_drawdown))
        scale_factor = float(params.get("scale_factor", 0.5))

        halted = False
        if current_drawdown < max_drawdown:
            if action == "halt":
                findings.append({
                    "level": "error",
                    "rule": "drawdown_halt",
                    "detail": (
                        f"Current drawdown {current_drawdown:.4f} breaches "
                        f"max_drawdown {max_drawdown:.4f}. Action: halt."
                    ),
                })
                halted = True
            else:
                findings.append({
                    "level": "warn",
                    "rule": "drawdown_halt",
                    "detail": (
                        f"Current drawdown {current_drawdown:.4f} breaches "
                        f"max_drawdown {max_drawdown:.4f}. Action: warn."
                    ),
                })

        if current_drawdown < scale_threshold and not halted:
            findings.append({
                "level": "info",
                "rule": "drawdown_scale",
                "detail": (
                    f"Current drawdown {current_drawdown:.4f} breaches "
                    f"scale_threshold {scale_threshold:.4f}. "
                    f"Recommend scaling positions by {scale_factor}."
                ),
            })

        return findings

    def check_orders(
        self,
        orders: pd.DataFrame,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Order checks not applicable for drawdown control; always returns empty."""
        return []
