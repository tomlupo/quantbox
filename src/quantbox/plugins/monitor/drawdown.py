"""Drawdown monitor plugin.

Checks run result metrics for max drawdown and total return breaches,
emitting alerts when thresholds are exceeded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quantbox.contracts import PluginMeta, RunResult


@dataclass
class DrawdownMonitor:
    """Monitor that alerts when drawdown or total loss exceeds thresholds.

    Params:
        max_drawdown: Worst acceptable drawdown (negative number, default -0.20).
        max_loss: Worst acceptable total return (negative number, default -0.30).
        action: "warn" (default) or "halt". Halt escalates level to "error".
    """

    meta = PluginMeta(
        name="monitor.drawdown.v1",
        kind="monitor",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Monitors run results for excessive drawdown and total loss. "
            "Emits warn or error alerts when thresholds are breached."
        ),
        tags=("monitor", "drawdown", "risk"),
        capabilities=("backtest", "paper", "live"),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "max_drawdown": {
                    "type": "number",
                    "default": -0.20,
                    "description": "Worst acceptable max drawdown (negative number).",
                },
                "max_loss": {
                    "type": "number",
                    "default": -0.30,
                    "description": "Worst acceptable total return (negative number).",
                },
                "action": {
                    "type": "string",
                    "enum": ["warn", "halt"],
                    "default": "warn",
                    "description": "Action to take: 'warn' or 'halt' (escalates to error).",
                },
            },
        },
    )

    def check(
        self,
        result: RunResult,
        history: list[RunResult] | None,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Check run result metrics against drawdown and loss thresholds."""
        alerts: list[dict[str, Any]] = []
        action = params.get("action", "warn")
        level = "error" if action == "halt" else "warn"

        max_drawdown_threshold = float(params.get("max_drawdown", -0.20))
        actual_drawdown = result.metrics.get("max_drawdown")

        if actual_drawdown is not None and actual_drawdown < max_drawdown_threshold:
            alerts.append({
                "level": level,
                "rule": "max_drawdown_exceeded",
                "detail": (
                    f"Max drawdown {actual_drawdown:.4f} breaches "
                    f"threshold {max_drawdown_threshold:.4f}."
                ),
                "action": action,
            })

        max_loss_threshold = params.get("max_loss")
        if max_loss_threshold is not None:
            max_loss_threshold = float(max_loss_threshold)
            actual_return = result.metrics.get("total_return")

            if actual_return is not None and actual_return < max_loss_threshold:
                alerts.append({
                    "level": level,
                    "rule": "max_loss_exceeded",
                    "detail": (
                        f"Total return {actual_return:.4f} breaches "
                        f"max loss threshold {max_loss_threshold:.4f}."
                    ),
                    "action": action,
                })

        return alerts
