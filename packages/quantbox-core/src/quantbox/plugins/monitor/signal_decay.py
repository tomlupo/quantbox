"""Signal decay monitor plugin.

Detects deteriorating strategy performance by tracking Sharpe ratio
trends across recent run history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quantbox.contracts import PluginMeta, RunResult


@dataclass
class SignalDecayMonitor:
    """Monitor that alerts when recent Sharpe ratios fall below a threshold.

    Extracts ``metrics["sharpe"]`` from the last ``window`` entries in
    history, computes the mean, and emits an alert if the mean Sharpe
    falls below ``min_sharpe``.

    Params:
        min_sharpe: Minimum acceptable mean Sharpe (default 0.5).
        window: Number of recent history entries to examine (default 5).
    """

    meta = PluginMeta(
        name="monitor.signal_decay.v1",
        kind="monitor",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Monitors rolling Sharpe ratio across recent runs. "
            "Emits a signal_decay alert when mean Sharpe drops below threshold."
        ),
        tags=("monitor", "signal", "sharpe"),
        capabilities=("backtest", "paper", "live"),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "min_sharpe": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Minimum acceptable mean Sharpe ratio.",
                },
                "window": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 5,
                    "description": "Number of recent runs to average over.",
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
        """Check for signal decay by examining recent Sharpe ratios."""
        if not history:
            return []

        min_sharpe = float(params.get("min_sharpe", 0.5))
        window = int(params.get("window", 5))

        recent = history[-window:]
        sharpe_values = [
            r.metrics["sharpe"]
            for r in recent
            if "sharpe" in r.metrics
        ]

        if len(sharpe_values) < window:
            return []

        mean_sharpe = sum(sharpe_values) / len(sharpe_values)

        if mean_sharpe < min_sharpe:
            return [{
                "level": "warn",
                "rule": "signal_decay",
                "detail": (
                    f"Mean Sharpe {mean_sharpe:.4f} over last {window} runs "
                    f"is below threshold {min_sharpe:.4f}."
                ),
                "action": "warn",
            }]

        return []
