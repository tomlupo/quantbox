"""Turnover validation plugin.

Measures portfolio turnover from weight changes, computes cost-adjusted returns
and Sharpe, and estimates breakeven transaction cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta


def _annualized_sharpe(returns: np.ndarray, trading_days: int) -> float:
    """Compute annualized Sharpe ratio from an array of returns."""
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std == 0:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(trading_days))


def _cost_adjusted_sharpe(
    raw_returns: np.ndarray,
    daily_turnover: np.ndarray,
    cost_bps: float,
    trading_days: int,
) -> float:
    """Compute Sharpe of returns after subtracting turnover-based costs."""
    cost_per_day = daily_turnover * cost_bps / 10_000
    adjusted = raw_returns - cost_per_day
    return _annualized_sharpe(adjusted, trading_days)


@dataclass
class TurnoverValidation:
    meta = PluginMeta(
        name="validation.turnover.v1",
        kind="validation",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Turnover validation: computes daily/annual turnover from weight changes, "
            "cost-adjusted returns and Sharpe, and breakeven transaction cost."
        ),
        tags=("validation", "turnover", "costs"),
    )

    def validate(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        benchmark: pd.DataFrame | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        cost_bps: float = params.get("cost_bps", 10)
        trading_days: int = params.get("trading_days", 365)

        rets = returns.iloc[:, 0].values

        # Compute daily turnover: sum(abs(weight_diff)) / 2 per day, averaged
        weight_diffs = weights.diff().iloc[1:]
        daily_to = weight_diffs.abs().sum(axis=1) / 2
        daily_to_values = daily_to.values

        # Align lengths: daily_turnover has one fewer row than weights
        # Align with returns (skip first return to match diff)
        min_len = min(len(rets) - 1, len(daily_to_values))
        aligned_rets = rets[1 : 1 + min_len]
        aligned_to = daily_to_values[:min_len]

        avg_daily_turnover = float(np.mean(daily_to_values)) if len(daily_to_values) > 0 else 0.0
        annual_turnover = avg_daily_turnover * trading_days

        raw_sharpe = _annualized_sharpe(rets, trading_days)

        ca_sharpe = _cost_adjusted_sharpe(aligned_rets, aligned_to, cost_bps, trading_days)

        # Breakeven cost: binary search for max cost_bps where adjusted Sharpe > 0
        breakeven = _find_breakeven_cost(aligned_rets, aligned_to, trading_days)

        findings: list[dict[str, Any]] = []

        if annual_turnover > 50:
            findings.append({
                "level": "warn",
                "rule": "high_annual_turnover",
                "detail": f"Annual turnover is {annual_turnover:.1f}x, which may erode returns.",
            })

        if ca_sharpe < 0 < raw_sharpe:
            findings.append({
                "level": "warn",
                "rule": "costs_eliminate_edge",
                "detail": (
                    f"Cost-adjusted Sharpe ({ca_sharpe:.4f}) is negative while "
                    f"raw Sharpe ({raw_sharpe:.4f}) is positive at {cost_bps} bps."
                ),
            })

        passed = len(findings) == 0

        return {
            "findings": findings,
            "metrics": {
                "daily_turnover": avg_daily_turnover,
                "annual_turnover": annual_turnover,
                "raw_sharpe": raw_sharpe,
                "cost_adjusted_sharpe": ca_sharpe,
                "breakeven_cost_bps": breakeven,
                "cost_bps_used": cost_bps,
            },
            "passed": passed,
        }


def _find_breakeven_cost(
    returns: np.ndarray,
    daily_turnover: np.ndarray,
    trading_days: int,
) -> float:
    """Binary search for the maximum cost (in bps) where adjusted Sharpe > 0."""
    if len(returns) < 2 or len(daily_turnover) < 2:
        return 0.0

    # If raw Sharpe is already <= 0, breakeven is 0
    if _annualized_sharpe(returns, trading_days) <= 0:
        return 0.0

    # If there is no turnover, breakeven is effectively infinite
    if np.mean(daily_turnover) < 1e-12:
        return float("inf")

    lo, hi = 0.0, 10000.0
    for _ in range(50):
        mid = (lo + hi) / 2
        s = _cost_adjusted_sharpe(returns, daily_turnover, mid, trading_days)
        if s > 0:
            lo = mid
        else:
            hi = mid

    return round(lo, 2)
