"""Regime validation plugin.

Classifies each day into a market regime (trending up, trending down, high vol,
low vol) based on rolling statistics and reports per-regime performance metrics.
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


@dataclass
class RegimeValidation:
    meta = PluginMeta(
        name="validation.regime.v1",
        kind="validation",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Regime validation: classifies days into trending-up, trending-down, "
            "high-vol, and low-vol regimes, then reports per-regime Sharpe, return, "
            "and time allocation."
        ),
        tags=("validation", "regime", "market-conditions"),
    )

    def validate(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        benchmark: pd.DataFrame | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        window: int = params.get("window", 60)
        trading_days: int = params.get("trading_days", 365)

        rets = returns.iloc[:, 0]

        rolling_ret = rets.rolling(window).sum()
        rolling_vol = rets.rolling(window).std()

        # Need enough data for rolling stats
        valid_mask = rolling_ret.notna() & rolling_vol.notna()
        valid_rets = rets[valid_mask]
        valid_rolling_ret = rolling_ret[valid_mask]
        valid_rolling_vol = rolling_vol[valid_mask]

        if len(valid_rets) == 0:
            return {
                "findings": [],
                "metrics": {"regime_breakdown": []},
                "passed": True,
            }

        ret_std = valid_rolling_ret.std()
        vol_median = valid_rolling_vol.median()

        # Classify regimes
        labels = pd.Series("low_vol", index=valid_rets.index)

        trending_up = valid_rolling_ret > ret_std
        trending_down = valid_rolling_ret < -ret_std
        labels[trending_up] = "trending_up"
        labels[trending_down] = "trending_down"

        not_trending = ~trending_up & ~trending_down
        high_vol = not_trending & (valid_rolling_vol > vol_median)
        labels[high_vol] = "high_vol"

        # Compute per-regime metrics
        regime_breakdown: list[dict[str, Any]] = []
        total_days = len(valid_rets)

        for regime_name in ["trending_up", "trending_down", "high_vol", "low_vol"]:
            mask = labels == regime_name
            regime_rets = valid_rets[mask].values

            if len(regime_rets) == 0:
                continue

            sharpe = _annualized_sharpe(regime_rets, trading_days)
            total_ret = float(np.sum(regime_rets))
            pct_time = len(regime_rets) / total_days

            regime_breakdown.append({
                "regime": regime_name,
                "sharpe": sharpe,
                "return": total_ret,
                "pct_time": pct_time,
            })

        findings: list[dict[str, Any]] = []

        # Flag regimes with notably poor performance
        for entry in regime_breakdown:
            if entry["sharpe"] < -1.0 and entry["pct_time"] > 0.1:
                findings.append({
                    "level": "warn",
                    "rule": "poor_regime_performance",
                    "detail": (
                        f"Regime '{entry['regime']}' has Sharpe {entry['sharpe']:.2f} "
                        f"over {entry['pct_time']:.0%} of the period."
                    ),
                })

        passed = len(findings) == 0

        return {
            "findings": findings,
            "metrics": {
                "regime_breakdown": regime_breakdown,
                "window": window,
                "n_classified_days": total_days,
            },
            "passed": passed,
        }
