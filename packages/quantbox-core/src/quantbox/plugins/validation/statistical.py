"""Statistical validation plugin.

Applies deflated Sharpe ratio analysis, bootstrap confidence intervals, and
multiple-testing haircut to assess whether observed performance is statistically
significant or likely a false positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

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
class StatisticalValidation:
    meta = PluginMeta(
        name="validation.statistical.v1",
        kind="validation",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Statistical validation: deflated Sharpe ratio, bootstrap confidence "
            "intervals, and multiple-testing haircut for observed Sharpe."
        ),
        tags=("validation", "statistics", "sharpe"),
    )

    def validate(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        benchmark: pd.DataFrame | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        n_trials: int = params.get("n_trials", 100)
        n_bootstrap: int = params.get("n_bootstrap", 1000)
        confidence: float = params.get("confidence", 0.95)
        n_strategies_tested: int = params.get("n_strategies_tested", 1)
        trading_days: int = params.get("trading_days", 365)

        rets = returns.iloc[:, 0].values
        n = len(rets)

        observed_sharpe = _annualized_sharpe(rets, trading_days)

        # Deflated Sharpe: simulate null distribution of Sharpe ratios
        rng = np.random.default_rng(42)
        null_sharpes = np.array([
            _annualized_sharpe(rng.normal(0, np.std(rets, ddof=1), size=n), trading_days)
            for _ in range(n_trials)
        ])
        pct_exceeding = float(np.mean(null_sharpes >= observed_sharpe))
        # Deflated Sharpe: observed if it passes the significance test,
        # otherwise scaled by how far it is into the null distribution
        if pct_exceeding <= (1 - confidence):
            deflated_sharpe = observed_sharpe
        else:
            deflated_sharpe = observed_sharpe * (1 - pct_exceeding)

        # Bootstrap CI
        bootstrap_sharpes = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(rets, size=n, replace=True)
            bootstrap_sharpes[i] = _annualized_sharpe(sample, trading_days)

        alpha = 1 - confidence
        ci_lower = float(np.percentile(bootstrap_sharpes, alpha / 2 * 100))
        ci_upper = float(np.percentile(bootstrap_sharpes, (1 - alpha / 2) * 100))

        # Haircut Sharpe for multiple testing
        penalty = n_strategies_tested * 0.02
        haircut_sharpe = observed_sharpe * max(0.0, 1.0 - penalty)

        findings: list[dict[str, Any]] = []

        if deflated_sharpe <= 0:
            findings.append({
                "level": "warn",
                "rule": "deflated_sharpe_not_significant",
                "detail": (
                    f"Deflated Sharpe ({deflated_sharpe:.4f}) is not positive, "
                    f"suggesting observed Sharpe ({observed_sharpe:.4f}) may be a false positive."
                ),
            })

        if ci_lower < 0:
            findings.append({
                "level": "warn",
                "rule": "sharpe_ci_includes_zero",
                "detail": (
                    f"Bootstrap {confidence:.0%} CI [{ci_lower:.4f}, {ci_upper:.4f}] "
                    f"includes zero."
                ),
            })

        passed = len(findings) == 0

        return {
            "findings": findings,
            "metrics": {
                "observed_sharpe": observed_sharpe,
                "deflated_sharpe": deflated_sharpe,
                "sharpe_ci_lower": ci_lower,
                "sharpe_ci_upper": ci_upper,
                "haircut_sharpe": haircut_sharpe,
                "n_trials": n_trials,
                "n_bootstrap": n_bootstrap,
                "n_strategies_tested": n_strategies_tested,
                "pct_null_exceeding": pct_exceeding,
            },
            "passed": passed,
        }
