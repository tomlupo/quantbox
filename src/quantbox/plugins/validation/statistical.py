"""Bootstrap Sharpe significance validation plugin.

Tests whether an observed Sharpe ratio is distinguishable from a zero-mean null
via Monte-Carlo simulation, reports a bootstrap confidence interval, and applies
a linear multiple-testing haircut.

This is **not** the Bailey & Lopez de Prado (2014) analytic Deflated Sharpe
Ratio (DSR) — it does not use the skewness/kurtosis-adjusted probabilistic
Sharpe ratio formula, and its multiple-testing adjustment is a flat 2%-per-
trial haircut rather than the expected-maximum-Sharpe-under-N-trials
correction. For the literal BLP DSR, see `validation.deflated_sharpe_blp.v1`
(`quantbox.plugins.validation.deflated_sharpe_blp.DeflatedSharpeBLPValidation`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta
from quantbox.metrics import sharpe_ratio


@dataclass
class StatisticalValidation:
    meta = PluginMeta(
        name="validation.statistical.v1",
        kind="validation",
        version="0.2.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Bootstrap Sharpe significance test: Monte-Carlo null-distribution test for "
            "Sharpe > 0, bootstrap confidence intervals, and a flat multiple-testing "
            "haircut. NOT the Bailey-Lopez de Prado analytic Deflated Sharpe Ratio -- "
            "see validation.deflated_sharpe_blp.v1 for that."
        ),
        tags=("validation", "statistics", "sharpe", "bootstrap"),
        params_schema={
            "type": "object",
            "properties": {
                "n_trials": {"type": "integer", "default": 100, "description": "Random-strategy trials of the null."},
                "n_bootstrap": {"type": "integer", "default": 1000, "description": "Bootstrap resamples."},
                "confidence": {"type": "number", "default": 0.95, "description": "Confidence level of the intervals."},
                "n_strategies_tested": {
                    "type": "integer",
                    "default": 1,
                    "description": "Strategies tested, for the multiple-testing correction.",
                },
                "trading_days": {"type": "integer", "default": 365, "description": "Bars per year for annualisation."},
            },
        },
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

        rets_col = "returns" if "returns" in returns.columns else returns.select_dtypes("number").columns[0]
        rets = returns[rets_col].values
        n = len(rets)

        observed_sharpe = sharpe_ratio(rets, trading_days)

        # Bootstrap null-distribution test: simulate n_trials zero-mean series at the
        # observed series' own volatility, and see how often their Sharpe reaches the
        # observed Sharpe by chance alone.
        rng = np.random.default_rng(42)
        null_sharpes = np.array(
            [sharpe_ratio(rng.normal(0, np.std(rets, ddof=1), size=n), trading_days) for _ in range(n_trials)]
        )
        pct_exceeding = float(np.mean(null_sharpes >= observed_sharpe))
        # Sharpe adjusted for null-test significance: observed if it clears the
        # confidence threshold, otherwise scaled down by how deep into the null
        # distribution it falls. This is a heuristic scaling, not an analytic deflation.
        if pct_exceeding <= (1 - confidence):
            bootstrap_adjusted_sharpe = observed_sharpe
        else:
            bootstrap_adjusted_sharpe = observed_sharpe * (1 - pct_exceeding)

        # Bootstrap CI
        bootstrap_sharpes = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(rets, size=n, replace=True)
            bootstrap_sharpes[i] = sharpe_ratio(sample, trading_days)

        alpha = 1 - confidence
        ci_lower = float(np.percentile(bootstrap_sharpes, alpha / 2 * 100))
        ci_upper = float(np.percentile(bootstrap_sharpes, (1 - alpha / 2) * 100))

        # Flat multiple-testing haircut: 2% of observed Sharpe per strategy tested.
        # This is a crude linear heuristic, not the BLP expected-max-Sharpe-under-N
        # correction (see validation.deflated_sharpe_blp.v1 for that).
        penalty = n_strategies_tested * 0.02
        haircut_sharpe = observed_sharpe * max(0.0, 1.0 - penalty)

        findings: list[dict[str, Any]] = []

        if bootstrap_adjusted_sharpe <= 0:
            findings.append(
                {
                    "level": "warn",
                    "rule": "bootstrap_adjusted_sharpe_not_significant",
                    "detail": (
                        f"Bootstrap-adjusted Sharpe ({bootstrap_adjusted_sharpe:.4f}) is not positive, "
                        f"suggesting observed Sharpe ({observed_sharpe:.4f}) may be a false positive."
                    ),
                }
            )

        if ci_lower < 0:
            findings.append(
                {
                    "level": "warn",
                    "rule": "sharpe_ci_includes_zero",
                    "detail": (f"Bootstrap {confidence:.0%} CI [{ci_lower:.4f}, {ci_upper:.4f}] includes zero."),
                }
            )

        passed = len(findings) == 0

        return {
            "findings": findings,
            "metrics": {
                "observed_sharpe": observed_sharpe,
                "bootstrap_adjusted_sharpe": bootstrap_adjusted_sharpe,
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
