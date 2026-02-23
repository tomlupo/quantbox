"""Benchmark validation plugin.

Computes standard benchmark comparison metrics: beta, alpha, tracking error,
information ratio, and R-squared. Gracefully skips when no benchmark is provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta


def _extract_series(df: pd.DataFrame) -> pd.Series:
    """Extract a returns Series from a DataFrame (first column or 'returns' column)."""
    if "returns" in df.columns:
        return df["returns"]
    return df.iloc[:, 0]


@dataclass
class BenchmarkValidation:
    meta = PluginMeta(
        name="validation.benchmark.v1",
        kind="validation",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Benchmark validation: computes beta, alpha, tracking error, "
            "information ratio, and R-squared against a benchmark."
        ),
        tags=("validation", "benchmark", "relative-performance"),
    )

    def validate(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        benchmark: pd.DataFrame | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if benchmark is None:
            return {"findings": [], "metrics": {}, "passed": True}

        trading_days: int = params.get("trading_days", 365)

        strategy = _extract_series(returns).values
        bench = _extract_series(benchmark).values

        # Align lengths
        n = min(len(strategy), len(bench))
        strategy = strategy[:n]
        bench = bench[:n]

        bench_var = float(np.var(bench, ddof=0))
        if bench_var == 0:
            beta = 0.0
        else:
            beta = float(np.cov(strategy, bench, ddof=0)[0, 1] / bench_var)

        alpha = float((np.mean(strategy) - beta * np.mean(bench)) * trading_days)

        excess = strategy - bench
        tracking_error = float(np.std(excess, ddof=1) * np.sqrt(trading_days))

        if tracking_error > 0:
            information_ratio = float(np.mean(excess) * trading_days / tracking_error)
        else:
            information_ratio = 0.0

        corr_matrix = np.corrcoef(strategy, bench)
        r_squared = float(corr_matrix[0, 1] ** 2)

        findings: list[dict[str, Any]] = []

        if abs(beta) > 1.5:
            findings.append({
                "level": "warn",
                "rule": "high_beta",
                "detail": f"Strategy beta ({beta:.4f}) is unusually high.",
            })

        if r_squared < 0.1:
            findings.append({
                "level": "warn",
                "rule": "low_r_squared",
                "detail": (
                    f"R-squared ({r_squared:.4f}) is very low, suggesting the "
                    f"benchmark may not be appropriate."
                ),
            })

        passed = len(findings) == 0

        return {
            "findings": findings,
            "metrics": {
                "beta": beta,
                "alpha": alpha,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "r_squared": r_squared,
            },
            "passed": passed,
        }
