"""Walk-forward validation plugin.

Splits a return series into sequential folds and compares in-sample vs
out-of-sample Sharpe ratios to detect overfitting.
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
class WalkForwardValidation:
    meta = PluginMeta(
        name="validation.walk_forward.v1",
        kind="validation",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Walk-forward validation: splits returns into sequential folds, "
            "computes in-sample vs out-of-sample Sharpe, and flags overfitting."
        ),
        tags=("validation", "overfitting", "walk-forward"),
    )

    def validate(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        benchmark: pd.DataFrame | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        n_splits: int = params.get("n_splits", 5)
        train_ratio: float = params.get("train_ratio", 0.7)
        trading_days: int = params.get("trading_days", 365)

        rets = returns.iloc[:, 0].values
        n = len(rets)
        fold_size = n // n_splits

        is_sharpes: list[float] = []
        oos_sharpes: list[float] = []

        for i in range(n_splits):
            start = i * fold_size
            end = start + fold_size if i < n_splits - 1 else n
            fold = rets[start:end]

            split_idx = int(len(fold) * train_ratio)
            is_portion = fold[:split_idx]
            oos_portion = fold[split_idx:]

            is_sharpes.append(_annualized_sharpe(is_portion, trading_days))
            oos_sharpes.append(_annualized_sharpe(oos_portion, trading_days))

        is_mean = float(np.mean(is_sharpes))
        oos_mean = float(np.mean(oos_sharpes))

        if abs(is_mean) > 1e-10:
            degradation = (oos_mean - is_mean) / abs(is_mean)
        else:
            degradation = 0.0

        findings: list[dict[str, Any]] = []

        if oos_mean < 0:
            findings.append({
                "level": "warn",
                "rule": "negative_oos_sharpe",
                "detail": f"Mean OOS Sharpe is negative ({oos_mean:.4f}).",
            })

        if degradation < -0.5:
            findings.append({
                "level": "warn",
                "rule": "sharpe_degradation_excessive",
                "detail": (
                    f"Sharpe degradation of {degradation:.4f} exceeds "
                    f"-0.5 threshold (IS={is_mean:.4f}, OOS={oos_mean:.4f})."
                ),
            })

        passed = len(findings) == 0

        return {
            "findings": findings,
            "metrics": {
                "is_sharpe_mean": is_mean,
                "oos_sharpe_mean": oos_mean,
                "sharpe_degradation": degradation,
                "n_splits": n_splits,
                "is_sharpes": is_sharpes,
                "oos_sharpes": oos_sharpes,
            },
            "passed": passed,
        }
