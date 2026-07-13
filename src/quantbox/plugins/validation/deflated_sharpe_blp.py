"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) validation plugin.

Implements the analytic Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe
Ratio (DSR) from Bailey, D. H. and Lopez de Prado, M. (2014), "The Deflated
Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and
Non-Normality", Journal of Portfolio Management.

DSR is a *probability* in [0, 1]: the confidence that the true Sharpe ratio
exceeds the Sharpe expected by chance alone from the best of N independent
trials, after adjusting the Sharpe estimator's standard error for the return
series' own skewness and kurtosis (non-normal returns inflate/deflate that
standard error relative to the Gaussian case).

    PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(T-1) / sqrt(1 - skew*SR_hat + ((kurt-1)/4)*SR_hat^2) )
    SR0      = sigma_SR * [ (1-gamma)*Phi^-1(1 - 1/N) + gamma*Phi^-1(1 - 1/(N*e)) ]
    DSR      = PSR(SR0)

where SR_hat/skew/kurt are computed on *per-period* returns, T is the number
of observations, gamma is the Euler-Mascheroni constant, and sigma_SR is the
standard deviation of the Sharpe ratio across the N trials actually attempted.

sigma_SR requires the individual Sharpe ratios of all N trials to be exact.
Pass them via `params["trial_sharpes"]` (recommended -- e.g. every variant's
Sharpe from a parameter sweep) when available. Without them, this plugin falls
back to using the observed strategy's own Sharpe standard error as a proxy for
sigma_SR -- a common practical approximation, but strictly less rigorous than
supplying the actual trial distribution; the `sigma_sr_source` metric reports
which mode was used.

`trial_sharpes` and the observed Sharpe are compared in the same units
(annualized, matching `observed_sharpe`) -- annualizing the per-period
standard error consistently preserves the underlying PSR/DSR ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from quantbox.contracts import PluginMeta

_EULER_MASCHERONI = 0.5772156649015329


def _skew_kurtosis(x: np.ndarray) -> tuple[float, float]:
    """Sample skewness and (non-excess, Gaussian=3) kurtosis."""
    n = len(x)
    if n < 3:
        return 0.0, 3.0
    mean = x.mean()
    std = x.std(ddof=0)
    if std == 0:
        return 0.0, 3.0
    z = (x - mean) / std
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    return skew, kurt


def _annualized_sharpe(returns: np.ndarray, trading_days: int) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std == 0:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(trading_days))


def _se_sharpe_annual(returns: np.ndarray, sr_hat_daily: float, trading_days: int) -> tuple[float, float, float]:
    """Standard error of the annualized Sharpe estimator (Bailey & Lopez de Prado 2012).

    Computed on per-period returns/Sharpe, then annualized by sqrt(trading_days) --
    this is equivalent to deriving the SE directly in annualized units, since the
    Gaussian-vs-fat-tail correction term depends only on skew/kurtosis (scale-free)
    and the per-period SR_hat.
    """
    t = len(returns)
    skew, kurt = _skew_kurtosis(returns)
    if t < 2:
        return float("nan"), skew, kurt
    variance_term = max(1.0 - skew * sr_hat_daily + ((kurt - 1.0) / 4.0) * sr_hat_daily**2, 1e-12)
    se_daily = float(np.sqrt(variance_term / (t - 1)))
    se_annual = se_daily * float(np.sqrt(trading_days))
    return se_annual, skew, kurt


def _expected_max_sharpe(
    trial_sharpes: list[float] | None,
    n_trials: int,
    se_sr_annual_fallback: float,
) -> tuple[float, float, str]:
    """Expected maximum Sharpe achievable by chance across N independent trials.

    Returns (sr0, sigma_sr, sigma_sr_source). sigma_sr_source is "trial_sharpes"
    when the caller supplied the actual per-trial Sharpe distribution, or
    "se_proxy_approximation" when falling back to the observed strategy's own
    Sharpe standard error.
    """
    if trial_sharpes and len(trial_sharpes) >= 2:
        sigma_sr = float(np.std(np.asarray(trial_sharpes, dtype=float), ddof=1))
        n = len(trial_sharpes)
        source = "trial_sharpes"
    else:
        sigma_sr = se_sr_annual_fallback
        n = max(int(n_trials), 1)
        source = "se_proxy_approximation"

    if n <= 1 or not np.isfinite(sigma_sr):
        return 0.0, sigma_sr, source

    sr0 = sigma_sr * (
        (1 - _EULER_MASCHERONI) * norm.ppf(1 - 1.0 / n) + _EULER_MASCHERONI * norm.ppf(1 - 1.0 / (n * np.e))
    )
    return float(sr0), sigma_sr, source


@dataclass
class DeflatedSharpeBLPValidation:
    meta = PluginMeta(
        name="validation.deflated_sharpe_blp.v1",
        kind="validation",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Analytic Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): "
            "skewness/kurtosis-adjusted Probabilistic Sharpe Ratio evaluated against "
            "the expected maximum Sharpe of N independent trials. Outputs a "
            "probability (DSR) in [0, 1], not a Sharpe-valued number."
        ),
        tags=("validation", "statistics", "sharpe", "dsr", "psr", "multiple-testing"),
    )

    def validate(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        benchmark: pd.DataFrame | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        confidence: float = params.get("confidence", 0.95)
        trading_days: int = params.get("trading_days", 365)
        n_trials: int = params.get("n_trials", 1)
        trial_sharpes: list[float] | None = params.get("trial_sharpes")

        rets_col = "returns" if "returns" in returns.columns else returns.select_dtypes("number").columns[0]
        rets = returns[rets_col].to_numpy(dtype=float)
        t = len(rets)

        if t < 3:
            return {
                "findings": [
                    {
                        "level": "error",
                        "rule": "insufficient_observations",
                        "detail": f"Need at least 3 return observations for DSR, got {t}.",
                    }
                ],
                "metrics": {"n_observations": t},
                "passed": False,
            }

        sr_hat_daily = float(np.mean(rets) / np.std(rets, ddof=1)) if np.std(rets, ddof=1) != 0 else 0.0
        observed_sharpe = _annualized_sharpe(rets, trading_days)

        se_annual, skew, kurt = _se_sharpe_annual(rets, sr_hat_daily, trading_days)
        sr0_annual, sigma_sr, sigma_sr_source = _expected_max_sharpe(trial_sharpes, n_trials, se_annual)

        if not np.isfinite(se_annual) or se_annual <= 0:
            psr = float("nan")
            dsr = float("nan")
        else:
            psr = float(norm.cdf((observed_sharpe - 0.0) / se_annual))
            dsr = float(norm.cdf((observed_sharpe - sr0_annual) / se_annual))

        findings: list[dict[str, Any]] = []

        if sigma_sr_source == "se_proxy_approximation" and (trial_sharpes is None or len(trial_sharpes) < 2):
            findings.append(
                {
                    "level": "info",
                    "rule": "sigma_sr_approximated",
                    "detail": (
                        "No trial_sharpes supplied -- sigma_SR (spread of Sharpe ratios across the "
                        f"{n_trials} trials) was approximated using the observed strategy's own Sharpe "
                        "standard error. Pass params['trial_sharpes'] with the actual per-variant Sharpe "
                        "ratios for a rigorous DSR."
                    ),
                }
            )

        if np.isnan(dsr):
            findings.append(
                {
                    "level": "error",
                    "rule": "dsr_undefined",
                    "detail": "Standard error of the Sharpe estimator is zero or undefined; DSR is undefined.",
                }
            )
        elif dsr < confidence:
            findings.append(
                {
                    "level": "warn",
                    "rule": "dsr_below_confidence",
                    "detail": (
                        f"DSR ({dsr:.4f}) is below the {confidence:.0%} confidence threshold -- "
                        f"observed Sharpe ({observed_sharpe:.4f}) is not statistically distinguishable "
                        f"from the expected best-of-{max(len(trial_sharpes) if trial_sharpes else n_trials, 1)} "
                        f"chance result ({sr0_annual:.4f})."
                    ),
                }
            )

        passed = bool(np.isfinite(dsr) and dsr >= confidence)

        return {
            "findings": findings,
            "metrics": {
                "observed_sharpe": observed_sharpe,
                "n_observations": t,
                "skewness": skew,
                "kurtosis": kurt,
                "sharpe_standard_error": se_annual,
                "n_trials": len(trial_sharpes) if trial_sharpes else n_trials,
                "sigma_sr": sigma_sr,
                "sigma_sr_source": sigma_sr_source,
                "expected_max_sharpe_null": sr0_annual,
                "psr": psr,
                "dsr": dsr,
            },
            "passed": passed,
        }
