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

**The scalar math is NOT reimplemented here.** ``quantbox.analysis.dsr`` owns
the Sharpe-estimator variance (``sr_estimator_std``) and the expected-max-Sharpe
deflation term (``expected_max_sr``) for the whole ecosystem, and this plugin
imports both. What belongs to this plugin is only the layer around them:
selecting sigma_SR (see below), annualising, and translating that module's
raised errors into the ``findings``/``passed`` dict a validation plugin must
return. Do not re-derive skew/kurtosis/expected-max-SR formulas in this file --
that drift is exactly what produced the defects this module was fixed for.

sigma_SR requires the individual Sharpe ratios of all N trials to be exact.
Pass them via ``params["trial_sharpes"]`` (recommended -- e.g. the Sharpe of
every variant in a parameter sweep) when available. Without them, this plugin
falls back to using the observed strategy's own Sharpe standard error as a
proxy for sigma_SR -- a common practical approximation, but strictly less
rigorous than supplying the actual trial distribution; the ``sigma_sr_source``
metric reports which mode was used. This sigma_SR choice is why the plugin
cannot simply call ``deflated_sharpe_ratio_from_returns``: that entry point
hardcodes the SE-proxy mode and has no way to accept a real trial distribution.

``trial_sharpes`` and the observed Sharpe are compared in the same units
(annualized, matching ``observed_sharpe``) -- annualizing the per-period
standard error consistently preserves the underlying PSR/DSR ratio.

**Fail-closed contract.** A validation gate that cannot COMPUTE a verdict must
never emit a passing one. Every input this plugin cannot honestly evaluate --
a series that is constant to within floating-point noise, NaN/Inf
 observations, a non-positive or
non-integral ``n_trials``, an undefined Sharpe-estimator variance -- returns
``passed=False`` with an ``error``-level finding and ``psr``/``dsr`` set to
``None``, never to a plausible-looking number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from quantbox.analysis.dsr import DEGENERATE_RTOL, expected_max_sr, sr_estimator_std
from quantbox.contracts import PluginMeta


class _UndefinedDSR(ValueError):
    """An input this plugin cannot compute a DSR from.

    Carries the ``rule`` name the failure should be reported under, so the
    fail-closed translation in ``validate`` stays a single code path.
    """

    def __init__(self, rule: str, detail: str) -> None:
        super().__init__(detail)
        self.rule = rule
        self.detail = detail


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


def _as_trial_count(n_trials: Any) -> int:
    """Read a config-supplied ``n_trials`` as a trial count WITHOUT rounding it.

    ``n_trials`` arrives straight from a config's YAML via ``runner.py``.
    Silently coercing a nonsensical value (0, -5, 2.7) into a usable one
    removes the entire multiple-testing penalty and yields a plausible number
    instead of an obvious error -- the 2026-07 gate bug that
    ``analysis.dsr.expected_max_sr`` was written to refuse. Non-positive values
    are rejected by ``expected_max_sr`` itself; this only rejects the shapes it
    cannot see.
    """
    if isinstance(n_trials, bool) or not isinstance(n_trials, (int, float, np.integer, np.floating)):
        raise _UndefinedDSR("invalid_n_trials", f"n_trials must be a positive integer, got {n_trials!r}")
    if not np.isfinite(float(n_trials)) or float(n_trials) != int(n_trials):
        raise _UndefinedDSR("invalid_n_trials", f"n_trials must be a whole number, got {n_trials!r}")
    return int(n_trials)


def _expected_max_sharpe(
    trial_sharpes: list[float] | None,
    n_trials: Any,
    se_sr_annual_fallback: float,
) -> tuple[float, float, str]:
    """Expected maximum Sharpe achievable by chance across N independent trials.

    Returns (sr0, sigma_sr, sigma_sr_source). sigma_sr_source is "trial_sharpes"
    when the caller supplied the actual per-trial Sharpe distribution, or
    "se_proxy_approximation" when falling back to the observed strategy's own
    Sharpe standard error.

    The expected-max-SR term comes from ``quantbox.analysis.dsr``, which RAISES
    on a non-positive trial count rather than coercing it to 1. Raises
    ``_UndefinedDSR`` when no deflation benchmark can be computed.
    """
    if trial_sharpes and len(trial_sharpes) >= 2:
        sigma_sr = float(np.std(np.asarray(trial_sharpes, dtype=float), ddof=1))
        n = len(trial_sharpes)
        source = "trial_sharpes"
    else:
        sigma_sr = float(se_sr_annual_fallback)
        n = _as_trial_count(n_trials)
        source = "se_proxy_approximation"

    if not np.isfinite(sigma_sr):
        raise _UndefinedDSR(
            "sigma_sr_undefined",
            f"sigma_SR is not finite ({sigma_sr!r}) -- the deflation benchmark cannot be computed.",
        )

    try:
        e_max = expected_max_sr(n)
    except ValueError as exc:  # non-positive trial count
        raise _UndefinedDSR("invalid_n_trials", str(exc)) from exc

    return float(sigma_sr * e_max), sigma_sr, source


@dataclass
class DeflatedSharpeBLPValidation:
    meta = PluginMeta(
        name="validation.deflated_sharpe_blp.v1",
        kind="validation",
        version="0.2.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Analytic Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): "
            "skewness/kurtosis-adjusted Probabilistic Sharpe Ratio evaluated against "
            "the expected maximum Sharpe of N independent trials. Outputs a "
            "probability (DSR) in [0, 1], not a Sharpe-valued number."
        ),
        tags=("validation", "statistics", "sharpe", "dsr", "psr", "multiple-testing"),
    )

    @staticmethod
    def _undefined(rule: str, detail: str, metrics: dict[str, Any]) -> dict[str, Any]:
        """Fail closed: no verdict was computable, so emit no number to mistake for one."""
        return {
            "findings": [{"level": "error", "rule": rule, "detail": detail}],
            "metrics": {**metrics, "psr": None, "dsr": None},
            "passed": False,
        }

    def validate(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        benchmark: pd.DataFrame | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        confidence: float = params.get("confidence", 0.95)
        trading_days: int = params.get("trading_days", 365)
        n_trials: Any = params.get("n_trials", 1)
        trial_sharpes: list[float] | None = params.get("trial_sharpes")

        rets_col = "returns" if "returns" in returns.columns else returns.select_dtypes("number").columns[0]
        rets = returns[rets_col].to_numpy(dtype=float)
        t = len(rets)

        if t < 3:
            return self._undefined(
                "insufficient_observations",
                f"Need at least 3 return observations for DSR, got {t}.",
                {"n_observations": t},
            )

        n_nonfinite = int((~np.isfinite(rets)).sum())
        if n_nonfinite:
            return self._undefined(
                "non_finite_returns",
                f"{n_nonfinite} of {t} return observations are NaN/Inf -- refusing to emit a DSR "
                "verdict for a corrupted series.",
                {"n_observations": t, "n_nonfinite": n_nonfinite},
            )

        std_period = float(np.std(rets, ddof=1))
        scale = float(np.mean(np.abs(rets)))
        if std_period <= DEGENERATE_RTOL * scale:
            return self._undefined(
                "degenerate_returns",
                f"degenerate returns: standard deviation ({std_period!r}) is negligible against the "
                f"scale of the series itself (mean |return| = {scale!r}) -- the series is constant to "
                "within floating-point noise, so it carries no signal and no Sharpe ratio exists. "
                "A constant series must never be reported as passing.",
                {"n_observations": t, "std": std_period, "scale": scale},
            )

        sr_hat_period = float(np.mean(rets) / std_period)
        observed_sharpe = sr_hat_period * float(np.sqrt(trading_days))
        skew, kurt = _skew_kurtosis(rets)

        partial_metrics: dict[str, Any] = {
            "observed_sharpe": observed_sharpe,
            "n_observations": t,
            "skewness": skew,
            "kurtosis": kurt,
        }

        try:
            se_annual = sr_estimator_std(t, sr_hat_period, skew, kurt) * float(np.sqrt(trading_days))
            if not np.isfinite(se_annual) or se_annual <= 0:
                raise _UndefinedDSR(
                    "dsr_undefined",
                    "Standard error of the Sharpe estimator is zero or undefined; DSR is undefined.",
                )
            partial_metrics["sharpe_standard_error"] = se_annual
            sr0_annual, sigma_sr, sigma_sr_source = _expected_max_sharpe(trial_sharpes, n_trials, se_annual)
        except _UndefinedDSR as exc:
            return self._undefined(exc.rule, exc.detail, partial_metrics)
        except ValueError as exc:  # negative OR degenerate SR-estimator variance (see sr_estimator_std)
            return self._undefined("sr_variance_undefined", str(exc), partial_metrics)

        n_trials_used = len(trial_sharpes) if sigma_sr_source == "trial_sharpes" else _as_trial_count(n_trials)

        psr = float(norm.cdf(observed_sharpe / se_annual))
        dsr = float(norm.cdf((observed_sharpe - sr0_annual) / se_annual))

        findings: list[dict[str, Any]] = []

        if sigma_sr_source == "se_proxy_approximation":
            findings.append(
                {
                    "level": "info",
                    "rule": "sigma_sr_approximated",
                    "detail": (
                        "No trial_sharpes supplied -- sigma_SR (spread of Sharpe ratios across the "
                        f"{n_trials_used} trials) was approximated using the observed strategy's own Sharpe "
                        "standard error. Pass params['trial_sharpes'] with the actual per-variant Sharpe "
                        "ratios for a rigorous DSR."
                    ),
                }
            )

        if dsr < confidence:
            findings.append(
                {
                    "level": "warn",
                    "rule": "dsr_below_confidence",
                    "detail": (
                        f"DSR ({dsr:.4f}) is below the {confidence:.0%} confidence threshold -- "
                        f"observed Sharpe ({observed_sharpe:.4f}) is not statistically distinguishable "
                        f"from the expected best-of-{n_trials_used} "
                        f"chance result ({sr0_annual:.4f})."
                    ),
                }
            )

        return {
            "findings": findings,
            "metrics": {
                "observed_sharpe": observed_sharpe,
                "n_observations": t,
                "skewness": skew,
                "kurtosis": kurt,
                "sharpe_standard_error": se_annual,
                "n_trials": n_trials_used,
                "sigma_sr": sigma_sr,
                "sigma_sr_source": sigma_sr_source,
                "expected_max_sharpe_null": sr0_annual,
                "psr": psr,
                "dsr": dsr,
            },
            "passed": bool(dsr >= confidence),
        }
