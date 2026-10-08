"""Newey-West / HAC statistics — the analysis-side door to them.

Two autocorrelation-robust statistics, so no downstream repo has to hand-roll a
Bartlett-kernel HAC sandwich estimator again:

* :func:`newey_west_tstat` — the HAC t-stat on the mean of a return series
  (is the average return significant once serial correlation is corrected
  for?). It lives in :mod:`quantbox.metrics` (TOM-1596) and is re-exported
  here as the SAME object, so the ``nw`` gate and the metrics module run one
  implementation.
* :func:`factor_regression` — Jensen's-alpha factor decomposition: OLS of a
  strategy return series on an intercept + known factor returns, with the
  HAC-robust standard error / t-stat on the intercept (is the return novel,
  or just paid-for factor exposure?).

Both fit through :func:`quantbox.metrics.hac_ols`, which delegates the HAC
covariance to ``statsmodels`` rather than reimplementing the
``(X'X)^-1 S (X'X)^-1`` sandwich by hand. This follows the
adapter-not-reimplementation principle: statsmodels is the maintained,
battle-tested reference for HAC inference; we do not compete with it.

The fit passes ``use_correction=False`` deliberately. statsmodels' default
applies a ``nobs/(nobs-k)`` small-sample correction to the HAC covariance; the
retired hand-rolled implementations these functions replace used the
*uncorrected* estimator. Disabling the correction makes this a provably
behaviour-preserving migration: the framework reproduces the retired gates'
numbers to machine precision (see ``tests/test_hac_parity.py``), so no
historical promotion decision silently flips. The correction is a finite-sample
refinement — with it enabled, t-stats shrink by exactly ``sqrt(nobs/(nobs-k))``
(~0.1% for a mean t-stat, ~0.4% for a 3-factor alpha), the
marginally-more-conservative direction for an acceptance gate. Adopting it is a
reasonable future policy change, but belongs in a deliberate decision, not a
silent side effect of this port.

All statistics are returned at full precision; rounding for display/JSON is the
caller's (CLI's) concern, not this module's.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from ..metrics import DEGENERATE_RTOL, hac_ols, newey_west_auto_lags, newey_west_tstat, require_finite

__all__ = ["factor_regression", "newey_west_auto_lags", "newey_west_tstat", "require_finite"]


def factor_regression(y, factors, factor_names: list[str], lags: int | None = None) -> dict:
    """OLS of ``y`` on an intercept + ``factors`` with Newey-West HAC SEs.

    Fits ``y = alpha + factors @ beta + e`` and returns the betas, Jensen's
    alpha (the intercept), and the HAC-robust SE / one-sided t-stat of the
    alpha under H0: alpha <= 0. statsmodels computes the HAC sandwich; this
    function only assembles the design matrix, picks the lag length, and shapes
    the result.

    ``y`` is the strategy return series (shape n); ``factors`` is the aligned
    factor panel (shape n x k); ``factor_names`` labels the k columns.

    A degenerate fit (too few observations, or a singular / perfectly collinear
    design) yields ``alpha_tstat = None`` rather than a spurious number.

    Requires at least one factor column — an intercept-only "regression" is not
    a factor decomposition and must not be constructed here; the caller is
    responsible for refusing an empty factor list before this is called.
    """
    if not factor_names:
        raise ValueError("factor_regression requires at least one factor column, got an empty factor_names list")

    y = np.asarray(y, dtype=float).ravel()
    F = np.asarray(factors, dtype=float)
    if F.ndim == 1:
        F = F.reshape(-1, 1)
    n, k_f = F.shape
    if k_f == 0:
        raise ValueError("factor panel has zero columns — cannot run a factor decomposition")
    if len(factor_names) != k_f:
        raise ValueError(
            f"factor_names has {len(factor_names)} entries but the factor panel has {k_f} "
            "columns — these must match 1:1."
        )
    if y.size != n:
        raise ValueError(f"y has {y.size} observations but the factor panel has {n} rows — these must match 1:1.")

    y_bad = int((~np.isfinite(y)).sum())
    if y_bad:
        raise ValueError(
            f"{y_bad} of {y.size} y observations are NaN/Inf — refusing to silently drop them "
            "(a corrupt file must not pass on the surviving subset)."
        )
    f_bad = int((~np.isfinite(F)).sum())
    if f_bad:
        raise ValueError(
            f"{f_bad} of {F.size} factor panel entries are NaN/Inf — refusing to silently drop "
            "them (a corrupt file must not pass on the surviving subset)."
        )

    k = k_f + 1  # +1 for the intercept

    base = {
        "n_obs": int(n),
        "n_factors": int(k_f),
        "factors": list(factor_names),
        "hac_lags": 0,
        "betas": None,
        "alpha": None,
        "alpha_se": None,
        "alpha_tstat": None,
        "alpha_pvalue_onesided": None,
        "r_squared": None,
    }
    # Need strictly more observations than parameters for any residual d.o.f.
    if n < k + 1:
        return base

    X = np.column_stack([np.ones(n), F])  # n x k, intercept first
    # Perfectly collinear factors — refuse to fabricate an alpha (mirrors the
    # retired estimator's np.linalg.inv LinAlgError branch).
    if np.linalg.matrix_rank(X) < k:
        return base

    if lags is None:
        lags = newey_west_auto_lags(n)
    lags = max(0, min(lags, n - 1))
    base["hac_lags"] = int(lags)

    try:
        fit = hac_ols(y, X, lags)
    except (np.linalg.LinAlgError, ValueError):
        return base

    # A perfect fit (residuals cancelled to noise — a constant y, or y built
    # exactly from the factors) has no HAC variance; statsmodels would return a
    # noise-sized SE and an alpha t-stat of ~1e16 (TOM-1351). Refuse it.
    if float(np.std(fit.resid)) <= DEGENERATE_RTOL * float(np.mean(np.abs(y))):
        return base

    var_diag = np.asarray(fit.bse, dtype=float) ** 2
    if not np.all(np.isfinite(var_diag)) or var_diag[0] <= 0:
        return base

    beta = np.asarray(fit.params, dtype=float)
    alpha = float(beta[0])
    alpha_se = float(fit.bse[0])
    alpha_t = alpha / alpha_se
    # One-sided (H1: alpha > 0). p = P(Z > t) = 1 - Phi(t).
    alpha_p = float(1.0 - norm.cdf(alpha_t))
    r_squared = float(fit.rsquared) if math.isfinite(fit.rsquared) else None

    betas = {name: float(b) for name, b in zip(factor_names, beta[1:], strict=True)}
    return {
        "n_obs": int(n),
        "n_factors": int(k_f),
        "factors": list(factor_names),
        "hac_lags": int(lags),
        "betas": betas,
        "alpha": alpha,
        "alpha_se": alpha_se,
        "alpha_tstat": float(alpha_t),
        "alpha_pvalue_onesided": alpha_p,
        "r_squared": r_squared,
    }
