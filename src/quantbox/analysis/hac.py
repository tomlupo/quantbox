"""Newey-West / HAC statistics — thin wrappers over statsmodels.

This module owns two autocorrelation-robust statistics for the whole
ecosystem, so no downstream repo has to hand-roll a Bartlett-kernel HAC
sandwich estimator again:

* :func:`newey_west_tstat` — the HAC t-stat on the mean of a return series
  (is the average return significant once serial correlation is corrected
  for?). This is an OLS of the series on a constant.
* :func:`factor_regression` — Jensen's-alpha factor decomposition: OLS of a
  strategy return series on an intercept + known factor returns, with the
  HAC-robust standard error / t-stat on the intercept (is the return novel,
  or just paid-for factor exposure?).

Both delegate the HAC covariance to ``statsmodels`` —
``OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": k})`` — rather than
reimplementing the ``(X'X)^-1 S (X'X)^-1`` sandwich by hand. This follows the
adapter-not-reimplementation principle: statsmodels is the maintained,
battle-tested reference for HAC inference; we do not compete with it.

We pass ``use_correction=False`` deliberately. statsmodels' default applies a
``nobs/(nobs-k)`` small-sample correction to the HAC covariance; the retired
hand-rolled implementations these functions replace used the *uncorrected*
estimator. Disabling the correction makes this a provably behaviour-preserving
migration: the framework reproduces the retired gates' numbers to machine
precision (see ``tests/test_hac_parity.py``), so no historical promotion
decision silently flips. The correction is a finite-sample refinement — with it
enabled, t-stats shrink by exactly ``sqrt(nobs/(nobs-k))`` (~0.1% for a mean
t-stat, ~0.4% for a 3-factor alpha), the marginally-more-conservative direction
for an acceptance gate. Adopting it is a reasonable future policy change, but
belongs in a deliberate decision, not a silent side effect of this port.

All statistics are returned at full precision; rounding for display/JSON is the
caller's (CLI's) concern, not this module's.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def newey_west_auto_lags(n: int) -> int:
    """Newey-West (1994) automatic lag truncation: floor(4 * (n/100)^(2/9))."""
    return int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def require_finite(returns, *, allow_nonfinite_drop: bool = False) -> tuple[np.ndarray, int]:
    """Validate a raw return array: FAIL LOUDLY on NaN/Inf unless explicitly opted out.

    Returns ``(finite_returns, n_dropped)``. Raises ``ValueError`` when
    non-finite values are present and ``allow_nonfinite_drop`` is False — a
    corrupt input must never silently shrink the sample a gate then reports as
    complete. This is the same invariant as
    :func:`quantbox.analysis.dsr.deflated_sharpe_ratio_from_returns`.
    """
    r = np.asarray(returns, dtype=float)
    finite_mask = np.isfinite(r)
    n_dropped = int((~finite_mask).sum())
    if n_dropped and not allow_nonfinite_drop:
        raise ValueError(
            f"{n_dropped} of {r.size} return observations are NaN/Inf — refusing to silently "
            "drop them (a corrupt file must not pass on the surviving subset). Pass "
            "allow_nonfinite_drop=True to explicitly opt into dropping them and continuing."
        )
    return r[finite_mask], n_dropped


def newey_west_tstat(returns, lags: int | None = None, *, allow_nonfinite_drop: bool = False) -> dict:
    """Newey-West HAC t-stat on the mean of ``returns``.

    Equivalent to an OLS of ``returns`` on a constant with a HAC (Bartlett
    kernel) covariance — statsmodels does the sandwich. Corrects the standard
    error of the mean for the serial correlation that inflates a naive t-stat
    on overlapping/trend-following returns.

    Non-finite observations RAISE by default (see :func:`require_finite`);
    ``allow_nonfinite_drop=True`` opts into dropping them, and the returned
    ``n_obs_raw`` / ``n_nonfinite_dropped`` keep that loss visible.

    Returns a dict with full-precision floats. A degenerate series (fewer than
    2 finite observations, or zero long-run variance) yields ``None`` for the
    SE/t-stat/p-value rather than ``inf``/``nan``.
    """
    r, n_dropped = require_finite(returns, allow_nonfinite_drop=allow_nonfinite_drop)
    n = int(r.size)
    n_raw = int(np.asarray(returns).size)

    base = {
        "n_obs": n,
        "n_obs_raw": n_raw,
        "n_nonfinite_dropped": n_dropped,
        "nw_lags": 0,
        "mean_return": None,
        "nw_se": None,
        "nw_tstat": None,
        "nw_pvalue": None,
    }
    if n < 2:
        return base

    if lags is None:
        lags = newey_west_auto_lags(n)
    lags = max(0, min(lags, n - 1))  # can't use more lags than we have data
    base["nw_lags"] = lags

    mu = float(r.mean())
    base["mean_return"] = mu

    # Zero-variance series: statsmodels would divide by a zero SE. Guard first.
    if r.std(ddof=0) == 0:
        return base

    import statsmodels.api as sm

    x = np.ones((n, 1))
    fit = sm.OLS(r, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False})
    se = float(fit.bse[0])
    if not math.isfinite(se) or se <= 0:
        return base
    t = mu / se
    # Two-sided normal p-value, matching the retired gate's convention.
    pval = 2.0 * (1.0 - norm.cdf(abs(t)))
    return {
        **base,
        "nw_se": se,
        "nw_tstat": float(t),
        "nw_pvalue": float(pval),
    }


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

    import statsmodels.api as sm

    X = sm.add_constant(F, prepend=True, has_constant="add")  # n x k, intercept first
    # Perfectly collinear factors — refuse to fabricate an alpha (mirrors the
    # retired estimator's np.linalg.inv LinAlgError branch).
    if np.linalg.matrix_rank(X) < k:
        return base

    if lags is None:
        lags = newey_west_auto_lags(n)
    lags = max(0, min(lags, n - 1))
    base["hac_lags"] = int(lags)

    try:
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False})
    except (np.linalg.LinAlgError, ValueError):
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
