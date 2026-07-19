"""Numeric-parity proof: statsmodels HAC == the retired hand-rolled estimators.

The framework's ``quantbox.analysis.hac`` functions delegate the Newey-West HAC
sandwich to statsmodels, replacing hand-rolled ``(X'X)^-1 S (X'X)^-1`` Bartlett
loops that previously lived in quantbox-lab's ``nw-tstat-gate.py`` and
``factor-decomp-gate.py``. Those gates decided real promotions, so the port must
be provably behaviour-preserving.

This module carries the retired hand-rolled implementations verbatim as a frozen
oracle and pins two facts:

1. The framework (statsmodels, ``use_correction=False``) reproduces the
   hand-rolled Newey-West t-stat and factor-decomposition alpha t-stat to
   machine precision — i.e. the hand-rolled math was arithmetically correct and
   the migration changes no numbers.
2. statsmodels' *default* small-sample correction (``use_correction=True``)
   differs from the hand-rolled/framework result by exactly the textbook factor
   ``sqrt(nobs / (nobs - k))`` — a documented, marginally-more-conservative
   refinement, not a discrepancy. This pins the size of the choice so any future
   decision to adopt it is made with eyes open.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantbox.analysis import factor_regression, newey_west_auto_lags, newey_west_tstat


# --------------------------------------------------------------------------
# Retired hand-rolled implementations (verbatim, unrounded) — the oracle.
# Copied from quantbox-lab nw-tstat-gate.py::nw_tstat and
# factor-decomp-gate.py::factor_regression as they stood at the time of the port.
# --------------------------------------------------------------------------
def _handrolled_nw_tstat(returns, lags=None):
    r = np.asarray(returns, dtype=float)
    n = r.size
    if lags is None:
        lags = newey_west_auto_lags(n)
    lags = max(0, min(lags, n - 1))
    mu = float(r.mean())
    e = r - mu
    lrv = float(e @ e) / n  # gamma_0
    for lag in range(1, lags + 1):
        cov = float(e[lag:] @ e[:-lag]) / n  # gamma_l
        weight = 1.0 - lag / (lags + 1.0)  # Bartlett
        lrv += 2.0 * weight * cov
    se = math.sqrt(lrv / n)
    return {"lags": lags, "mean": mu, "se": se, "tstat": mu / se}


def _handrolled_factor_regression(y, factors, lags=None):
    y = np.asarray(y, dtype=float).ravel()
    F = np.asarray(factors, dtype=float)
    if F.ndim == 1:
        F = F.reshape(-1, 1)
    n = F.shape[0]
    X = np.column_stack([np.ones(n), F])  # intercept first
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    if lags is None:
        lags = newey_west_auto_lags(n)
    lags = max(0, min(lags, n - 1))
    s = X * resid[:, None]
    meat = s.T @ s
    for lag in range(1, lags + 1):
        g = s[lag:].T @ s[:-lag]
        weight = 1.0 - lag / (lags + 1.0)
        meat += weight * (g + g.T)
    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return {
        "lags": lags,
        "alpha": float(beta[0]),
        "alpha_se": float(se[0]),
        "alpha_tstat": float(beta[0] / se[0]),
        "betas": beta[1:].tolist(),
    }


# --------------------------------------------------------------------------
# Fixtures — autocorrelated returns + a factor panel (fixed seed).
# --------------------------------------------------------------------------
def _make_data(n=504, seed=42):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.01, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.0005 + 0.3 * r[i - 1] + e[i]  # AR(1)
    F = rng.normal(0, 0.012, (n, 3))
    y = 0.0003 + 0.5 * F[:, 0] + 0.2 * F[:, 1] + 0.4 * r
    return r, F, y


# --------------------------------------------------------------------------
# 1. Machine-precision parity — the migration changes no numbers.
# --------------------------------------------------------------------------
def test_newey_west_tstat_matches_handrolled_to_machine_precision():
    r, _, _ = _make_data()
    hr = _handrolled_nw_tstat(r)
    fw = newey_west_tstat(r)
    assert fw["nw_tstat"] == pytest.approx(hr["tstat"], abs=1e-10, rel=0)
    assert fw["nw_se"] == pytest.approx(hr["se"], abs=1e-12, rel=0)
    assert fw["mean_return"] == pytest.approx(hr["mean"], abs=1e-15, rel=0)
    assert fw["nw_lags"] == hr["lags"]


@pytest.mark.parametrize("lags", [0, 1, 5, 20])
def test_newey_west_tstat_parity_across_lag_choices(lags):
    r, _, _ = _make_data(seed=7)
    hr = _handrolled_nw_tstat(r, lags=lags)
    fw = newey_west_tstat(r, lags=lags)
    assert fw["nw_tstat"] == pytest.approx(hr["tstat"], abs=1e-10, rel=0)


def test_factor_regression_alpha_matches_handrolled_to_machine_precision():
    _, F, y = _make_data()
    names = ["mkt", "mom", "carry"]
    hr = _handrolled_factor_regression(y, F)
    fw = factor_regression(y, F, names)
    assert fw["alpha_tstat"] == pytest.approx(hr["alpha_tstat"], abs=1e-9, rel=0)
    assert fw["alpha"] == pytest.approx(hr["alpha"], abs=1e-12, rel=0)
    assert fw["alpha_se"] == pytest.approx(hr["alpha_se"], abs=1e-12, rel=0)
    for name, hr_beta in zip(names, hr["betas"], strict=True):
        assert fw["betas"][name] == pytest.approx(hr_beta, abs=1e-12, rel=0)


# --------------------------------------------------------------------------
# 2. The small-sample correction delta is exactly sqrt(n/(n-k)) — pinned.
# --------------------------------------------------------------------------
def test_correction_delta_equals_sqrt_n_over_n_minus_k_mean():
    """statsmodels use_correction=True vs the framework (=False): SE ratio for the
    mean t-stat (k=1) must equal sqrt(n/(n-1))."""
    import statsmodels.api as sm

    r, _, _ = _make_data()
    n = r.size
    lags = newey_west_auto_lags(n)
    corrected = sm.OLS(r, np.ones((n, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True})
    fw = newey_west_tstat(r, lags=lags)
    se_ratio = float(corrected.bse[0]) / fw["nw_se"]
    assert se_ratio == pytest.approx(math.sqrt(n / (n - 1)), rel=1e-9)
    # ~0.1% on 504 obs — real but small, and in the conservative direction.
    assert 1.0 < se_ratio < 1.002


def test_correction_delta_equals_sqrt_n_over_n_minus_k_factor():
    """For the 3-factor regression (k=4) the correction is sqrt(n/(n-4)) ~ 0.4%."""
    import statsmodels.api as sm

    _, F, y = _make_data()
    n, k = F.shape[0], F.shape[1] + 1
    lags = newey_west_auto_lags(n)
    X = sm.add_constant(F, prepend=True)
    corrected = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True})
    fw = factor_regression(y, F, ["mkt", "mom", "carry"])
    se_ratio = float(corrected.bse[0]) / fw["alpha_se"]
    assert se_ratio == pytest.approx(math.sqrt(n / (n - k)), rel=1e-9)
