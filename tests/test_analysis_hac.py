"""Tests for ``quantbox.analysis.hac`` — Newey-West HAC t-stat + factor decomposition.

Functional correctness, degenerate-input handling, and input validation. The
numeric-parity proof against the retired hand-rolled estimators lives in
``test_hac_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantbox.analysis import (
    factor_regression,
    newey_west_auto_lags,
    newey_west_tstat,
    require_finite,
)


def _ar1(n, phi=0.3, mu=0.0005, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = mu + phi * r[i - 1] + e[i]
    return r


# --- auto-lag rule ---


def test_auto_lags_newey_west_1994():
    assert newey_west_auto_lags(100) == 4
    assert newey_west_auto_lags(504) == 5
    assert newey_west_auto_lags(1000) == 6


# --- newey_west_tstat ---


def test_nw_tstat_positive_mean_detected():
    r = _ar1(504, mu=0.001, seed=1)
    out = newey_west_tstat(r)
    assert out["mean_return"] > 0
    assert out["nw_tstat"] is not None
    assert out["n_obs"] == 504
    assert out["nw_lags"] == newey_west_auto_lags(504)


def test_nw_tstat_lags_can_be_overridden():
    r = _ar1(300, seed=2)
    assert newey_west_tstat(r, lags=0)["nw_lags"] == 0
    assert newey_west_tstat(r, lags=10)["nw_lags"] == 10


def test_nw_tstat_autocorrelation_widens_se_vs_iid():
    """Positively autocorrelated returns should get a LARGER HAC SE than the
    naive iid SE (that is the whole point of the correction)."""
    r = _ar1(1000, phi=0.6, seed=3)
    hac = newey_west_tstat(r, lags=10)["nw_se"]
    iid = float(np.std(r, ddof=1) / np.sqrt(len(r)))
    assert hac > iid


def test_nw_tstat_degenerate_too_few_obs():
    out = newey_west_tstat(np.array([0.01]))
    assert out["nw_tstat"] is None
    assert out["mean_return"] is None
    assert out["n_obs"] == 1


def test_nw_tstat_zero_variance_returns_none_not_inf():
    out = newey_west_tstat(np.zeros(100))
    assert out["nw_tstat"] is None
    assert out["nw_se"] is None
    assert out["mean_return"] == 0.0


def test_nw_tstat_nan_rejected_by_default():
    r = _ar1(200, seed=4)
    r[10] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        newey_west_tstat(r)


def test_nw_tstat_nonfinite_drop_opt_in_reports_counts():
    r = _ar1(300, seed=5)
    r[3] = np.inf
    r[7] = np.nan
    out = newey_west_tstat(r, allow_nonfinite_drop=True)
    assert out["n_obs_raw"] == 300
    assert out["n_nonfinite_dropped"] == 2
    assert out["n_obs"] == 298


# --- require_finite ---


def test_require_finite_raises_on_nonfinite():
    with pytest.raises(ValueError, match="NaN/Inf"):
        require_finite(np.array([1.0, np.nan, 2.0]))


def test_require_finite_opt_in_drops_and_counts():
    finite, dropped = require_finite(np.array([1.0, np.nan, np.inf, 2.0]), allow_nonfinite_drop=True)
    assert dropped == 2
    assert finite.tolist() == [1.0, 2.0]


# --- factor_regression ---


def test_factor_regression_recovers_known_betas():
    rng = np.random.default_rng(6)
    n = 800
    F = rng.normal(0, 0.012, (n, 3))
    true_alpha, true_beta = 0.0004, np.array([0.8, -0.3, 0.5])
    noise = rng.normal(0, 0.002, n)
    y = true_alpha + F @ true_beta + noise
    reg = factor_regression(y, F, ["mkt", "mom", "carry"])
    assert reg["alpha"] == pytest.approx(true_alpha, abs=3e-4)
    assert reg["betas"]["mkt"] == pytest.approx(0.8, abs=0.02)
    assert reg["betas"]["mom"] == pytest.approx(-0.3, abs=0.02)
    assert reg["betas"]["carry"] == pytest.approx(0.5, abs=0.02)
    assert reg["r_squared"] > 0.95
    assert reg["n_factors"] == 3


def test_factor_regression_empty_factor_list_raises():
    with pytest.raises(ValueError, match="at least one factor column"):
        factor_regression(np.zeros(100), np.zeros((100, 0)), [])


def test_factor_regression_too_few_obs_returns_none_alpha():
    F = np.random.default_rng(8).normal(0, 0.01, (3, 3))
    reg = factor_regression(F[:, 0], F, ["a", "b", "c"])
    assert reg["alpha_tstat"] is None
    assert reg["betas"] is None


def test_factor_regression_collinear_factors_return_none_alpha():
    """Perfectly collinear factors (rank-deficient design) must not fabricate an alpha."""
    rng = np.random.default_rng(9)
    n = 300
    f = rng.normal(0, 0.01, n)
    F = np.column_stack([f, 2.0 * f])  # exact collinearity
    y = rng.normal(0, 0.01, n)
    reg = factor_regression(y, F, ["a", "b"])
    assert reg["alpha_tstat"] is None


def test_factor_regression_factor_names_length_mismatch_raises():
    rng = np.random.default_rng(11)
    n = 200
    F = rng.normal(0, 0.01, (n, 3))
    y = rng.normal(0, 0.01, n)
    with pytest.raises(ValueError, match="factor_names has"):
        factor_regression(y, F, ["mkt", "mom"])  # 2 names, 3 factor columns


def test_factor_regression_nonfinite_y_raises():
    rng = np.random.default_rng(12)
    n = 200
    F = rng.normal(0, 0.01, (n, 2))
    y = rng.normal(0, 0.01, n)
    y[5] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        factor_regression(y, F, ["a", "b"])


def test_factor_regression_nonfinite_factors_raises():
    rng = np.random.default_rng(13)
    n = 200
    F = rng.normal(0, 0.01, (n, 2))
    F[9, 1] = np.inf
    y = rng.normal(0, 0.01, n)
    with pytest.raises(ValueError, match="NaN/Inf"):
        factor_regression(y, F, ["a", "b"])


def test_factor_regression_y_length_mismatch_raises():
    rng = np.random.default_rng(14)
    F = rng.normal(0, 0.01, (200, 2))
    y = rng.normal(0, 0.01, 199)  # one short of the factor panel's row count
    with pytest.raises(ValueError, match="y has"):
        factor_regression(y, F, ["a", "b"])


def test_factor_regression_one_sided_pvalue_consistent_with_tstat():
    from scipy.stats import norm

    rng = np.random.default_rng(10)
    n = 500
    F = rng.normal(0, 0.01, (n, 2))
    y = 0.001 + F @ np.array([0.5, 0.2]) + rng.normal(0, 0.003, n)
    reg = factor_regression(y, F, ["a", "b"])
    assert reg["alpha_pvalue_onesided"] == pytest.approx(1.0 - norm.cdf(reg["alpha_tstat"]), abs=1e-9)
