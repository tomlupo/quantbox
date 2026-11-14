"""Statistical inference on a return series: the ONE implementation of each test (TOM-1618).

:mod:`quantbox.metrics` DESCRIBES a run (Sharpe, drawdown, turnover, IC, ...).
This module TESTS a claim about one: is the Sharpe real after deflation, is the
mean significant after autocorrelation, is the return more than factor
exposure, does a difference survive a resample. :mod:`quantbox.gates` DECIDES:
it puts thresholds on these statistics and returns a verdict.

The interface, grouped by question:

- Refusing an input: :class:`InferenceInputError` (a ``ValueError``) and
  :func:`require_finite`. Every function here refuses NaN/Inf by default and
  drops them only on ``allow_nonfinite_drop=True``, with the count reported.
- Moments: :func:`moments` (sample skew and kurtosis, Pearson by default) and
  :func:`return_moments` (the per-period Sharpe, skew and kurtosis of a return
  series, after the finite and degenerate checks).
- Newey-West / HAC: :func:`hac_ols`, :func:`newey_west_tstat`,
  :func:`newey_west_auto_lags`, :func:`factor_regression`. Each takes an
  optional calendar, ``segments`` (one label per row): lag products stay inside
  a run of equal labels, so a masked window is a gap, never compressed time
  (:func:`mask_runs` builds it from a keep mask).
- Predictive regression: :func:`predictive_regression` (OLS, Stambaugh
  bias-adjusted and Lewellen conditional slopes of r_t on x_{t-1}).
- Deflated Sharpe Ratio: :func:`deflated_sharpe_ratio`,
  :func:`deflated_sharpe_ratio_from_returns`, :func:`expected_max_sr`,
  :func:`sr_estimator_std`, :class:`DSRResult`.
- Sharpe difference: :func:`sharpe_difference_test` (Ledoit-Wolf 2008 HAC
  test of SR(candidate) = SR(baseline) on paired series).
- Multiple testing: :func:`multiple_testing` (Bonferroni, Holm, BHY),
  :func:`bonferroni_alpha`, :class:`MultipleTestingResult`.
- Resampling: :func:`bootstrap` (iid or paired stationary block),
  :func:`stationary_bootstrap_indices`, :func:`moving_block_indices`,
  :func:`gaussian_null`.
- Regime split: :func:`regime_sharpe` (the Sharpe split additively over
  bear / normal / bull benchmark-quantile regimes, a Gaussian null per part).
- Drawdown episodes: :func:`largest_drawdown_episode`.

Conventions, stated once:

- Sharpe, skew and kurtosis are PER PERIOD; annualising is the caller's display.
- Kurtosis is PEARSON (3.0 = normal) unless ``excess=True`` is asked for.
- Drawdowns are negative fractions, as in :mod:`quantbox.metrics`.
- A statistic that cannot be computed is refused (raise) when the INPUT is the
  problem, and ``None`` in the result dict when the FIT is degenerate.
- "Zero" is tested relatively, never with ``== 0`` (``DEGENERATE_RTOL``).

statsmodels computes the HAC sandwich and scipy the moments and the normal
tails (adapter, not reimplementation). All results are full precision;
rounding is the CLI's concern.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ._numerics import DEGENERATE_RTOL, flat
from .metrics import compute_drawdown_series, sharpe_ratio

EULER_MASCHERONI = 0.5772156649015329


class InferenceInputError(ValueError):
    """This input cannot produce the statistic — never a result, never a verdict.

    A ``ValueError`` so every ``except ValueError`` written before TOM-1618 still
    catches it. :data:`quantbox.gates.GateInputError` is this class.
    """


# ── refusing an input ──────────────────────────────────────────────────────


def require_finite(values, *, allow_nonfinite_drop: bool = False, what: str = "return") -> tuple[np.ndarray, int]:
    """``values`` without its NaN/Inf, and how many were dropped — RAISING unless the drop is opted into.

    A 1-D input is checked per observation. A 2-D input (rows x columns: a
    paired series, a factor panel) is checked per ROW: a row with any
    non-finite entry is dropped whole, so the columns stay aligned. A corrupt
    input must never shrink the sample silently while a test reports it as
    complete; ``allow_nonfinite_drop=True`` opts in and the count is returned.
    """
    a = np.asarray(values, dtype=float)
    if a.ndim > 1:
        mask = np.isfinite(a).all(axis=1)
        unit, verb = "rows", "carry"
    else:
        a = a.ravel()
        mask = np.isfinite(a)
        unit, verb = "observations", "are"
    dropped = int((~mask).sum())
    if dropped and not allow_nonfinite_drop:
        raise InferenceInputError(
            f"{dropped} of {mask.size} {what} {unit} {verb} NaN/Inf — refusing to drop them silently "
            "(a corrupt file must not pass on its surviving subset). Pass allow_nonfinite_drop=True "
            "(--allow-nonfinite-drop on the CLI) to drop them; the count is recorded."
        )
    return a[mask], dropped


# ── moments ────────────────────────────────────────────────────────────────


def moments(x, *, excess: bool = False, bias: bool = True) -> tuple[float, float]:
    """Sample ``(skew, kurtosis)`` of ``x`` — the one place a third or fourth moment is computed.

    ``kurtosis`` is Pearson (3.0 = normal) unless ``excess=True``.
    ``bias=False`` applies the small-sample correction (pandas' ``skew()`` /
    ``kurt()``). A series with fewer than two values, or one constant to
    floating-point noise, has the moments of a normal: ``(0.0, 3.0)`` (or
    ``(0.0, 0.0)`` in excess). Non-finite values are the caller's to refuse.
    """
    from scipy import stats

    a = np.asarray(x, dtype=float).ravel()
    if a.size < 2 or flat(a):
        return 0.0, (0.0 if excess else 3.0)
    return float(stats.skew(a, bias=bias)), float(stats.kurtosis(a, fisher=excess, bias=bias))


@dataclass(frozen=True)
class ReturnMoments:
    """What a Sharpe test needs from a return series, per period."""

    sr_period: float  # mean / std (ddof=1), not annualised
    T: int  # finite observations used
    skew: float
    kurtosis: float  # Pearson
    n_obs_raw: int  # observations before any non-finite drop
    n_nonfinite_dropped: int


def return_moments(returns, *, allow_nonfinite_drop: bool = False) -> ReturnMoments:
    """Per-period Sharpe, skew and Pearson kurtosis of ``returns``, after refusing what has none.

    Refuses (:class:`InferenceInputError`) NaN/Inf unless opted out, fewer than
    two finite observations, and a series constant to floating-point noise
    (``std(ddof=1)`` within ``DEGENERATE_RTOL`` of its mean absolute value).
    """
    raw = np.asarray(returns, dtype=float)
    r, n_dropped = require_finite(raw, allow_nonfinite_drop=allow_nonfinite_drop)
    T = int(r.size)
    if T <= 1:
        raise InferenceInputError(f"need at least 2 finite return observations, got T={T!r}")
    if flat(r, ddof=1):
        raise InferenceInputError(
            f"zero-variance returns — cannot compute a Sharpe ratio: the standard deviation "
            f"({float(r.std(ddof=1))!r}) is negligible against the scale of the series itself "
            f"(mean |return| = {float(np.mean(np.abs(r)))!r}), i.e. the series is constant to within "
            "floating-point noise"
        )
    skew, kurtosis = moments(r)
    return ReturnMoments(
        sr_period=sharpe_ratio(r, 1),
        T=T,
        skew=skew,
        kurtosis=kurtosis,
        n_obs_raw=int(raw.size),
        n_nonfinite_dropped=n_dropped,
    )


# ── Newey-West / HAC ───────────────────────────────────────────────────────


def newey_west_auto_lags(n: int) -> int:
    """Newey-West (1994) automatic lag truncation: floor(4 * (n/100)^(2/9))."""
    return int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def hac_ols(y: np.ndarray, x: np.ndarray, lags: int, segments: np.ndarray | None = None) -> Any:
    """OLS of ``y`` on ``x`` with a Newey-West (Bartlett kernel) HAC covariance — the ONE HAC fit.

    statsmodels computes the sandwich (adapter, not reimplementation). The
    ``nobs/(nobs-k)`` small-sample correction is OFF: the retired hand-rolled
    estimators (and robo-lab's ``_ols_nw``) used the uncorrected estimator, and
    ``tests/test_hac_parity.py`` pins that parity. With it on, t-stats shrink by
    exactly ``sqrt(nobs/(nobs-k))`` (~0.1% for a mean t-stat, ~0.4% for a
    3-factor alpha); adopting it is a policy decision, not a side effect.

    ``segments`` is the calendar (TOM-3700): one label per row, a new run of
    consecutive periods wherever the label changes. The lag-``l`` products
    ``u_t u_{t-l}`` are summed inside each run only, so they never cross a gap
    (statsmodels' ``hac-panel`` with the runs as groups, the same Bartlett
    weights, no correction). One run is the plain HAC fit. A run of ``m`` rows
    holds no product past lag ``m - 1``: when the longest run is that short,
    the sum stops there with the Bartlett weights of ``lags`` (statsmodels
    refuses a lag no run holds), and when every run is one row it is the
    White (HC0) SE. Returns the statsmodels result.
    """
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import weights_bartlett

    if segments is None:
        return sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False})
    s = np.asarray(segments).ravel()
    longest = int(np.diff(np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])).max())
    held = min(lags, longest - 1)  # the highest lag some run holds a product at
    if held == 0:
        return sm.OLS(y, x).fit(cov_type="HC0")
    weights = weights_bartlett(lags)[: held + 1]
    return sm.OLS(y, x).fit(
        cov_type="hac-panel",
        cov_kwds={"maxlags": held, "groups": s, "kernel": lambda _: weights, "use_correction": False},
    )


def _segment_runs(segments, finite: np.ndarray) -> np.ndarray:
    """``segments`` (one label per raw row) as run ids of the finite rows; a dropped row breaks its run.

    A run is a stretch of consecutive rows with one label. A non-finite row the
    caller drops is a missing period too, so the rows on either side of it land
    in different runs: dropping it must not glue them into neighbours.
    """
    s = np.asarray(segments).ravel()
    if s.size != finite.size:
        raise InferenceInputError(
            f"segments has {s.size} labels but the series has {finite.size} rows — one label per row"
        )
    breaks = np.concatenate([[False], s[1:] != s[:-1]]) | ~finite
    return np.cumsum(breaks)[finite]


def mask_runs(keep) -> np.ndarray:
    """The calendar of a masked series: run ids of the kept rows, a new run after every removed stretch."""
    keep = np.asarray(keep, dtype=bool).ravel()
    return np.cumsum(~keep)[keep]


def _check_overlap(overlap, lags: int | None) -> None:
    """Refuse an ``overlap`` that is not a whole number of periods >= 1, or one an explicit ``lags`` undercuts."""
    if isinstance(overlap, bool) or not isinstance(overlap, (int, np.integer)) or overlap < 1:
        raise InferenceInputError(
            f"overlap must be a whole number of periods >= 1 (1 = non-overlapping), got {overlap!r}"
        )
    if lags is not None and lags < overlap - 1:
        raise InferenceInputError(
            f"lags={lags!r} conflicts with overlap={overlap!r}: observations that overlap by "
            f"{overlap - 1} periods need at least {overlap - 1} Newey-West lags (Hansen-Hodrick 1980). "
            "Pass lags >= overlap - 1, or drop lags and let overlap set them."
        )


def newey_west_tstat(
    returns,
    lags: int | None = None,
    *,
    overlap: int | None = None,
    allow_nonfinite_drop: bool = False,
    segments=None,
) -> dict:
    """Newey-West HAC t-stat on the mean of ``returns`` — the ``nw`` gate calls this one.

    An OLS of ``returns`` on a constant with a HAC (Bartlett kernel)
    covariance (:func:`hac_ols`). Corrects the standard error of the mean for
    the serial correlation that inflates a naive t-stat on overlapping or
    trend-following returns. ``lags=None`` picks :func:`newey_west_auto_lags`.

    ``overlap=h`` declares that each observation spans ``h`` periods of the
    series and its neighbours share ``h - 1`` of them (an IC on h-day forward
    returns sampled daily, a rolling h-day spread). Such a series is an
    MA(h-1) by construction, and the automatic lag count alone leaves its
    t-stat overstated. The rule (Hansen and Hodrick 1980, "Forward Exchange
    Rates as Optimal Predictors of Future Spot Rates", JPE 88(5)): the lag
    count is ``max(newey_west_auto_lags(n), h - 1)``. ``h`` counts periods
    of THIS series; converting a calendar window (21 trading days in a
    monthly series) is the caller's. Test each horizon on its own, with that
    horizon's ``h``. An explicit ``lags`` still wins, but one below
    ``h - 1`` is refused, and so is an ``h`` longer than the sample (its
    floor could only be met by clamping, which would hide the overstatement).
    ``overlap=1`` means non-overlapping and changes nothing.

    Non-finite observations RAISE by default (see :func:`require_finite`);
    ``allow_nonfinite_drop=True`` opts into dropping them, and the returned
    ``n_obs_raw`` / ``n_nonfinite_dropped`` keep that loss visible.

    ``segments`` is the series' calendar (see :func:`hac_ols`), one label per
    row of ``returns``: the mean is over every finite row, the lag products
    stay inside each run of consecutive periods. ``lags=None`` still counts
    ``n`` as the total. A dropped non-finite row breaks its run.

    Returns a dict with full-precision floats: ``n_obs``, ``n_obs_raw``,
    ``n_nonfinite_dropped``, ``nw_lags``, ``nw_overlap`` (the ``overlap``
    passed, or ``None``), ``mean_return``, ``nw_se``, ``nw_tstat`` and
    ``nw_pvalue`` (two-sided, normal). A degenerate series (fewer than 2
    finite observations, or zero long-run variance) yields ``None`` for the
    SE/t-stat/p-value rather than ``inf``/``nan``.
    """
    if overlap is not None:
        _check_overlap(overlap, lags)
        overlap = int(overlap)
    r, n_dropped = require_finite(returns, allow_nonfinite_drop=allow_nonfinite_drop)
    n = int(r.size)
    n_raw = int(np.asarray(returns).size)
    if segments is not None:
        segments = _segment_runs(segments, np.isfinite(np.asarray(returns, dtype=float).ravel()))
    if overlap is not None and overlap > n:
        raise InferenceInputError(
            f"overlap={overlap} needs at least {overlap - 1} Newey-West lags, but {n} finite "
            f"observations allow at most {max(n - 1, 0)} — the sample is shorter than one overlap window"
        )

    base = {
        "n_obs": n,
        "n_obs_raw": n_raw,
        "n_nonfinite_dropped": n_dropped,
        "nw_lags": 0,
        "nw_overlap": overlap,
        "mean_return": None,
        "nw_se": None,
        "nw_tstat": None,
        "nw_pvalue": None,
    }
    if n < 2:
        return base

    if lags is None:
        lags = newey_west_auto_lags(n)
        if overlap is not None:
            lags = max(lags, overlap - 1)  # Hansen-Hodrick floor
    lags = max(0, min(lags, n - 1))  # can't use more lags than we have data
    base["nw_lags"] = lags

    mu = float(r.mean())
    base["mean_return"] = mu

    # Zero-variance series: statsmodels would divide by a zero SE. Guard first,
    # RELATIVELY (see DEGENERATE_RTOL): a constant series does not reliably
    # come out with std == 0.0 — `[0.001] * 337` gives ~2e-19, which statsmodels
    # turned into a t-stat of 3.5e16 (29 of 48 constant series probed, TOM-1351).
    if flat(r):
        return base

    fit = hac_ols(r, np.ones((n, 1)), lags, segments)
    se = float(fit.bse[0])
    if not math.isfinite(se) or se <= 0:
        return base
    t = mu / se
    from scipy.stats import norm

    # Two-sided normal p-value, matching the retired gate's convention.
    pval = 2.0 * (1.0 - norm.cdf(abs(t)))
    return {
        **base,
        "nw_se": se,
        "nw_tstat": float(t),
        "nw_pvalue": float(pval),
    }


def factor_regression(y, factors, factor_names: list[str], lags: int | None = None, *, segments=None) -> dict:
    """OLS of ``y`` on an intercept + ``factors`` with Newey-West HAC SEs (Jensen's alpha).

    Fits ``y = alpha + factors @ beta + e`` and returns the betas, Jensen's
    alpha (the intercept), and the HAC-robust SE / one-sided t-stat of the
    alpha under H0: alpha <= 0. ``y`` is the strategy return series (shape n);
    ``factors`` is the aligned factor panel (shape n x k); ``factor_names``
    labels the k columns.

    A degenerate fit (too few observations, a singular / perfectly collinear
    design, residuals cancelled to noise) yields ``alpha_tstat = None`` rather
    than a spurious number. Refuses an empty factor list (an intercept-only
    fit is not a decomposition), mismatched shapes and NaN/Inf.

    ``segments`` is the calendar (see :func:`hac_ols`): one label per row, the
    HAC lag products stay inside each run of consecutive periods.
    """
    from scipy.stats import norm

    if not factor_names:
        raise InferenceInputError(
            "factor_regression requires at least one factor column, got an empty factor_names list"
        )

    y = np.asarray(y, dtype=float).ravel()
    F = np.asarray(factors, dtype=float)
    if F.ndim == 1:
        F = F.reshape(-1, 1)
    n, k_f = F.shape
    if k_f == 0:
        raise InferenceInputError("factor panel has zero columns — cannot run a factor decomposition")
    if len(factor_names) != k_f:
        raise InferenceInputError(
            f"factor_names has {len(factor_names)} entries but the factor panel has {k_f} "
            "columns — these must match 1:1."
        )
    if y.size != n:
        raise InferenceInputError(
            f"y has {y.size} observations but the factor panel has {n} rows — these must match 1:1."
        )
    require_finite(y, what="y")
    require_finite(F, what="factor panel")
    if segments is not None:
        segments = _segment_runs(segments, np.ones(n, dtype=bool))

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
        fit = hac_ols(y, X, lags, segments)
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


# ── predictive regression ──────────────────────────────────────────────────
#
# r_t = alpha + beta * x_{t-1} + e_t, with the predictor an AR(1),
# x_t = phi + rho * x_{t-1} + u_t. When e and u are correlated (a price-scaled
# ratio such as D/P: a high return lowers the yield, so gamma, the slope of e on
# u, is < 0), OLS beta is biased by gamma * E[rho_hat - rho] (Stambaugh 1999,
# "Predictive Regressions", JFE 54(3)), and rho_hat is biased down by about
# (1 + 3 rho) / T (Kendall 1954). Both corrections below are the coefficient on
# x_{t-1} in ONE augmented regression, r_t on [1, x_{t-1}, x_t - rho* x_{t-1}],
# which equals beta_hat - gamma_hat * (rho_hat - rho*) exactly; only rho* differs.

#: Lewellen's (2004) bound on the predictor's autocorrelation: stationarity, rho < 1.
LEWELLEN_RHO_BOUND = 0.9999


def predictive_regression(
    returns, predictor, *, rho_bound: float = LEWELLEN_RHO_BOUND, allow_nonfinite_drop: bool = False
) -> dict:
    """OLS, Stambaugh bias-adjusted and Lewellen conditional slopes of ``returns[t]`` on ``predictor[t-1]``.

    ``returns`` and ``predictor`` are ALIGNED per-period series of one length:
    row ``t`` holds the return over period ``t`` and the predictor observed at
    the END of period ``t``. This function does the lag itself, so a caller
    who lags ``predictor`` first tests a contemporaneous regression; the first
    return and the last predictor value carry no pair. Different lengths are
    refused. NaN/Inf raise unless ``allow_nonfinite_drop=True``; a pair
    ``(r_t, x_{t-1}, x_t)`` with any non-finite entry is then dropped whole, so
    no surviving row pairs a return with the wrong lag.

    Three estimates of beta, all with classical (not HAC) standard errors, as
    in both papers; every p-value is ONE-sided, H1: beta > 0 (negate the
    predictor to test beta < 0):

    - ``beta_ols``: the plain OLS slope, t-tail with ``n - 2`` d.o.f.
    - ``beta_stambaugh``: ``beta_ols - stambaugh_bias``, where
      ``stambaugh_bias = gamma * E[rho_hat - rho] = -gamma * (1 + 3 rho_hat)/n``
      estimates ``E[beta_hat - beta]`` with the first-order AR(1) bias taken
      at ``rho_hat`` (Stambaugh 1999; Lewellen's "Stambaugh" column). It is
      the augmented fit at ``rho_corrected = rho_hat + (1 + 3 rho_hat)/n``,
      which is NOT capped: above 1 it says the correction assumes a
      non-stationary predictor, which is Lewellen's case for the conditional
      test. The standard error adds the noise of the correction by the delta
      method, ``se**2 = se_aug**2 + gamma**2 * (1 + 3/n)**2 * var(rho_hat)``
      (the first-order form of Amihud and Hurvich 2004, "Predictive
      Regressions: A Reduced-Bias Estimation and Inference Approach", JFQA
      39(4)); normal tail. Lewellen simulates this p-value instead.
    - ``beta_conditional``: Lewellen (2004, "Predicting Returns with Financial
      Ratios", JFE 74(2), Appendix A.1), the augmented regression with rho
      fixed at ``rho_bound`` (default 0.9999: the true rho is assumed to be at
      most that, i.e. the predictor is stationary). Its OLS standard error is
      exact conditional on the predictor, t-tail with ``n - 3`` d.o.f. With
      ``gamma < 0`` the bias is largest at the largest rho, so this slope is a
      LOWER bound on the bias-adjusted beta and the test is conservative
      (``conditional_conservative``); with ``gamma >= 0`` it is an UPPER
      bound and must not be read as evidence for beta > 0. It gains power
      only when ``rho_hat`` is close to the bound; far below it the
      correction over-adjusts.

    Refuses (:class:`InferenceInputError`) mismatched lengths, a ``rho_bound``
    outside (0, 1], fewer than 5 usable pairs (the augmented fit has 3
    parameters), a constant return or predictor series, a predictor whose
    AR(1) innovations cancel to noise (no ``gamma`` exists), and returns that
    are an exact function of the predictor (no residual variance). Rows must
    be CONSECUTIVE periods: a period missing from both series is not
    detectable here, and the lag would pair across it. All floats are full
    precision.
    """
    import statsmodels.api as sm
    from scipy.stats import norm
    from scipy.stats import t as student_t

    if isinstance(rho_bound, bool) or not isinstance(rho_bound, (int, float, np.floating)):
        raise InferenceInputError(f"rho_bound must be a number in (0, 1], got {rho_bound!r}")
    if not (math.isfinite(rho_bound) and 0.0 < rho_bound <= 1.0):
        raise InferenceInputError(f"rho_bound must be in (0, 1] (a stationarity bound on rho), got {rho_bound!r}")
    r = np.asarray(returns, dtype=float).ravel()
    x = np.asarray(predictor, dtype=float).ravel()
    if r.size != x.size:
        raise InferenceInputError(
            f"returns has {r.size} observations and predictor {x.size} — a predictive regression needs the "
            "same periods in both, aligned row by row (the function lags the predictor itself)"
        )
    if r.size < 2:
        raise InferenceInputError(f"need at least 2 aligned periods to form one (r_t, x_t-1) pair, got {r.size}")
    rows, dropped = require_finite(
        np.column_stack([r[1:], x[:-1], x[1:]]), allow_nonfinite_drop=allow_nonfinite_drop, what="(r_t, x_t-1, x_t)"
    )
    n = int(rows.shape[0])
    if n < 5:
        raise InferenceInputError(f"need at least 5 finite (r_t, x_t-1, x_t) pairs for the augmented fit, got {n}")
    y, x_lag, x_now = rows[:, 0], rows[:, 1], rows[:, 2]
    if flat(y, ddof=1):
        raise InferenceInputError("constant returns — there is no variation for the predictor to explain")
    if flat(x_lag, ddof=1):
        raise InferenceInputError("constant predictor — its slope is not identified")

    ones = np.ones(n)
    ols = sm.OLS(y, np.column_stack([ones, x_lag])).fit()
    ar = sm.OLS(x_now, np.column_stack([ones, x_lag])).fit()
    u = np.asarray(ar.resid, dtype=float)
    # Relative to the PREDICTOR's scale, not the residuals' own: an exact AR(1) leaves
    # residuals that are pure rounding noise, and noise is never flat against itself.
    if float(np.std(u)) <= DEGENERATE_RTOL * float(np.mean(np.abs(x_now))):
        raise InferenceInputError(
            "the predictor is a deterministic AR(1) (its innovations cancel to noise) — gamma is not identified"
        )
    rho_hat, rho_se = float(ar.params[1]), float(ar.bse[1])
    gamma = float(np.dot(np.asarray(ols.resid, dtype=float), u) / np.dot(u, u))

    def augmented(rho_star: float):
        return sm.OLS(y, np.column_stack([ones, x_lag, x_now - rho_star * x_lag])).fit()

    beta_ols, ols_se = float(ols.params[1]), float(ols.bse[1])
    ols_t = beta_ols / ols_se

    rho_corrected = rho_hat + (1 + 3 * rho_hat) / n  # rho_hat - E[rho_hat - rho], Kendall's first order
    aug_s = augmented(rho_corrected)
    beta_s = float(aug_s.params[1])
    var_rho_c = (1 + 3 / n) ** 2 * rho_se**2
    se_s = math.sqrt(float(aug_s.bse[1]) ** 2 + gamma**2 * var_rho_c)
    t_s = beta_s / se_s

    aug_c = augmented(float(rho_bound))
    # A return built exactly from the predictor has no residual variance, and statsmodels
    # would report a noise-sized SE and a t-stat of ~1e16 (as in factor_regression, TOM-1351).
    if float(np.std(aug_c.resid)) <= DEGENERATE_RTOL * float(np.mean(np.abs(y))):
        raise InferenceInputError("the returns are an exact function of the predictor — no residual variance")
    beta_c, se_c = float(aug_c.params[1]), float(aug_c.bse[1])
    t_c = beta_c / se_c

    return {
        "n_obs": n,
        "n_obs_raw": int(r.size - 1),
        "n_nonfinite_dropped": dropped,
        "beta_ols": beta_ols,
        "beta_ols_se": ols_se,
        "beta_ols_tstat": ols_t,
        "beta_ols_pvalue_onesided": float(student_t.sf(ols_t, n - 2)),
        "r_squared": float(ols.rsquared),
        "rho_hat": rho_hat,
        "rho_hat_se": rho_se,
        "rho_corrected": rho_corrected,
        "gamma": gamma,
        "corr_e_u": float(np.corrcoef(np.asarray(ols.resid, dtype=float), u)[0, 1]),
        "stambaugh_bias": beta_ols - beta_s,
        "beta_stambaugh": beta_s,
        "beta_stambaugh_se": se_s,
        "beta_stambaugh_tstat": t_s,
        "beta_stambaugh_pvalue_onesided": float(norm.sf(t_s)),
        "rho_bound": float(rho_bound),
        "beta_conditional": beta_c,
        "beta_conditional_se": se_c,
        "beta_conditional_tstat": t_c,
        "beta_conditional_pvalue_onesided": float(student_t.sf(t_c, n - 3)),
        "conditional_df": n - 3,
        "conditional_conservative": bool(gamma < 0),
    }


# ── Deflated Sharpe Ratio ──────────────────────────────────────────────────
#
# Bailey & López de Prado (2014), "The Deflated Sharpe Ratio", with the Mertens
# (2002) correction for skew/kurtosis in the variance of the Sharpe estimator.
# Ported verbatim (math unchanged) from quantbox-lab ``quark/fix-dsr-gate``
# (``scripts/lib/dsr.py``), confirmed by an independent cross-model review. That
# fix replaced a gate that tested ``t = sharpe * sqrt(n_years)`` against a
# Bonferroni NORMAL null — silently assuming zero skew/kurtosis and
# under-deflating at small trial counts. The ``sqrt(periods)`` annualisation a
# caller applies for display assumes i.i.d. returns; for an autocorrelation-
# robust t-stat use :func:`newey_west_tstat`.


@dataclass(frozen=True)
class DSRResult:
    T: int
    sr_period: float
    skew: float
    kurtosis: float  # non-excess (Pearson) kurtosis — matches scipy.stats.kurtosis(fisher=False)
    sr_std: float  # std of the Sharpe-ratio estimator, per-period units
    sr0_period: float  # expected-max-SR deflation benchmark, per-period units
    z: float  # (sr - sr0) / sr_std
    psr_vs_zero: float  # P(true SR > 0) — undeflated, single-trial reference
    dsr: float  # P(true SR > deflated benchmark) — the actual DSR statistic
    n_trials: int
    n_obs_raw: int = 0  # raw observation count BEFORE any non-finite filtering
    n_nonfinite_dropped: int = 0  # how many NaN/Inf observations were dropped (0 unless opted in)


def expected_max_sr(n_trials: int) -> float:
    """Expected max Sharpe over ``n_trials`` iid N(0,1) trials.

    Exact Bailey & López de Prado (2014) form (not the leading-order
    asymptotic, which under-deflates at small N).

    Raises for n_trials <= 0 — there is no such thing as "zero or negative
    trials"; silently coercing this to 1 (as the old gate did) quietly removes
    the entire multiple-testing penalty.
    """
    from scipy import stats

    if n_trials <= 0:
        raise InferenceInputError(f"n_trials must be a positive integer, got {n_trials!r}")
    if n_trials == 1:
        return 0.0
    return (1 - EULER_MASCHERONI) * stats.norm.ppf(1 - 1.0 / n_trials) + EULER_MASCHERONI * stats.norm.ppf(
        1 - 1.0 / (n_trials * math.e)
    )


def sr_estimator_std(T: int, sr: float, skew: float, kurtosis: float) -> float:
    """Std of the Sharpe-ratio estimator (Mertens 2002 / Bailey-López de Prado), per-period units."""
    if T <= 1:
        raise InferenceInputError(f"need at least 2 observations to estimate SR variance, got T={T!r}")
    # The numerator is a three-term sum, and the terms can cancel. Keep them
    # separately so the cancellation can be measured against their own scale.
    terms = (1.0, -skew * sr, (kurtosis - 1) / 4 * sr**2)
    numerator = sum(terms)
    scale = sum(abs(t) for t in terms)
    # Degeneracy is tested FIRST, and on |numerator|, because the sign of a
    # cancelled sum is itself rounding noise. Identity:
    #
    #     numerator - (1 - skew*sr/2)**2 == ((kurtosis - 1) - skew**2)/4 * sr**2
    #
    # so under the Pearson bound kurtosis >= skew**2 + 1 (enforced by the
    # caller) the numerator is bounded below by (1 - skew*sr/2)**2 and reaches
    # zero only on the knife edge kurtosis == skew**2 + 1 AND skew*sr == 2 --
    # a two-point distribution whose Sharpe estimator has no spread, where the
    # DSR is genuinely undefined. In exact arithmetic that edge gives 0; in
    # floating point it lands either side of it, so 21.5% of a 4000-point sweep
    # along the edge came out at ~-2e-17 and, when `numerator < 0` was tested
    # first, was reported as "check inputs" -- telling a researcher their
    # moments were garbage when they were a valid two-point distribution. That
    # is the same exact-comparison lottery this guard exists to remove, so the
    # order matters and is part of the contract.
    #
    # The test is relative to the TERMS, not to the numerator alone, because
    # this is a catastrophic-cancellation test: it asks whether the sum still
    # carries significant digits. (Note the numerator is T-independent -- the
    # 1/(T-1) is applied below -- so this is not about long series.)
    #
    # It closes the numerics sliver, NOT the modelling class: at skew*sr - 2 =
    # 1e-5 the numerator is still accurate to 8 significant digits, so the
    # series is accepted and reports dsr ~ 1.0. Whether a near-two-point
    # distribution should be refused at all is a modelling decision this
    # function deliberately does not make.
    if abs(numerator) <= DEGENERATE_RTOL * scale:
        raise InferenceInputError(
            f"degenerate SR-estimator variance: the numerator ({numerator!r}) is negligible "
            f"against the scale of its own terms ({scale!r}) — the moments T={T}, sr={sr}, "
            f"skew={skew}, kurtosis={kurtosis} describe a distribution whose Sharpe estimator "
            "has no spread, so the DSR is undefined"
        )
    if numerator < 0:
        # Genuinely negative (not cancellation noise -- that was caught above):
        # pathological (garbage) skew/kurtosis/sr combinations. Fail loudly
        # rather than silently sqrt()-ing a negative number to NaN.
        raise InferenceInputError(
            f"negative SR-estimator variance ({numerator / (T - 1)!r}) from T={T}, sr={sr}, "
            f"skew={skew}, kurtosis={kurtosis} — check inputs"
        )
    return math.sqrt(numerator / (T - 1))


def deflated_sharpe_ratio(
    sr: float,
    T: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
    *,
    trials_sr_std: float | None = None,
) -> DSRResult:
    """Compute the genuine Deflated Sharpe Ratio.

    Parameters are all PER-PERIOD (not annualised): ``sr`` is the per-period
    Sharpe, ``skew``/``kurtosis`` are the per-period return-distribution
    moments, ``T`` is the number of return observations.

    The deflation benchmark is ``sr0 = sigma * expected_max_sr(n_trials)``, where
    sigma is the cross-trial standard deviation of the Sharpe ratios tried. When
    the sweep's Sharpes are known, pass their per-period std as
    ``trials_sr_std`` (the textbook form). Omitted, sigma is this series' own
    estimator std — the null in which every trial had zero true Sharpe and this
    estimator's noise (the default, and the only form before TOM-1351).

    Returns a DSRResult; ``result.dsr`` is P(true SR exceeds the
    multiple-testing-deflated benchmark). This function applies no threshold —
    an acceptance gate is a policy decision (:mod:`quantbox.gates`).
    """
    from scipy import stats

    if not math.isfinite(sr):
        raise InferenceInputError(f"sharpe must be finite, got {sr!r}")
    if not math.isfinite(skew) or not math.isfinite(kurtosis):
        raise InferenceInputError(f"skew/kurtosis must be finite, got skew={skew!r} kurtosis={kurtosis!r}")
    # Pearson (non-excess) kurtosis is mathematically bounded below by
    # skew**2 + 1 for ANY real distribution — this is not a modelling
    # choice, it's an algebraic identity (Var(Z^2) >= 0 for standardized Z).
    # A caller passing e.g. skew=2, kurtosis=1 (impossible: min is 5) is
    # almost always a units/convention bug (most commonly: excess kurtosis
    # from scipy.stats.kurtosis()'s default fisher=True passed where this
    # function expects Pearson/non-excess). Feeding an impossible moment
    # pair into the Mertens variance term can silently flip a FAIL to a
    # PASS, so this is refused rather than "trusted".
    min_kurtosis = skew**2 + 1
    # Relative slack, for the same reason as DEGENERATE_RTOL: this bound is
    # attained EXACTLY by a two-point distribution (a binary-payoff strategy:
    # win x, lose y, flat sizing -- an ordinary object in this domain), and its
    # sample moments land ~1e-14 either side of the bound. Measured on 4416
    # two-point samples, 40.6% fell just below it and were refused with a
    # confident, wrong, actionable-in-the-wrong-direction diagnosis ("this
    # usually means EXCESS kurtosis was passed"). The slack is ~1e-12 relative
    # and cannot mask the bug this check is for: excess-vs-Pearson confusion is
    # an O(3) error (kurtosis 0.0 where 3.0 was required), twelve orders larger.
    kurtosis_tol = DEGENERATE_RTOL * (abs(kurtosis) + min_kurtosis)
    if kurtosis < min_kurtosis - kurtosis_tol:
        raise InferenceInputError(
            f"impossible moments: kurtosis={kurtosis!r} < skew**2 + 1 = {min_kurtosis!r} "
            f"(skew={skew!r}). Pearson (non-excess) kurtosis is always >= skew**2 + 1 for any "
            "real distribution. This usually means EXCESS kurtosis was passed instead of Pearson "
            "— scipy.stats.kurtosis() returns excess kurtosis by default (fisher=True); this "
            "function requires fisher=False (3.0 for a normal distribution, not 0.0)."
        )

    # No `sr_std == 0` check here: sr_estimator_std owns the cancellation test,
    # where the terms of the variance sum exist to measure it against. It
    # returns a strictly positive value (its scale always includes the literal
    # 1.0 term), and a second, weaker copy of a guard is how the first one stops
    # being the real protection.
    sr_std = sr_estimator_std(T, sr, skew, kurtosis)

    if trials_sr_std is not None and not (math.isfinite(trials_sr_std) and trials_sr_std > 0):
        raise InferenceInputError(f"trials_sr_std must be finite and positive, got {trials_sr_std!r}")
    e_max = expected_max_sr(n_trials)
    sr0 = (sr_std if trials_sr_std is None else trials_sr_std) * e_max
    z = (sr - sr0) / sr_std
    psr0 = stats.norm.cdf(sr / sr_std)
    dsr = stats.norm.cdf(z)

    return DSRResult(
        T=T,
        sr_period=sr,
        skew=skew,
        kurtosis=kurtosis,
        sr_std=sr_std,
        sr0_period=sr0,
        z=z,
        psr_vs_zero=psr0,
        dsr=dsr,
        n_trials=n_trials,
        # Called directly with already-computed moments — no filtering has
        # happened at this layer, so raw == surviving and nothing was dropped.
        n_obs_raw=T,
        n_nonfinite_dropped=0,
    )


def deflated_sharpe_ratio_from_returns(returns, n_trials: int, *, allow_nonfinite_drop: bool = False) -> DSRResult:
    """The DSR of a per-period return series (any 1-D array-like).

    The series goes through :func:`return_moments` — the same refusals as
    every test here: NaN/Inf raise unless ``allow_nonfinite_drop=True`` (then
    ``n_obs_raw`` and ``n_nonfinite_dropped`` report the loss), fewer than two
    finite observations raise, a series constant to floating-point noise raises.
    """
    m = return_moments(returns, allow_nonfinite_drop=allow_nonfinite_drop)
    result = deflated_sharpe_ratio(sr=m.sr_period, T=m.T, skew=m.skew, kurtosis=m.kurtosis, n_trials=n_trials)
    return replace(result, n_obs_raw=m.n_obs_raw, n_nonfinite_dropped=m.n_nonfinite_dropped)


# ── Sharpe difference (Ledoit-Wolf 2008) ───────────────────────────────────
#
# Ledoit & Wolf (2008), "Robust performance hypothesis testing with the Sharpe
# ratio", J. Empirical Finance 15(5), 850-859, Section 3.1. Delta = SR_c - SR_b
# is a smooth function f of v = (mu_c, mu_b, g_c, g_b), the means and
# uncentred second moments of the two series; by the delta method
# se(Delta) = sqrt(grad f' Psi grad f / T), with Psi the long-run covariance
# of y_t = (r_c - mu_c, r_b - mu_b, r_c^2 - g_c, r_b^2 - g_b). Psi is the
# paper's kernel estimate (its Section 4, columns HAC and HACPW): the
# Quadratic-Spectral kernel with Andrews' (1991, "Heteroskedasticity and
# Autocorrelation Consistent Covariance Matrix Estimation", Econometrica 59(3))
# automatic bandwidth from AR(1) fits, the T / (T - 4) correction, and
# optionally Andrews-Monahan (1992, Econometrica 60(4)) VAR(1) prewhitening.
# The lag sum is statsmodels' ``S_hac_simple`` (adapter, as in
# :func:`hac_ols`); its Bartlett weights are swapped for the QS ones because
# the paper's reference p-values are QS (Bartlett/Newey-West lags give
# different ones). The idea of offering this test came from the qis library;
# the code is written from the paper and the authors' published R code, no qis
# code is copied and qis is not a dependency.


_RETURN_UNITS = {"fraction": 100.0, "percent": 1.0}  # the factor to percent, the paper's unit


def _quadratic_spectral(x: np.ndarray) -> np.ndarray:
    """The QS kernel, 25 / (12 pi^2 x^2) * (sin(z) / z - cos(z)) with z = 6 pi x / 5, and 1 at x = 0."""
    x = np.asarray(x, dtype=float)
    z = 6 * np.pi * x / 5
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 25 / (12 * np.pi**2 * x**2) * (np.sin(z) / z - np.cos(z))
    return np.where(x == 0, 1.0, k)


def _lw_long_run_cov(y: np.ndarray) -> tuple[np.ndarray, float]:
    """QS-kernel long-run covariance of the rows of ``y`` (mean zero), times T / (T - k), and its bandwidth.

    The bandwidth is Andrews (1991) eq. 6.2 for the QS kernel,
    ``1.3221 * (alpha(2) * T)^(1/5)``, with ``alpha(2)`` from an AR(1) fit per
    column, equally weighted (his eq. 6.4). Every lag 1 .. T-1 is summed.
    """
    from statsmodels.stats.sandwich_covariance import S_hac_simple

    T, k = y.shape
    num = den = 0.0
    for col in y.T:
        x = np.column_stack([np.ones(T - 1), col[:-1]])
        coef, *_ = np.linalg.lstsq(x, col[1:], rcond=None)
        rho, s2 = coef[1], float(np.mean((col[1:] - x @ coef) ** 2))
        num += 4 * rho**2 * s2**2 / (1 - rho) ** 8
        den += s2**2 / (1 - rho) ** 4
    bandwidth = 1.3221 * (num / den * T) ** 0.2
    weights = _quadratic_spectral(np.arange(T) / bandwidth)
    psi = S_hac_simple(y, nlags=T - 1, weights_func=lambda _: weights) / T
    return psi * T / (T - k), float(bandwidth)


def _prewhiten(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Andrews-Monahan (1992) VAR(1) prewhitening: residuals and the recolouring matrix (I - A)^-1.

    The VAR(1) matrix's singular values are capped at 0.97, as in Ledoit and
    Wolf's code, so (I - A) stays invertible on a near-unit-root column.
    """
    coef, *_ = np.linalg.lstsq(y[:-1], y[1:], rcond=None)  # y_t = A y_{t-1} + e_t, A = coef.T
    u, d, vt = np.linalg.svd(coef.T)
    a = (u * np.minimum(d, 0.97)) @ vt
    return y[1:] - y[:-1] @ a.T, np.linalg.inv(np.eye(y.shape[1]) - a)


def sharpe_difference_test(
    candidate,
    baseline,
    *,
    prewhiten: bool = False,
    unit: str = "fraction",
    allow_nonfinite_drop: bool = False,
) -> dict:
    """Ledoit-Wolf (2008) HAC test of H0: SR(candidate) = SR(baseline) on two PAIRED return series.

    ``candidate`` and ``baseline`` are aligned per-period returns (the same
    dates in both; a row with a NaN/Inf in either is refused, or dropped from
    both with ``allow_nonfinite_drop=True``). ``delta`` is
    ``SR(candidate) - SR(baseline)``, each the per-period ``mean / std``
    (ddof=1), so a positive ``delta`` means the candidate's Sharpe is higher.
    Excess returns are the caller's: subtract the risk-free rate first.

    The standard error is the paper's HAC one (see the section comment): it
    stays valid under fat tails, volatility clustering and autocorrelation,
    where Jobson-Korkie/Memmel does not. ``prewhiten=False`` is the paper's
    HAC column, ``prewhiten=True`` its HACPW column. The paper finds HAC liberal in small samples (T = 120 rejects
    a true null at 5% about 7% of the time); its studentised block bootstrap
    is the remedy and is not implemented here.

    ``unit`` says what the returns are in: ``"fraction"`` (0.01 = 1%, the
    package's convention) or ``"percent"`` (1.0 = 1%). The Sharpes do not
    depend on it, but the paper's HAC does: Andrews' bandwidth weights the
    four moment columns by their own scale, and the prewhitening cap acts on
    the VAR's singular values, so the same series in percent and in fractions
    gets different p-values (the paper's hedge-fund pair: 14.7% in percent,
    17.9% in fractions). The paper works in percent, so the test always runs
    in percent: a ``"fraction"`` series is multiplied by 100 first, and the
    result matches the paper's computation on the same returns. ``unit`` pins
    the paper's convention; it does not make the HAC scale-free. Returns of a
    different size (a daily series is about a fifth of a monthly one) still
    get a different bandwidth than the paper's monthly data would, as in
    Andrews' unweighted rule itself. An AR(1) coefficient of 1 in a moment
    column gives a non-finite bandwidth and so ``se`` of ``None``.

    Returns full-precision floats: ``n_obs``, ``n_obs_raw``,
    ``n_nonfinite_dropped``, ``sharpe_candidate``, ``sharpe_baseline``,
    ``delta``, ``se``, ``zstat`` (delta / se), ``pvalue`` (two-sided, normal,
    the paper's), ``pvalue_onesided`` (H1: delta > 0, for a "beats" claim),
    ``bandwidth`` (the Andrews QS bandwidth, in periods), ``prewhiten`` and ``unit``. Refuses
    an unknown ``unit``,
    (:class:`InferenceInputError`) series of different lengths, fewer than
    6 paired observations (the T - 4 correction needs T > 4 and the
    prewhitened fit one more), and a series constant to floating-point noise
    (no Sharpe). A degenerate long-run variance yields ``None`` for ``se``,
    ``zstat`` and both p-values.
    """
    from scipy.stats import norm

    if unit not in _RETURN_UNITS:
        raise InferenceInputError(f"unit must be one of {tuple(_RETURN_UNITS)}, got {unit!r}")
    c = np.asarray(candidate, dtype=float).ravel()
    b = np.asarray(baseline, dtype=float).ravel()
    if c.size != b.size:
        raise InferenceInputError(
            f"candidate has {c.size} observations and baseline {b.size} — a paired test needs the "
            "same rows in both (align them on dates first)"
        )
    both, dropped = require_finite(np.column_stack([c, b]), allow_nonfinite_drop=allow_nonfinite_drop, what="paired")
    T = int(both.shape[0])
    if T < 6:
        raise InferenceInputError(f"need at least 6 paired finite observations, got {T}")
    for name, col in (("candidate", both[:, 0]), ("baseline", both[:, 1])):
        if flat(col, ddof=1):
            raise InferenceInputError(f"zero-variance {name} returns — its Sharpe ratio does not exist")

    sr_c, sr_b = (float(col.mean() / col.std(ddof=1)) for col in both.T)
    pct = both * _RETURN_UNITS[unit]  # the paper's unit: the HAC below is not scale-free
    data = np.column_stack([pct, pct**2])
    mu_c, mu_b, g_c, g_b = data.mean(axis=0)
    var_c, var_b = g_c - mu_c**2, g_b - mu_b**2  # population variances (the delta-method gradient's)
    grad = np.array([g_c / var_c**1.5, -g_b / var_b**1.5, -0.5 * mu_c / var_c**1.5, 0.5 * mu_b / var_b**1.5])
    y = data - data.mean(axis=0)
    if prewhiten:
        y_star, recolour = _prewhiten(y)
        psi_star, bandwidth = _lw_long_run_cov(y_star)
        psi = recolour @ psi_star @ recolour.T
    else:
        psi, bandwidth = _lw_long_run_cov(y)

    delta = sr_c - sr_b
    out = {
        "n_obs": T,
        "n_obs_raw": int(c.size),
        "n_nonfinite_dropped": dropped,
        "sharpe_candidate": sr_c,
        "sharpe_baseline": sr_b,
        "delta": delta,
        "se": None,
        "zstat": None,
        "pvalue": None,
        "pvalue_onesided": None,
        "bandwidth": bandwidth,
        "prewhiten": bool(prewhiten),
        "unit": unit,
    }
    var_delta = float(grad @ psi @ grad) / T
    scale = float(np.abs(grad) @ np.abs(psi) @ np.abs(grad)) / T  # zero is tested relatively (DEGENERATE_RTOL)
    if not (math.isfinite(var_delta) and var_delta > DEGENERATE_RTOL * scale):
        return out  # e.g. candidate == baseline: the gradient cancels the common moments
    se = math.sqrt(var_delta)
    z = delta / se
    return {**out, "se": se, "zstat": z, "pvalue": float(2.0 * norm.sf(abs(z))), "pvalue_onesided": float(norm.sf(z))}


# ── multiple testing ───────────────────────────────────────────────────────
#
# Error control over m tests run on one question (TOM-1646): family-wise
# (Bonferroni, Holm) or the false discovery rate under any dependence (BHY,
# TOM-3717). statsmodels' ``multipletests`` computes the corrections (adapter,
# not reimplementation); this module owns the refusals and the result shape.

MULTIPLE_TESTING_METHODS = ("bonferroni", "holm", "bhy")
_STATSMODELS_METHOD = {"bonferroni": "bonferroni", "holm": "holm", "bhy": "fdr_by"}


def _check_alpha(alpha) -> float:
    """Refuse a family-wise level outside the open interval (0, 1)."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating)) or not 0.0 < alpha < 1.0:
        raise InferenceInputError(f"alpha must be a family-wise level in the open interval (0, 1), got {alpha!r}")
    return float(alpha)


def bonferroni_alpha(n_tests: int, *, alpha: float = 0.05) -> float:
    """The level each of ``n_tests`` tests runs at for a family-wise ``alpha``: ``alpha / n_tests`` (Bonferroni).

    The one place this level is computed. A test with no p-value uses it
    directly: a two-sided bootstrap CI at ``1 - bonferroni_alpha(m)`` cuts at
    the ``100 * bonferroni_alpha(m) / 2`` percentile on each side. With
    p-values in hand, :func:`multiple_testing` applies it (and Holm).
    Refuses an ``n_tests`` that is not a whole number >= 1, and an ``alpha``
    outside (0, 1).
    """
    if isinstance(n_tests, bool) or not isinstance(n_tests, (int, np.integer)) or n_tests < 1:
        raise InferenceInputError(f"n_tests must be a whole number of tests >= 1, got {n_tests!r}")
    return _check_alpha(alpha) / int(n_tests)


@dataclass(frozen=True)
class MultipleTestingResult:
    """A family of p-values after a multiple-testing correction, in the INPUT order."""

    method: str  # "bonferroni", "holm" or "bhy"
    alpha: float  # the family-wise level (bhy: the false discovery rate)
    m: int  # tests in the family
    pvalues: np.ndarray  # the raw p-values, as passed
    adjusted: np.ndarray  # adjusted p-values, capped at 1; reject == (adjusted <= alpha)
    reject: np.ndarray  # bool: rejected at alpha (FWER for bonferroni/holm, FDR for bhy)
    per_test_alpha: float | None  # Bonferroni: alpha / m. Holm, BHY: None (the level depends on the rank)


def multiple_testing(pvalues, *, alpha: float = 0.05, method: str = "bonferroni") -> MultipleTestingResult:
    """Adjusted p-values and a reject mask for a family of ``m`` tests at family-wise level ``alpha``.

    ``method="bonferroni"`` rejects a test when ``p <= alpha / m``
    (:func:`bonferroni_alpha`); its adjusted p-value is ``min(1, m * p)``.
    ``method="holm"`` (Holm 1979, "A Simple Sequentially Rejective Multiple
    Test Procedure", Scand. J. Statist. 6(2)) steps down through the sorted
    p-values: the k-th smallest is tested against ``alpha / (m - k + 1)`` and
    the first one that fails stops the procedure. It controls the same
    family-wise error as Bonferroni and rejects at least as much. Its
    adjusted p-value is the running maximum of ``(m - k + 1) * p_(k)``,
    capped at 1. Neither assumes the tests are independent.

    ``method="bhy"`` (Benjamini and Yekutieli 2001, "The control of the false
    discovery rate in multiple testing under dependency", Ann. Statist. 29(4),
    Theorem 1.3) controls the FALSE DISCOVERY RATE at ``alpha`` (the expected
    share of rejections that are false), not the family-wise error: it steps
    up through the sorted p-values against ``k * alpha / (m * c(m))``, with
    ``c(m) = 1 + 1/2 + ... + 1/m``, which holds under any dependence between
    the tests. Its adjusted p-value is the running minimum, from the largest
    down, of ``m * c(m) * p_(k) / k``, capped at 1. It rejects more than Holm
    when many tests are true discoveries; it is the BHY of Harvey, Liu and
    Zhu (2016, "... and the Cross-Section of Expected Returns", RFS 29(1)).

    The contract is ``reject == (adjusted <= alpha)``, for every method. In
    exact arithmetic that is the rule above; in floating point a p-value
    within one rounding of its level (``p = 0.1/3`` at ``alpha=0.1``, ``m=3``)
    can fall on either side, and the mask follows the adjusted p-value.

    ``m`` is the size of the family passed. Every test tried on the question
    belongs in it, the ones that failed included; leaving one out loosens
    every other test's level. For the same reason a NaN/Inf p-value is
    refused, never dropped. Also refused: an empty or non-1-D family, a
    p-value outside [0, 1], an ``alpha`` outside (0, 1), an unknown method.
    """
    from statsmodels.stats.multitest import multipletests

    if method not in MULTIPLE_TESTING_METHODS:
        raise InferenceInputError(f"method must be one of {MULTIPLE_TESTING_METHODS}, got {method!r}")
    alpha = _check_alpha(alpha)
    raw = np.asarray(pvalues, dtype=float)
    if raw.ndim != 1:
        raise InferenceInputError(f"pvalues must be a 1-D family of p-values, got shape {raw.shape}")
    if raw.size == 0:
        raise InferenceInputError("pvalues must hold at least one p-value, got an empty family")
    try:
        p, _ = require_finite(raw, what="p-value")
    except InferenceInputError as e:
        # No drop option here: dropping a p-value would shrink m and loosen every other test's level.
        raise InferenceInputError(
            f"{e} multiple_testing takes no drop option: fix the test that produced the NaN, or pass "
            "1.0 for a test that ran and has no evidence."
        ) from e
    if ((p < 0.0) | (p > 1.0)).any():
        raise InferenceInputError(f"every p-value must lie in [0, 1], got {p[(p < 0.0) | (p > 1.0)].tolist()}")

    # The reject mask is read off the adjusted p-values, not taken from
    # statsmodels' own mask: the two are the same rule in exact arithmetic, and
    # one source keeps them from disagreeing in the last bit at the boundary.
    _, adjusted, _, _ = multipletests(p, alpha=alpha, method=_STATSMODELS_METHOD[method])
    adjusted = np.minimum(np.asarray(adjusted, dtype=float), 1.0)
    m = int(p.size)
    return MultipleTestingResult(
        method=method,
        alpha=alpha,
        m=m,
        pvalues=p,
        adjusted=adjusted,
        reject=adjusted <= alpha,
        per_test_alpha=bonferroni_alpha(m, alpha=alpha) if method == "bonferroni" else None,
    )


# ── resampling ─────────────────────────────────────────────────────────────


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One Politis-Romano (1994) stationary-bootstrap resample of ``range(n)``.

    Blocks start at a uniform random row, run forward CIRCULARLY (row n-1 is
    followed by row 0), and end after a geometric length with mean ``mean_block``:
    each step starts a new block with probability ``1 / mean_block``. The
    circular wrap is what makes the resample stationary and the bootstrap mean
    unbiased for the sample mean.
    """
    p = 1.0 / mean_block
    new = rng.random(n) < p
    new[0] = True
    starts = rng.integers(0, n, size=n)
    t = np.arange(n)
    block_begin = np.maximum.accumulate(np.where(new, t, 0))
    return (starts[block_begin] + (t - block_begin)) % n


def moving_block_indices(n: int, block: int, length: int, rng: np.random.Generator) -> np.ndarray:
    """``length`` row positions built from fixed-size blocks of ``range(n)`` (moving-block bootstrap).

    Each block starts at a uniform row in ``[0, n - block)`` and runs forward
    ``block`` rows without wrapping; blocks are drawn until ``length`` rows are
    covered and the last one is cut.
    """
    n_blocks = (length + block - 1) // block
    starts = [int(rng.integers(0, n - block)) for _ in range(n_blocks)]
    return np.concatenate([np.arange(s, s + block) for s in starts])[:length]


def bootstrap(
    statistic: Callable[..., float],
    *series: np.ndarray,
    draws: int,
    rng: np.random.Generator,
    mean_block: float | None = None,
) -> np.ndarray:
    """``statistic`` on ``draws`` resamples of ``series`` — the one resampling loop.

    Every series is resampled on the SAME rows (paired: common shocks stay
    matched), so they must have one length. ``mean_block=None`` is the iid
    bootstrap (rows drawn uniformly with replacement); a number is the
    stationary block bootstrap with that mean block length
    (:func:`stationary_bootstrap_indices`). Returns the ``draws`` values in draw
    order; an undefined statistic stays NaN for the caller to count.
    """
    n = len(series[0])
    values = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, size=n) if mean_block is None else stationary_bootstrap_indices(n, mean_block, rng)
        values[i] = statistic(*(s[idx] for s in series))
    return values


def gaussian_null(
    statistic: Callable[[np.ndarray], float], n: int, scale: float, *, draws: int, rng: np.random.Generator
) -> np.ndarray:
    """``statistic`` on ``draws`` zero-mean normal series of length ``n`` and std ``scale`` (a Monte-Carlo null)."""
    return np.array([statistic(rng.normal(0, scale, size=n)) for _ in range(draws)])


# ── regime split of the Sharpe ─────────────────────────────────────────────
#
# Where in the market the Sharpe was earned (TOM-3716). The idea, regimes cut
# at the bottom and top quantiles of the benchmark's period returns and the
# Sharpe split additively over them, is qis's regime classifier
# (ArturSepp/QuantInvestStrats, MIT); no qis code is copied and qis is not a
# dependency.

REGIMES = ("bear", "normal", "bull")
DEFAULT_REGIME_QUANTILES = (1 / 6, 5 / 6)


def _regime_thresholds(benchmark: np.ndarray, quantiles, thresholds) -> tuple[float, float]:
    """The (bear, bull) cut-offs: ``thresholds`` as given, else the full-sample ``quantiles`` of ``benchmark``."""
    pair = quantiles if thresholds is None else thresholds
    if isinstance(pair, str) or np.ndim(pair) != 1 or len(pair) != 2:
        raise InferenceInputError(f"quantiles and thresholds are (bear, bull) pairs, got {pair!r}")
    if thresholds is None:
        lo_q, hi_q = (float(q) for q in quantiles)
        if not 0.0 < lo_q < hi_q < 1.0:
            raise InferenceInputError(f"quantiles must satisfy 0 < bear < bull < 1, got {tuple(quantiles)!r}")
        lo, hi = (float(v) for v in np.quantile(benchmark, [lo_q, hi_q]))
    else:
        lo, hi = (float(v) for v in thresholds)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise InferenceInputError(
            f"the bear threshold must be below the bull threshold, got ({lo!r}, {hi!r}) — "
            "a benchmark this tied has no three regimes at these quantiles"
        )
    return lo, hi


def _regime_part(x: np.ndarray, mask: np.ndarray) -> float:
    """Regime ``mask``'s additive part of the per-period Sharpe of ``x``: ``sum(x[mask]) / (T * std(x, ddof=1))``."""
    return float(x[mask].sum() / (x.size * x.std(ddof=1)))


def regime_sharpe(
    returns,
    benchmark,
    *,
    quantiles=DEFAULT_REGIME_QUANTILES,
    thresholds=None,
    draws: int = 2000,
    seed: int = 0,
    allow_nonfinite_drop: bool = False,
) -> dict:
    """The per-period Sharpe of ``returns`` split ADDITIVELY over bear / normal / bull benchmark regimes.

    ``returns`` and ``benchmark`` are aligned per-period series (the same rows;
    a row with a NaN/Inf in either is refused, or dropped from both with
    ``allow_nonfinite_drop=True``). The period is the caller's: resample both
    (to months, quarters) before calling for regimes of that length. For a
    claim about ALPHA, pass active or excess returns, not the raw strategy.

    Regimes. Row t is ``"bear"`` when ``benchmark[t] <= bear threshold``,
    ``"bull"`` when ``benchmark[t] >= bull threshold``, ``"normal"`` otherwise:
    a row's label uses that row's benchmark return and nothing from any other
    row EXCEPT the thresholds. By default the thresholds are the FULL-SAMPLE
    ``quantiles`` of the benchmark (``np.quantile``, linear; 1/6 and 5/6, a
    sixth of the rows in each tail), so they see every period, future ones
    included: the split is an ex-post attribution, never a signal to trade on.
    Pass ``thresholds=(bear, bull)`` fixed in advance to use no full-sample
    information at all (``quantiles`` is then ignored).

    The split. With ``s`` the full-sample std (ddof=1) and ``T`` the rows,
    regime k's part is ``sum(returns[t in k]) / (T * s)`` = ``share_k * mean_k / s``,
    so the three parts sum to the full Sharpe ``mean / s`` (to float rounding).
    A part is the regime's contribution to the full-sample Sharpe, not the
    Sharpe of that regime alone (``sharpe`` in each regime's entry).

    The null, a Gaussian one PER REGIME. Each part is tested one-sided (H1:
    part > 0) against :func:`gaussian_null`: ``draws`` zero-mean normal series
    of length T whose rows in regime k have regime k's own sample std (the
    full-sample ``s`` for a regime of one row or a flat one), each split on
    the SAME labels and studentised by its own full-sample std, as the
    observed part is. Regime-dependent volatility (a wilder bear market) is
    so in the null; a single full-sample std would make a high-volatility
    regime's part look significant too often. ``pvalue`` is
    ``(1 + #{null part >= part}) / (1 + draws)``, never 0. Within a regime
    the null is iid normal: it ignores autocorrelation and fat tails, and so
    is liberal on a serially dependent or heavy-tailed series.

    Returns full-precision floats: ``n_obs``, ``n_obs_raw``,
    ``n_nonfinite_dropped``, ``sharpe`` (full, per period), ``thresholds``
    (bear, bull), ``threshold_source`` (``"full_sample_quantiles"`` or
    ``"given"``), ``quantiles`` (None when given), ``draws``, ``seed``, and
    ``regimes``: per name in :data:`REGIMES`, ``n_obs``, ``share``, ``mean``,
    ``std``, ``sharpe``, ``benchmark_mean``, ``sharpe_part``, ``null_std`` and
    ``pvalue``. An empty regime (given thresholds; or ``normal`` on a tied,
    discrete benchmark, whose tails then hold every row) has
    ``sharpe_part`` 0.0 and None for the rest; ``std`` and ``sharpe`` are None
    below two rows or on a flat regime. Refuses (:class:`InferenceInputError`)
    series of different lengths, fewer than 3 paired observations, flat
    returns, quantiles outside 0 < bear < bull < 1, thresholds not strictly
    increasing (each must be a pair), and ``draws`` below 2.
    """
    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or draws < 2:
        raise InferenceInputError(f"draws must be a whole number >= 2, got {draws!r}")
    r = np.asarray(returns, dtype=float).ravel()
    b = np.asarray(benchmark, dtype=float).ravel()
    if r.size != b.size:
        raise InferenceInputError(
            f"returns has {r.size} observations and benchmark {b.size} — the split needs the same rows "
            "in both (align them on dates first)"
        )
    both, dropped = require_finite(np.column_stack([r, b]), allow_nonfinite_drop=allow_nonfinite_drop, what="paired")
    r, b = np.ascontiguousarray(both[:, 0]), np.ascontiguousarray(both[:, 1])
    T = int(r.size)
    if T < 3:
        raise InferenceInputError(f"need at least 3 paired finite observations, got {T}")
    full = return_moments(r)  # refuses flat returns
    s = float(r.std(ddof=1))
    lo, hi = _regime_thresholds(b, quantiles, thresholds)
    masks = {"bear": b <= lo, "bull": b >= hi}
    masks["normal"] = ~(masks["bear"] | masks["bull"])

    sds = {k: None if m.sum() < 2 or flat(r[m], ddof=1) else float(r[m].std(ddof=1)) for k, m in masks.items()}
    sigma = np.empty(T)  # the null's per-row std: its regime's own, the full-sample one where that is undefined
    for k, m in masks.items():
        sigma[m] = s if sds[k] is None else sds[k]

    rng = np.random.default_rng(seed)
    regimes = {}
    for name in REGIMES:
        m = masks[name]
        n = int(m.sum())
        part = _regime_part(r, m)
        if n == 0:
            regimes[name] = {
                "n_obs": 0,
                "share": 0.0,
                "mean": None,
                "std": None,
                "sharpe": None,
                "benchmark_mean": None,
                "sharpe_part": 0.0,
                "null_std": None,
                "pvalue": None,
            }
            continue
        rk, sd = r[m], sds[name]
        null = gaussian_null(lambda z, m=m: _regime_part(z * sigma, m), T, 1.0, draws=draws, rng=rng)
        regimes[name] = {
            "n_obs": n,
            "share": n / T,
            "mean": float(rk.mean()),
            "std": sd,
            "sharpe": None if sd is None else float(rk.mean() / sd),
            "benchmark_mean": float(b[m].mean()),
            "sharpe_part": part,
            "null_std": float(null.std(ddof=1)),
            "pvalue": float((1 + np.count_nonzero(null >= part)) / (1 + draws)),
        }
    return {
        "n_obs": T,
        "n_obs_raw": T + dropped,
        "n_nonfinite_dropped": dropped,
        "sharpe": full.sr_period,
        "thresholds": (lo, hi),
        "threshold_source": "full_sample_quantiles" if thresholds is None else "given",
        "quantiles": None if thresholds is not None else tuple(float(q) for q in quantiles),
        "draws": int(draws),
        "seed": int(seed),
        "regimes": regimes,
    }


# ── drawdown episodes ──────────────────────────────────────────────────────


def largest_drawdown_episode(returns) -> dict | None:
    """The deepest drawdown episode of the compounded equity, as row positions.

    ``start`` is the first return row after the peak, ``trough`` the row at the
    bottom, ``end`` the row on which equity regains the peak — or the last row
    when it never does (``recovered`` False). ``depth`` is a negative fraction
    (the package's drawdown convention). ``None`` when equity never falls below
    a previous peak. The starting equity 1.0 is a peak, so a first-row loss
    opens an episode at row 0.
    """
    r = np.asarray(returns, dtype=float).ravel()
    eq = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    dd = compute_drawdown_series(eq)
    trough_e = int(np.argmin(dd))
    if dd[trough_e] >= 0:
        return None
    level = float(np.max(eq[: trough_e + 1]))  # the peak the trough is measured from
    peak_e = int(np.flatnonzero(eq[: trough_e + 1] >= level)[-1])
    after = np.flatnonzero(eq[trough_e + 1 :] >= level)
    recovered = bool(after.size)
    end_e = trough_e + 1 + int(after[0]) if recovered else eq.size - 1
    return {
        "start": peak_e,
        "trough": trough_e - 1,
        "end": end_e - 1,
        "depth": float(dd[trough_e]),
        "recovered": recovered,
        "n_removed": end_e - peak_e,
    }


__all__ = [
    "DEFAULT_REGIME_QUANTILES",
    "DEGENERATE_RTOL",
    "DSRResult",
    "EULER_MASCHERONI",
    "InferenceInputError",
    "LEWELLEN_RHO_BOUND",
    "MULTIPLE_TESTING_METHODS",
    "MultipleTestingResult",
    "REGIMES",
    "ReturnMoments",
    "bonferroni_alpha",
    "bootstrap",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_returns",
    "expected_max_sr",
    "factor_regression",
    "gaussian_null",
    "hac_ols",
    "largest_drawdown_episode",
    "mask_runs",
    "moments",
    "moving_block_indices",
    "multiple_testing",
    "newey_west_auto_lags",
    "newey_west_tstat",
    "predictive_regression",
    "regime_sharpe",
    "require_finite",
    "return_moments",
    "sharpe_difference_test",
    "sr_estimator_std",
]
