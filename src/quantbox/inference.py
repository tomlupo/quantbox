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
  :func:`newey_west_auto_lags`, :func:`factor_regression`.
- Deflated Sharpe Ratio: :func:`deflated_sharpe_ratio`,
  :func:`deflated_sharpe_ratio_from_returns`, :func:`expected_max_sr`,
  :func:`sr_estimator_std`, :class:`DSRResult`.
- Resampling: :func:`bootstrap` (iid or paired stationary block),
  :func:`stationary_bootstrap_indices`, :func:`moving_block_indices`,
  :func:`gaussian_null`.
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


def hac_ols(y: np.ndarray, x: np.ndarray, lags: int) -> Any:
    """OLS of ``y`` on ``x`` with a Newey-West (Bartlett kernel) HAC covariance — the ONE HAC fit.

    statsmodels computes the sandwich (adapter, not reimplementation). The
    ``nobs/(nobs-k)`` small-sample correction is OFF: the retired hand-rolled
    estimators (and robo-lab's ``_ols_nw``) used the uncorrected estimator, and
    ``tests/test_hac_parity.py`` pins that parity. With it on, t-stats shrink by
    exactly ``sqrt(nobs/(nobs-k))`` (~0.1% for a mean t-stat, ~0.4% for a
    3-factor alpha); adopting it is a policy decision, not a side effect.
    Returns the statsmodels result.
    """
    import statsmodels.api as sm

    return sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False})


def newey_west_tstat(returns, lags: int | None = None, *, allow_nonfinite_drop: bool = False) -> dict:
    """Newey-West HAC t-stat on the mean of ``returns`` — the ``nw`` gate calls this one.

    An OLS of ``returns`` on a constant with a HAC (Bartlett kernel)
    covariance (:func:`hac_ols`). Corrects the standard error of the mean for
    the serial correlation that inflates a naive t-stat on overlapping or
    trend-following returns. ``lags=None`` picks :func:`newey_west_auto_lags`.

    Non-finite observations RAISE by default (see :func:`require_finite`);
    ``allow_nonfinite_drop=True`` opts into dropping them, and the returned
    ``n_obs_raw`` / ``n_nonfinite_dropped`` keep that loss visible.

    Returns a dict with full-precision floats: ``n_obs``, ``n_obs_raw``,
    ``n_nonfinite_dropped``, ``nw_lags``, ``mean_return``, ``nw_se``,
    ``nw_tstat`` and ``nw_pvalue`` (two-sided, normal). A degenerate series
    (fewer than 2 finite observations, or zero long-run variance) yields
    ``None`` for the SE/t-stat/p-value rather than ``inf``/``nan``.
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

    # Zero-variance series: statsmodels would divide by a zero SE. Guard first,
    # RELATIVELY (see DEGENERATE_RTOL): a constant series does not reliably
    # come out with std == 0.0 — `[0.001] * 337` gives ~2e-19, which statsmodels
    # turned into a t-stat of 3.5e16 (29 of 48 constant series probed, TOM-1351).
    if flat(r):
        return base

    fit = hac_ols(r, np.ones((n, 1)), lags)
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


def factor_regression(y, factors, factor_names: list[str], lags: int | None = None) -> dict:
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
    "DEGENERATE_RTOL",
    "DSRResult",
    "EULER_MASCHERONI",
    "InferenceInputError",
    "ReturnMoments",
    "bootstrap",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_returns",
    "expected_max_sr",
    "factor_regression",
    "gaussian_null",
    "hac_ols",
    "largest_drawdown_episode",
    "moments",
    "moving_block_indices",
    "newey_west_auto_lags",
    "newey_west_tstat",
    "require_finite",
    "return_moments",
    "sr_estimator_std",
]
