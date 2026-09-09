"""Deflated Sharpe Ratio (DSR) — the framework's single source of truth.

Bailey & López de Prado (2014), "The Deflated Sharpe Ratio", with the Mertens
(2002) correction for skew/kurtosis in the variance of the Sharpe estimator.

This module owns the DSR math for the whole ecosystem. Consuming repos
(quantbox-lab's ``dsr-gate.py``, the ``acceptance-gates`` skill, the pre-live
gauntlet) import from here rather than reimplementing the formulas. If you find
yourself about to write ``skew``/``kurtosis``/``expected max SR`` math anywhere
downstream, import from ``quantbox.analysis.dsr`` instead.

Provenance: ported verbatim (math unchanged) from quantbox-lab branch
``quark/fix-dsr-gate`` (``scripts/lib/dsr.py``), whose correctness was
confirmed by an independent cross-model review. That fix replaced a prior
version which tested ``t = sharpe * sqrt(n_years)`` against a Bonferroni
NORMAL null — silently assuming zero skew/kurtosis and under-deflating at small
trial counts.

All Sharpe/skew/kurtosis inputs and outputs here are in PER-PERIOD units
(daily, if your returns are daily) — annualisation is a display concern for the
caller, not part of the DSR test itself. The ``sqrt(periods)`` annualisation a
caller applies for display assumes returns are i.i.d. (no serial
autocorrelation); if the series is autocorrelated the annualised figure is
biased (use ``quantbox.analysis.hac`` for an autocorrelation-robust t-stat).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from scipy import stats

EULER_MASCHERONI = 0.5772156649015329

# Degeneracy is tested RELATIVELY -- never by exact float equality with zero.
#
# A quantity that is mathematically zero does not reliably come out as 0.0 in
# binary floating point. A constant returns series is the canonical example:
# `[0.001] * 200` accumulates rounding to std = 2.17e-19 while `[0.001] * 50`
# gives exactly 0.0, so whether an `== 0` guard fires is a lottery on the
# (value, length) pair rather than a property of the input. Measured on this
# module before the fix: of 32 constant series (8 values x 4 lengths), 20 hit
# the exact guard and 12 sailed past it into the moment path, where scipy hit
# catastrophic cancellation; those 12 were then refused -- by luck -- by two
# unrelated downstream checks, 8 by the skew/kurtosis finiteness test (NaN
# moments) and 4 by the Pearson-bound "impossible moments" test.
#
# The observed noise floor for a constant series is std/|value| ~ 2e-16
# (machine epsilon); 1e-12 leaves ~4000x headroom above it while staying far
# below any real series (std/scale = 1e-12 would imply a Sharpe of ~1e12).
# Being relative, the test is unit-independent: a genuinely tiny-but-real
# series (returns of order 1e-9 with std of order 1e-9) is unaffected. When
# every observation is exactly zero, scale is 0 and the test reduces to
# std <= 0, which still holds.
#
# This is the framework's single threshold for "cancelled to noise"; the
# validation plugin imports it rather than keeping a second copy.
DEGENERATE_RTOL = 1e-12


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

    Raises ValueError for n_trials <= 0 — there is no such thing as "zero or
    negative trials"; silently coercing this to 1 (as the old gate did)
    quietly removes the entire multiple-testing penalty.
    """
    if n_trials <= 0:
        raise ValueError(f"n_trials must be a positive integer, got {n_trials!r}")
    if n_trials == 1:
        return 0.0
    return (1 - EULER_MASCHERONI) * stats.norm.ppf(1 - 1.0 / n_trials) + EULER_MASCHERONI * stats.norm.ppf(
        1 - 1.0 / (n_trials * math.e)
    )


def sr_estimator_std(T: int, sr: float, skew: float, kurtosis: float) -> float:
    """Std of the Sharpe-ratio estimator (Mertens 2002 / Bailey-López de Prado), per-period units."""
    if T <= 1:
        raise ValueError(f"need at least 2 observations to estimate SR variance, got T={T!r}")
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
        raise ValueError(
            f"degenerate SR-estimator variance: the numerator ({numerator!r}) is negligible "
            f"against the scale of its own terms ({scale!r}) — the moments T={T}, sr={sr}, "
            f"skew={skew}, kurtosis={kurtosis} describe a distribution whose Sharpe estimator "
            "has no spread, so the DSR is undefined"
        )
    if numerator < 0:
        # Genuinely negative (not cancellation noise -- that was caught above):
        # pathological (garbage) skew/kurtosis/sr combinations. Fail loudly
        # rather than silently sqrt()-ing a negative number to NaN.
        raise ValueError(
            f"negative SR-estimator variance ({numerator / (T - 1)!r}) from T={T}, sr={sr}, "
            f"skew={skew}, kurtosis={kurtosis} — check inputs"
        )
    return math.sqrt(numerator / (T - 1))


def deflated_sharpe_ratio(sr: float, T: int, skew: float, kurtosis: float, n_trials: int) -> DSRResult:
    """Compute the genuine Deflated Sharpe Ratio.

    Parameters are all PER-PERIOD (not annualised): ``sr`` is the per-period
    Sharpe, ``skew``/``kurtosis`` are the per-period return-distribution
    moments, ``T`` is the number of return observations.

    Returns a DSRResult; ``result.dsr`` is P(true SR exceeds the
    multiple-testing-deflated benchmark) — compare against a threshold
    (e.g. 0.95) to decide pass/fail. This function does not itself apply a
    threshold — an acceptance gate is a policy decision made by the caller.
    """
    if not math.isfinite(sr):
        raise ValueError(f"sharpe must be finite, got {sr!r}")
    if not math.isfinite(skew) or not math.isfinite(kurtosis):
        raise ValueError(f"skew/kurtosis must be finite, got skew={skew!r} kurtosis={kurtosis!r}")
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
        raise ValueError(
            f"impossible moments: kurtosis={kurtosis!r} < skew**2 + 1 = {min_kurtosis!r} "
            f"(skew={skew!r}). Pearson (non-excess) kurtosis is always >= skew**2 + 1 for any "
            "real distribution. This usually means EXCESS kurtosis was passed instead of Pearson "
            "— scipy.stats.kurtosis() returns excess kurtosis by default (fisher=True); this "
            "function requires fisher=False (3.0 for a normal distribution, not 0.0)."
        )

    # No `sr_std == 0` check here any more. That was the second exact-float
    # equality guard; the check now lives inside sr_estimator_std, where the
    # terms of the variance sum exist to measure the cancellation against (see
    # DEGENERATE_RTOL). That function is guaranteed to return a strictly
    # positive value: the numerator's scale always includes the literal 1.0
    # term, so scale >= 1 and any accepted numerator therefore exceeds 1e-12.
    # Repeating the test here could never fire, and a second, weaker copy of a
    # guard is how the first one stops being the real protection.
    sr_std = sr_estimator_std(T, sr, skew, kurtosis)

    e_max = expected_max_sr(n_trials)
    sr0 = sr_std * e_max
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
    """Convenience wrapper: compute DSR directly from a per-period returns series.

    ``returns`` is any 1-D array-like of per-period returns (pandas Series /
    numpy array / list).

    Non-finite handling matches the framework's ``newey_west_tstat``
    convention exactly, by design (same invariant — "every gate fails loudly,
    never silently passes"):

    - By default (``allow_nonfinite_drop=False``), ANY NaN or ±Inf value in
      ``returns`` RAISES ``ValueError``. A corrupted/degenerate returns file
      must never silently pass on the surviving subset with no disclosure of
      how many observations vanished.
    - Pass ``allow_nonfinite_drop=True`` to explicitly opt into dropping
      non-finite observations and continuing. The returned ``DSRResult``
      then reports both ``n_obs_raw`` (the observation count before
      filtering) and ``n_nonfinite_dropped`` (how many were dropped) so the
      loss is always visible in the output, never silently absorbed into
      ``T``.

    Degenerate input (fewer than 2 finite observations, or a variance
    negligible relative to the scale of the series -- see ``DEGENERATE_RTOL``)
    still raises via the same paths as ``deflated_sharpe_ratio``.
    """
    import numpy as np

    raw = np.asarray(returns, dtype=float)
    finite_mask = np.isfinite(raw)
    n_dropped = int((~finite_mask).sum())
    if n_dropped and not allow_nonfinite_drop:
        raise ValueError(
            f"{n_dropped} of {raw.size} return observations are NaN/Inf — refusing to "
            "silently drop them (a corrupted returns file must not pass on the surviving "
            "subset). Pass allow_nonfinite_drop=True to explicitly opt into dropping them "
            "and continuing."
        )
    r = raw[finite_mask]
    T = len(r)
    if T <= 1:
        raise ValueError(f"need at least 2 finite return observations, got T={T!r}")
    std = float(r.std(ddof=1))
    # Relative, not `== 0` — see DEGENERATE_RTOL. `scale` is 0 only when every
    # observation is exactly zero, where this reduces to `std <= 0`.
    scale = float(np.mean(np.abs(r)))
    if std <= DEGENERATE_RTOL * scale:
        raise ValueError(
            f"zero-variance returns — cannot compute a Sharpe ratio: the standard deviation "
            f"({std!r}) is negligible against the scale of the series itself "
            f"(mean |return| = {scale!r}), i.e. the series is constant to within floating-point noise"
        )
    sr = r.mean() / std
    skew = float(stats.skew(r))
    kurtosis = float(stats.kurtosis(r, fisher=False))
    result = deflated_sharpe_ratio(sr=sr, T=T, skew=skew, kurtosis=kurtosis, n_trials=n_trials)
    return replace(result, n_obs_raw=int(raw.size), n_nonfinite_dropped=n_dropped)
