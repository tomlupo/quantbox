"""Tests for the Deflated Sharpe Ratio math in ``quantbox.analysis.dsr``.

Ported (math unchanged) from quantbox-lab's ``test_dsr_gate.py`` when the DSR
implementation moved into the framework. Guards the 2026-07-19 correctness fix:
the pre-fix gate computed ``t = sharpe * sqrt(n_years)`` against a
Bonferroni-corrected NORMAL null — a plain significance test, not DSR. The
load-bearing regression (``test_false_pass_band_*``) pins the specific false-PASS
band that old gate let through. Everything downstream (quantbox-lab's
``dsr-gate.py``, the acceptance-gates skill) imports this module, so this suite
is the single home for the DSR math regression.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy import stats

from quantbox.analysis import (
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_returns,
    expected_max_sr,
    sr_estimator_std,
)


def test_expected_max_sr_known_value_n20():
    """Bailey & Lopez de Prado (2014) exact form, N=20 -> ~1.9007."""
    assert expected_max_sr(20) == pytest.approx(1.9007079511811988, abs=1e-9)


def test_expected_max_sr_single_trial_is_zero():
    assert expected_max_sr(1) == 0.0


def test_expected_max_sr_rejects_nonpositive_trials():
    """No silent coercion to 1 — removing the penalty must be an explicit choice."""
    with pytest.raises(ValueError):
        expected_max_sr(0)
    with pytest.raises(ValueError):
        expected_max_sr(-5)


def test_dsr_normal_returns_matches_hand_derivation():
    """skew=0, kurtosis=3 (normal): sr_std = sqrt((1 + 0.5*sr^2)/(T-1))."""
    sr, T, n_trials = 0.1, 500, 10
    result = deflated_sharpe_ratio(sr=sr, T=T, skew=0.0, kurtosis=3.0, n_trials=n_trials)
    expected_sr_std = math.sqrt((1 - 0.0 * sr + (3.0 - 1) / 4 * sr**2) / (T - 1))
    assert result.sr_std == pytest.approx(expected_sr_std, rel=1e-9)
    expected_sr0 = expected_sr_std * expected_max_sr(n_trials)
    assert result.sr0_period == pytest.approx(expected_sr0, rel=1e-9)
    expected_z = (sr - expected_sr0) / expected_sr_std
    assert result.z == pytest.approx(expected_z, rel=1e-9)
    assert result.dsr == pytest.approx(stats.norm.cdf(expected_z), rel=1e-9)


def test_dsr_deflates_more_with_more_trials():
    kwargs = dict(sr=0.15, T=500, skew=-0.2, kurtosis=4.0)
    d1 = deflated_sharpe_ratio(n_trials=1, **kwargs)
    d20 = deflated_sharpe_ratio(n_trials=20, **kwargs)
    d100 = deflated_sharpe_ratio(n_trials=100, **kwargs)
    assert d1.sr0_period < d20.sr0_period < d100.sr0_period
    assert d1.dsr > d20.dsr > d100.dsr


def test_dsr_fat_tails_reduce_confidence_vs_normal():
    kwargs = dict(sr=0.2, T=500, n_trials=10)
    normal = deflated_sharpe_ratio(skew=0.0, kurtosis=3.0, **kwargs)
    fat_tailed = deflated_sharpe_ratio(skew=0.0, kurtosis=9.0, **kwargs)
    assert fat_tailed.sr_std > normal.sr_std
    assert fat_tailed.dsr < normal.dsr


@pytest.mark.parametrize("skew", [0.5, -0.5])
def test_dsr_nonzero_skew_matches_hand_derivation(skew):
    """Exact DSR at nonzero skew of BOTH signs, independently re-derived here — a
    sign error in the skew term would flip which sign gives the higher DSR."""
    sr, T, kurtosis, n_trials = 0.12, 400, 4.0, 8
    result = deflated_sharpe_ratio(sr=sr, T=T, skew=skew, kurtosis=kurtosis, n_trials=n_trials)

    expected_variance = (1 - skew * sr + (kurtosis - 1) / 4 * sr**2) / (T - 1)
    expected_sr_std = math.sqrt(expected_variance)
    expected_e_max = (1 - 0.5772156649015329) * stats.norm.ppf(
        1 - 1.0 / n_trials
    ) + 0.5772156649015329 * stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
    expected_sr0 = expected_sr_std * expected_e_max
    expected_z = (sr - expected_sr0) / expected_sr_std
    expected_dsr = stats.norm.cdf(expected_z)

    assert result.sr_std == pytest.approx(expected_sr_std, rel=1e-9)
    assert result.dsr == pytest.approx(expected_dsr, rel=1e-9)


def test_dsr_positive_vs_negative_skew_give_different_dsr():
    kwargs = dict(sr=0.12, T=400, kurtosis=4.0, n_trials=8)
    pos = deflated_sharpe_ratio(skew=0.5, **kwargs)
    neg = deflated_sharpe_ratio(skew=-0.5, **kwargs)
    assert pos.dsr != pytest.approx(neg.dsr, rel=1e-6)


# --- Impossible Pearson moments (kurtosis < skew**2 + 1) must be rejected ---


def test_impossible_moments_rejected():
    with pytest.raises(ValueError, match="impossible moments"):
        deflated_sharpe_ratio(sr=0.1, T=500, skew=2.0, kurtosis=1.0, n_trials=10)


def test_zero_skew_zero_kurtosis_now_rejected():
    """kurtosis=0 with skew=0 (excess-passed-as-Pearson bug) must be rejected."""
    with pytest.raises(ValueError, match="impossible moments"):
        deflated_sharpe_ratio(sr=0.1, T=500, skew=0.0, kurtosis=0.0, n_trials=10)


def test_boundary_moments_at_equality_are_accepted():
    """kurtosis == skew**2 + 1 exactly must NOT be rejected by a strict '<' off-by-one."""
    skew = 1.0
    result = deflated_sharpe_ratio(sr=0.1, T=500, skew=skew, kurtosis=skew**2 + 1, n_trials=10)
    assert math.isfinite(result.dsr)


def test_dsr_from_returns_matches_direct_computation():
    rng = np.random.default_rng(42)
    r = rng.standard_t(df=5, size=800) * 0.01 + 0.0003
    from_returns = deflated_sharpe_ratio_from_returns(r, n_trials=15)
    sr = r.mean() / r.std(ddof=1)
    sk = float(stats.skew(r))
    ku = float(stats.kurtosis(r, fisher=False))
    direct = deflated_sharpe_ratio(sr=sr, T=len(r), skew=sk, kurtosis=ku, n_trials=15)
    assert from_returns.dsr == pytest.approx(direct.dsr, rel=1e-9)


def test_from_returns_reports_n_obs_raw_matches_survivors_when_clean():
    rng = np.random.default_rng(7)
    r = rng.standard_t(df=5, size=300) * 0.01 + 0.0003
    result = deflated_sharpe_ratio_from_returns(r, n_trials=10)
    assert result.n_obs_raw == len(r)
    assert result.n_nonfinite_dropped == 0
    assert len(r) == result.T


# --- NaN/Inf handling: fail loudly by default, opt-in must disclose counts ---


def test_nan_returns_rejected_by_default():
    rng = np.random.default_rng(1)
    r = rng.standard_t(df=5, size=200) * 0.01 + 0.0003
    r[5] = np.nan
    r[17] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        deflated_sharpe_ratio_from_returns(r, n_trials=10)


def test_inf_returns_rejected_by_default():
    rng = np.random.default_rng(2)
    r = rng.standard_t(df=5, size=200) * 0.01 + 0.0003
    r[3] = np.inf
    r[40] = -np.inf
    with pytest.raises(ValueError, match="NaN/Inf"):
        deflated_sharpe_ratio_from_returns(r, n_trials=10)


def test_nonfinite_drop_opt_in_reports_raw_and_surviving_counts():
    rng = np.random.default_rng(4)
    r = rng.standard_t(df=5, size=500) * 0.01 + 0.0003
    n_bad = 37
    bad_idx = rng.choice(len(r), size=n_bad, replace=False)
    for i, idx in enumerate(bad_idx):
        r[idx] = np.nan if i % 2 == 0 else np.inf

    result = deflated_sharpe_ratio_from_returns(r, n_trials=10, allow_nonfinite_drop=True)
    assert result.n_obs_raw == len(r)
    assert result.n_nonfinite_dropped == n_bad
    assert len(r) - n_bad == result.T

    finite = r[np.isfinite(r)]
    sr = finite.mean() / finite.std(ddof=1)
    sk = float(stats.skew(finite))
    ku = float(stats.kurtosis(finite, fisher=False))
    direct = deflated_sharpe_ratio(sr=sr, T=len(finite), skew=sk, kurtosis=ku, n_trials=10)
    assert result.dsr == pytest.approx(direct.dsr, rel=1e-9)


def test_nonfinite_drop_opt_in_still_raises_if_too_few_survive():
    r = np.array([0.01, np.nan, np.nan, np.nan])
    with pytest.raises(ValueError):
        deflated_sharpe_ratio_from_returns(r, n_trials=10, allow_nonfinite_drop=True)


def test_all_finite_input_unaffected_by_allow_nonfinite_drop_flag():
    rng = np.random.default_rng(5)
    r = rng.standard_t(df=5, size=300) * 0.01 + 0.0003
    default_result = deflated_sharpe_ratio_from_returns(r, n_trials=10)
    opted_in_result = deflated_sharpe_ratio_from_returns(r, n_trials=10, allow_nonfinite_drop=True)
    assert default_result.dsr == pytest.approx(opted_in_result.dsr, rel=1e-12)
    assert default_result.n_nonfinite_dropped == opted_in_result.n_nonfinite_dropped == 0


# --- Degenerate-input holes must FAIL LOUDLY ---


def test_infinite_sharpe_rejected():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(sr=math.inf, T=500, skew=0.0, kurtosis=3.0, n_trials=10)


def test_nonpositive_n_trials_rejected_not_coerced():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(sr=0.1, T=500, skew=0.0, kurtosis=3.0, n_trials=0)


def test_too_few_observations_rejected():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(sr=0.1, T=1, skew=0.0, kurtosis=3.0, n_trials=10)


# --- degeneracy is detected RELATIVELY, not by exact float equality ---
#
# The guard this replaced was `if std == 0`, and the only test aiming at it
# passed `np.zeros(100)` with a bare `pytest.raises(ValueError)`. Both halves of
# that were blind: zeros is the one constant that IS exactly representable, and
# an unmatched ValueError accepts a refusal that came from somewhere else. Under
# the old guard 15 of the 32 (value, length) pairs below slipped past it and were
# refused only by an unrelated downstream finiteness check on NaN moments — the
# protection was not where the code said it was. Hence: many constants, and the
# message pinned.


@pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 0.001, 0.1, 1e-8, -0.02, 3.7])
@pytest.mark.parametrize("length", [3, 50, 200, 1000])
def test_constant_returns_rejected_regardless_of_representability(value, length):
    """Every constant series is degenerate, whatever the constant or the length.

    Whether ``std`` lands on exactly 0.0 or on ~1e-19 is an artefact of binary
    floating point (e.g. ``[0.001] * 50`` gives 0.0, ``[0.001] * 200`` gives
    2.17e-19). The refusal must not depend on it, and must come from the
    zero-variance guard rather than from a downstream NaN check.
    """
    returns = np.full(length, value, dtype=float)
    with pytest.raises(ValueError, match="zero-variance returns"):
        deflated_sharpe_ratio_from_returns(returns, n_trials=10)


def test_constant_returns_do_not_reach_the_moment_computation():
    """The guard fires BEFORE scipy sees near-identical data.

    scipy emits a catastrophic-cancellation RuntimeWarning when asked for the
    moments of a near-constant series. Its absence proves the refusal happened at
    the variance guard, not downstream of it -- which is the actual defect: the
    old code refused these inputs for the wrong reason.
    """
    returns = np.full(200, 0.001, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="zero-variance returns"):
            deflated_sharpe_ratio_from_returns(returns, n_trials=10)


@pytest.mark.parametrize("factor", [1e-12, 1e-6, 1.0, 1e6, 1e12])
def test_degeneracy_guard_is_scale_invariant(factor):
    """Rescaling a genuinely real series must not change the DSR at all.

    This is what separates a RELATIVE guard from an absolute one: an absolute
    threshold (``std <= 1e-12``) would pass the unscaled series and silently
    start refusing -- or, worse, keep accepting while meaning something else --
    as the units shrink. Bit-identical is the assertion; anything weaker (e.g.
    "a tiny series still works") is satisfied by an absolute threshold too.
    """
    rng = np.random.default_rng(20260909)
    base = rng.standard_normal(500) * 0.01 + 0.0005
    reference = deflated_sharpe_ratio_from_returns(base, n_trials=50)
    scaled = deflated_sharpe_ratio_from_returns(base * factor, n_trials=50)
    # The DSR itself is bit-identical; the intermediates differ by a couple of
    # ulp because mean/std of a rescaled array are not bit-exact reproductions.
    assert scaled.dsr == reference.dsr
    assert scaled.sr_period == pytest.approx(reference.sr_period, rel=1e-12)
    assert scaled.z == pytest.approx(reference.z, rel=1e-12)


def test_a_real_series_with_tiny_absolute_std_is_accepted():
    """Positive control: small in absolute terms is not the same as degenerate."""
    rng = np.random.default_rng(7)
    returns = rng.standard_normal(500) * 1e-15 + 1e-16
    result = deflated_sharpe_ratio_from_returns(returns, n_trials=10)
    assert math.isfinite(result.dsr)


# --- the second exact-equality guard: the SR-estimator variance ---


def test_sr_estimator_variance_degeneracy_is_reachable_and_refused():
    """kurtosis == skew**2 + 1 with skew*sr == 2 drives the variance to zero.

    Under the Pearson bound the numerator is bounded below by
    ``(1 - skew*sr/2)**2``, so this knife edge is the ONLY way it reaches zero --
    a two-point distribution whose Sharpe estimator has no spread.
    """
    with pytest.raises(ValueError, match="degenerate SR-estimator variance"):
        sr_estimator_std(T=500, sr=1.0, skew=2.0, kurtosis=5.0)


@pytest.mark.parametrize("sr", [1.0 - 1e-6, 1.0 - 1e-9, 1.0, 1.0 + 1e-9, 1.0 + 1e-6])
def test_neighbourhood_of_the_knife_edge_is_refused_not_just_the_exact_point(sr):
    """Near the edge the three-term sum has lost every significant digit.

    An ``== 0`` test refuses only where rounding happens to land on 0.0; at
    sr = 1 +/- 1e-6 the numerator is ~1e-12 and the old guard passed it through,
    reporting dsr = 1.0 from a division by noise.
    """
    with pytest.raises(ValueError, match="degenerate SR-estimator variance"):
        deflated_sharpe_ratio(sr=sr, T=500, skew=2.0, kurtosis=5.0, n_trials=100)


def test_small_sr_estimator_std_from_a_long_series_is_not_degenerate():
    """Positive control, and the reason this guard must NOT be relative to 1/sqrt(T).

    ``sr_std`` falls as ``1/sqrt(T-1)``, so any long series has a small one. Only
    cancellation among the numerator's own terms means degeneracy.
    """
    std = sr_estimator_std(T=1_000_000, sr=0.05, skew=0.0, kurtosis=3.0)
    assert 0 < std < 1e-2


# --- THE REGRESSION: the verified false-PASS band, pinned as FAIL under real DSR ---


@pytest.mark.parametrize(
    "z_naive,expected_dsr",
    [
        (3.1, 0.8800),
        (3.5, 0.9413),
    ],
)
def test_false_pass_band_now_fails_under_real_dsr(z_naive, expected_dsr):
    """With the OLD gate's naive normal-null z (z = sharpe_ann * sqrt(n_years)) in
    [3.1, 3.6] at N=20 trials, the old gate PASSED while the true DSR sits at
    0.88-0.95 — BELOW a 0.95 bar. Reproduce it via 1 year of daily data
    (T=365, periods=365) so sr_annualized == z_naive, skew=0/kurtosis=3."""
    T = 365
    sr_period = z_naive / math.sqrt(T)
    result = deflated_sharpe_ratio(sr=sr_period, T=T, skew=0.0, kurtosis=3.0, n_trials=20)

    old_style_pvalue = 2 * (1 - stats.norm.cdf(abs(z_naive)))
    old_style_alpha_adj = 0.05 / 20
    old_gate_would_pass = bool(z_naive > 0 and old_style_pvalue < old_style_alpha_adj)
    assert old_gate_would_pass is True

    assert result.dsr == pytest.approx(expected_dsr, abs=5e-3)
    assert result.dsr < 0.95
