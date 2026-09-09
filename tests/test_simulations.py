"""Basic tests for ``quantbox.features.simulations``.

Covers a happy-path call for ``parametric_mc`` (shape + determinism via
seeded Generator), plus the frequency constants export.

``simulations_stats`` is exercised by the host project (robo); the input
panel requires a specific (date, ticker, step, sim_no) layout built by
robo's sim driver, so a standalone quantbox-side integration test isn't
meaningful — it is a straight data-transform whose correctness rides on
the host.

That coverage used to be described here as robo's "byte-identical
end-to-end gate". This branch changes the random stream for robo's only
configuration (student-t, float32), so that gate cannot be byte-identical
across this change and has to be re-captured on the host side before it
means anything again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from quantbox.features import (
    FREQ_TO_PERIODS,
    parametric_mc,
    simulations_stats,
)
from quantbox.features.simulations import _draw_uncorrelated


def test_freq_to_periods_public() -> None:
    # Smoke: the legacy constant is re-exported and has the expected keys.
    assert FREQ_TO_PERIODS["B"] == 252
    assert FREQ_TO_PERIODS["D"] == 365
    assert set(FREQ_TO_PERIODS) >= {"B", "D", "W", "M", "Q", "Y"}


def test_simulations_stats_importable() -> None:
    # Smoke: the symbol is wired; full integration in host.
    assert callable(simulations_stats)


@pytest.fixture
def toy_prices() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    tickers = ["AAA", "BBB", "CCC"]
    rets = rng.standard_normal((len(dates), len(tickers))) * 0.01
    return pd.DataFrame(100 * np.exp(rets.cumsum(axis=0)), index=dates, columns=tickers)


def test_parametric_mc_shape_and_seeding(toy_prices: pd.DataFrame) -> None:
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    sim_a = parametric_mc(prices=toy_prices, iterations=50, steps=21, distribution="normal", seed=rng_a)
    sim_b = parametric_mc(prices=toy_prices, iterations=50, steps=21, distribution="normal", seed=rng_b)
    # Shape: (steps, tickers * iterations)
    assert sim_a.shape == (21, len(toy_prices.columns) * 50)
    assert list(sim_a.index.names) == ["step"]
    assert sim_a.columns.names[0] == "ticker"
    # Same seed → bit-identical output
    pd.testing.assert_frame_equal(sim_a, sim_b)


def test_parametric_mc_student_t_shape(toy_prices: pd.DataFrame) -> None:
    rng = np.random.default_rng(7)
    sim = parametric_mc(
        prices=toy_prices,
        iterations=25,
        steps=30,
        distribution="student-t",
        df=5,
        seed=rng,
    )
    assert sim.shape == (30, len(toy_prices.columns) * 25)


@pytest.fixture
def toy_params() -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(0)
    tickers = ["AAA", "BBB", "CCC"]
    mu = pd.Series(rng.uniform(0.02, 0.09, len(tickers)), index=tickers)
    a = rng.normal(size=(len(tickers), len(tickers)))
    cov = pd.DataFrame(a @ a.T / 50, index=tickers, columns=tickers)
    return mu, cov


def test_parametric_mc_does_not_mutate_caller_parameters(
    toy_params: tuple[pd.Series, pd.DataFrame],
) -> None:
    # Regression, 2026-09-09: `mu /= step_frequency` divided the CALLER's
    # Series in place, so simply calling this function rescaled parameters
    # the caller still held.
    mu, cov = toy_params
    mu_before = mu.copy()
    cov_before = cov.copy()

    parametric_mc(mu=mu, cov=cov, iterations=20, steps=10, seed=np.random.default_rng(1))

    pd.testing.assert_series_equal(mu, mu_before)
    pd.testing.assert_frame_equal(cov, cov_before)


def test_parametric_mc_reusing_parameters_is_repeatable(
    toy_params: tuple[pd.Series, pd.DataFrame],
) -> None:
    # The symptom that mutation produced, and the reason it is worth a
    # test of its own: two identically-seeded calls sharing parameter
    # objects returned different panels, because the second ran on values
    # already divided by step_frequency once.
    mu, cov = toy_params
    kw = dict(mu=mu, cov=cov, iterations=20, steps=10)

    first = parametric_mc(seed=np.random.default_rng(1), **kw)
    second = parametric_mc(seed=np.random.default_rng(1), **kw)

    pd.testing.assert_frame_equal(first, second)


# Bit-identity against the pre-refactor expressions — RE-SCOPED by this
# branch. Read the scope before trusting the name.
#
# On `dev` this test asserted bit-identity for every
# (correlated, distribution, precision) cell. This branch draws the shocks
# from `numpy.random.Generator` instead of `scipy.stats`, which changes the
# random stream ON PURPOSE, so six of those eight cells cannot hold. They
# are REMOVED rather than loosened, because a loosened version would be
# asserting a statistical resemblance under a name that promises identity:
#
#   * float32 — the draw is now native at float32 rather than float64
#     downcast, so the values differ from the first element on.
#   * student-t — assembled as `Z / sqrt(X / df)` from `standard_normal`
#     and `standard_gamma`, which consumes the bit stream differently from
#     `scipy.stats.t.rvs`.
#
# What survives is the (normal, float64) cell, and it is not a leftover.
# `scipy.stats.norm.rvs(random_state=<Generator>)` delegates to that same
# Generator's `standard_normal` at float64, so on this ONE path the new
# code must still be exactly the old code — and it is, which is worth
# pinning: it keeps the refactor's allocation changes (`out=`, the
# broadcast, the in-place `exp`) under an exact, same-machine comparison
# across both the correlated and the uncorrelated branch. Had this cell
# moved too, the stream change would have been wider than intended.
#
# The cells that had to go are covered instead by the regenerated stored
# reference below and by the distribution tests on `_draw_uncorrelated`.


def _reference_parametric_mc(
    mu: pd.Series,
    cov: pd.DataFrame,
    iterations: int,
    steps: int,
    correlated: bool,
    seed: np.random.Generator,
) -> np.ndarray:
    """The panel `parametric_mc` produced BEFORE this branch's refactor.

    Transcribed expression for expression from
    `src/quantbox/features/simulations.py` at 58e5e09, covering the
    `mu`/`cov`-supplied, normal, float64 call path only, and returning the
    raw `returns_sim` array from just before the reshape (which this branch
    does not touch). Deliberately dumb: its job is to be recognisably the
    old lines, not to be good code.

    The float32 and student-t branches of the original are deliberately NOT
    transcribed — this branch changes the random stream on both, so a
    comparison there would be asserting something false.
    """
    frequency = 252
    dtype = np.float64
    index = mu.index
    var = pd.Series(np.diag(cov), index=cov.index).loc[index]
    cov = cov.loc[index, index]

    step_frequency = frequency
    mu = mu / step_frequency
    cov = cov / step_frequency
    var = var / step_frequency

    drift = mu - 0.5 * var
    drift = drift.astype(dtype)
    if correlated:
        cov = cov.astype(dtype)
        chol = np.linalg.cholesky(cov)
    else:
        var = var.astype(dtype)

    uncorr_x = stats.norm.rvs(size=(len(mu), iterations * steps), random_state=seed).astype(dtype)

    if correlated:
        shock = np.dot(chol, uncorr_x).astype(dtype)
    else:
        shock = (uncorr_x * np.tile(np.atleast_2d(np.sqrt(var)).T, uncorr_x.shape[1])).astype(dtype)

    return np.exp(np.atleast_2d(drift).T + shock).astype(dtype) - 1


@pytest.mark.parametrize("correlated", [True, False])
def test_parametric_mc_matches_reference_implementation(correlated: bool) -> None:
    tickers = ["AAA", "BBB"]
    mu = pd.Series([0.05, 0.07], index=tickers)
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=tickers, columns=tickers)
    steps, iterations = 5, 7

    got = parametric_mc(
        mu=mu,
        cov=cov,
        seed=np.random.default_rng(2026),
        iterations=iterations,
        steps=steps,
        correlated=correlated,
        distribution="normal",
        precision="float64",
    ).to_numpy()

    reference = _reference_parametric_mc(
        mu=mu,
        cov=cov,
        seed=np.random.default_rng(2026),
        iterations=iterations,
        steps=steps,
        correlated=correlated,
    )
    # The reshape/concat the function applies to `returns_sim`, unchanged
    # by this branch and reproduced here so the comparison is on panels.
    want = np.hstack([reference[i].reshape(steps, iterations) for i in range(len(tickers))])

    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, want)


# REGENERATED on this branch, deliberately — the values below are NOT the
# ones `dev` carries, and that is the point rather than an accident.
#
# Drawing the shocks from `numpy.random.Generator` at the panel's own dtype
# instead of drawing float64 through `scipy.stats` and casting produces a
# DIFFERENT valid sample from the same distributions. Three of the four
# cells therefore had to be re-recorded.
#
# The fourth did not, and it is the useful control: ("float64", "normal")
# below is byte-for-byte the value `dev` stores, because
# `scipy.stats.norm.rvs(random_state=<Generator>)` delegates to that
# Generator's own `standard_normal` at float64. If a future change to the
# draw moved THAT cell too, it would be changing more than this branch
# claims to.
#
# Kept tiny and fully written out: a reference you can read is one you can
# reason about when it fails.
GOLDEN = {
    ("float64", "normal"): [
        [
            -0.00982477476270327,
            0.0031549272392272787,
            -0.023492080580430397,
            -0.00817810303398303,
            0.006539864605085377,
            -0.010802445446907338,
        ],
        [
            0.01786176510762849,
            0.008194195606294752,
            -0.003554074116662531,
            0.00028595179706170093,
            0.015648492582863494,
            0.008808880306370392,
        ],
    ],
    ("float64", "student-t"): [
        [
            -0.011489620607138429,
            0.003320298124244392,
            -0.032419219678561095,
            -0.007815025444610546,
            0.010271301125419852,
            -0.019548932413441134,
        ],
        [
            0.02155567453374929,
            0.011768119006467348,
            -0.004535207944862507,
            -0.003934026005773306,
            0.022941635514055037,
            0.0066686402786255705,
        ],
    ],
    ("float32", "normal"): [
        [-0.019417644, 0.0009651184, 0.0007904768, -0.0026753545, -0.008721948, 0.009429216],
        [-0.010847867, -0.011642039, 0.014804721, 0.011250019, -0.016670048, -0.0151949525],
    ],
    ("float32", "student-t"): [
        [-0.01148963, 0.0033203363, -0.032419264, -0.007815063, 0.010271311, -0.019548953],
        [0.021555662, 0.011768103, -0.0045351386, -0.0039340854, 0.02294159, 0.006668687],
    ],
}

# Why this one is `assert_allclose` and not `assert_array_equal`: as an
# exact assertion it went red on GitHub's runners while passing on every
# dev box, by 1 ULP (2.2e-16) in two of twelve float64 elements. `np.dot`
# here goes to BLAS, which selects its kernel from the CPU, so the last bit
# of a float64 product is a property of the machine, not of this code.
#
# So this test is a DRIFT ALARM, not the identity proof: the tolerances sit
# orders of magnitude above one ULP and orders of magnitude below any drift
# worth hearing about. What it now guards, since the draw is ours rather
# than scipy's, is a change in numpy's `standard_normal` / `standard_gamma`
# sampling internals — `pyproject.toml` bounds numpy only from below — and
# any accidental change to the block loop or the t construction. The exact
# claim, on the one path where identity still holds, is pinned by
# `test_parametric_mc_matches_reference_implementation` above.
_GOLDEN_RTOL = {"float64": 1e-12, "float32": 1e-5}


@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("distribution", ["normal", "student-t"])
def test_parametric_mc_matches_stored_reference(distribution: str, precision: str) -> None:
    tickers = ["AAA", "BBB"]
    mu = pd.Series([0.05, 0.07], index=tickers)
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=tickers, columns=tickers)

    sim = parametric_mc(
        mu=mu,
        cov=cov,
        iterations=3,
        steps=2,
        distribution=distribution,
        precision=precision,
        df=3,
        seed=np.random.default_rng(2026),
    )

    got = sim.to_numpy()
    assert got.dtype == getattr(np, precision)
    want = np.array(GOLDEN[(precision, distribution)], dtype=got.dtype)
    np.testing.assert_allclose(got, want, rtol=_GOLDEN_RTOL[precision], atol=0)


# --- _draw_uncorrelated -------------------------------------------------


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("distribution", ["normal", "student-t"])
def test_draw_uncorrelated_shape_and_dtype(distribution: str, precision: str) -> None:
    dtype = getattr(np, precision)
    out = _draw_uncorrelated((3, 5000), distribution, 3, dtype, np.random.default_rng(0))
    assert out.shape == (3, 5000)
    assert out.dtype == dtype


def test_draw_uncorrelated_is_seed_deterministic() -> None:
    kw = ((2, 10_000), "student-t", 3, np.float32)
    a = _draw_uncorrelated(*kw, np.random.default_rng(11))
    b = _draw_uncorrelated(*kw, np.random.default_rng(11))
    assert np.array_equal(a, b)


def test_draw_uncorrelated_blocking_does_not_change_the_distribution() -> None:
    # Same draw split into many blocks vs one: different stream, so the
    # quantiles must agree without being equal.
    size = (2, 200_000)
    one = _draw_uncorrelated(size, "normal", 3, np.float64, np.random.default_rng(3))
    many = _draw_uncorrelated(size, "normal", 3, np.float64, np.random.default_rng(4), target_bytes=64 * 1024)
    qs = [0.01, 0.25, 0.5, 0.75, 0.99]
    np.testing.assert_allclose(np.quantile(one, qs), np.quantile(many, qs), atol=0.03, rtol=0)


def test_draw_uncorrelated_writes_every_column_across_many_blocks() -> None:
    # The block loop's bookkeeping, tested by something that can actually
    # fail on it. `target_bytes=64KiB` at 2 assets gives 4096-column blocks,
    # so 49 of them over 200k columns.
    #
    # An off-by-one leaves a slice holding whatever `np.empty` found, which
    # on a fresh allocation is zero pages — and `standard_normal` returns
    # exactly 0.0 with probability ~0, so a single exact zero is evidence of
    # an unwritten column.
    #
    # Comparing quantiles cannot do this job, which is why it is a separate
    # test: a whole unwritten trailing block (2% of the panel, zeros) moves
    # the 1% quantile of a standard normal by about 0.009, well inside any
    # tolerance loose enough to survive two different seeds, and a realistic
    # `stop = start + block - 1` slip leaves 1/4096 of columns unwritten and
    # is invisible at any tolerance at all.
    out = _draw_uncorrelated((2, 200_000), "normal", 3, np.float64, np.random.default_rng(4), target_bytes=64 * 1024)
    # Only the exact-zero check earns its place. `assert isfinite(...)` was
    # here too and could not fail — `standard_normal` never returns a
    # non-finite value at either dtype, and this branch has no division.
    # Removed rather than left as decoration, in a file whose other
    # comments argue against exactly that.
    assert int((out == 0.0).sum()) == 0


def test_draw_uncorrelated_student_t_matches_scipy_quantiles() -> None:
    df = 3
    out = _draw_uncorrelated((1, 400_000), "student-t", df, np.float64, np.random.default_rng(5))
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    np.testing.assert_allclose(np.quantile(out, qs), stats.t.ppf(qs, df), atol=0.03, rtol=0)


@pytest.mark.parametrize("df", [1, 2, 3])
def test_student_t_draw_matches_the_distribution_across_df(df: int) -> None:
    # Quartiles against scipy's t(df), across the range where the tail gets
    # heavy. Fails on an algebra slip in Z / sqrt(X/df) or a wrong Gamma
    # shape. `target_bytes` is set so this spans many blocks rather than
    # one — at the default it would be a single block and the comment would
    # be describing coverage the call does not have.
    #
    # Note what is deliberately NOT asserted here. An `isfinite` check is
    # vacuous AT THESE df VALUES — the block is assembled in float64 and
    # only the quotient is cast, so at df in {1,2,3} it could only fail
    # above |t| > 3.4e38. (It is NOT vacuous across the whole accepted
    # domain: df=0.1 does overflow the float32 cast. That is why the draw
    # itself now checks finiteness, and why it is tested separately rather
    # than bolted on here.) A tail BOUND would be worse — a heavy tail is
    # the point of Student-t, so any threshold is unreachable or flaky.
    out = _draw_uncorrelated(
        (2, 500_000), "student-t", df, np.float32, np.random.default_rng(6), target_bytes=64 * 1024
    )
    lo, hi = stats.t.ppf([0.25, 0.75], df)
    np.testing.assert_allclose(np.quantile(out, [0.25, 0.75]), [lo, hi], atol=0.02, rtol=0)


@pytest.mark.parametrize(
    "bad",
    [
        0,  # ZeroDivisionError from 2.0 / df before the check existed
        -3,  # "shape < 0" out of standard_gamma
        float("inf"),  # standard_gamma(inf) -> inf -> 2/inf -> nan: a SILENTLY all-NaN panel
        float("nan"),
        True,  # an int subclass: would pass a naive check and draw t(1), Cauchy
        "3",
        None,
    ],
)
def test_draw_uncorrelated_rejects_a_nonsense_df(bad) -> None:
    # This is an IMPROVEMENT on scipy, not a restoration of it — measured,
    # because an earlier version of this comment claimed the latter. scipy
    # raised "Domain error in arguments" for 0, -3 and nan only. `inf`
    # returned NaN silently, True quietly drew t(1), and "3"/None raised
    # type errors rather than domain ones. Four of these seven cases scipy
    # never caught.
    with pytest.raises(ValueError, match="df must be a finite positive number"):
        _draw_uncorrelated((2, 16), "student-t", bad, np.float32, np.random.default_rng(0))


@pytest.mark.parametrize("tiny_df", [0.1, 1e-8])
def test_draw_uncorrelated_raises_on_a_df_too_small_to_sample(tiny_df: float) -> None:
    # The hole the `inf` rejection left open. A small-but-finite df passes
    # every type and domain check and then produces a panel that is partly
    # or almost entirely non-finite: at df=0.1 the float64 quotient
    # overflows the float32 cast, and at df=1e-8 `standard_gamma` returns
    # exact zeros and the division blows up. Rejecting `inf` for producing
    # silent garbage while waving these through was the inconsistency.
    #
    # There is no clean floor to hard-code — it depends on df, dtype and
    # how many values are drawn — so the draw checks the property instead.
    with pytest.raises(ValueError, match="non-finite values"):
        _draw_uncorrelated((2, 2_000_000), "student-t", tiny_df, np.float32, np.random.default_rng(0))


@pytest.mark.parametrize("good", [3, 3.0, np.int64(3), np.float32(3.0), np.float64(3.0)])
def test_draw_uncorrelated_accepts_numpy_scalar_df(good) -> None:
    # A caller deriving df from a config frame or a Series.item() hands over
    # a numpy scalar. `isinstance(x, (int, float))` is False for np.int64
    # and np.float32, so a naive check would reject a perfectly valid
    # argument that scipy always accepted.
    out = _draw_uncorrelated((2, 16), "student-t", good, np.float32, np.random.default_rng(0))
    assert out.shape == (2, 16)


def test_draw_uncorrelated_rejects_an_unknown_distribution() -> None:
    # Both branch points test == "normal" / != "normal", so before this
    # check `distribution="gaussian"` silently produced Student-t with
    # df=3 — three times the requested variance, no error, while
    # `precision` one screen above raised on a bad value.
    with pytest.raises(ValueError, match="distribution must be"):
        _draw_uncorrelated((2, 16), "gaussian", 3, np.float32, np.random.default_rng(0))
