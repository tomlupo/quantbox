"""Basic tests for ``quantbox.features.simulations``.

Covers a happy-path call for ``parametric_mc`` (shape + determinism via
seeded Generator), plus the frequency constants export.

``simulations_stats`` is exercised by the host project (robo) via its
byte-identical end-to-end gate; the input panel requires a specific
(date, ticker, step, sim_no) layout constructed by robo's sim driver,
so a standalone quantbox-side integration test isn't meaningful — it
is a straight data-transform whose correctness rides on the host.
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


# Bit-identity, checked two ways — the claim this branch makes is that the
# refactor above `parametric_mc`'s shock/returns lines changes allocations
# and nothing else.
#
# 1. `test_parametric_mc_matches_reference_implementation` runs the
#    pre-refactor expressions and the current ones on the SAME machine and
#    demands exact equality. That is the actual claim, it covers both the
#    correlated and the uncorrelated branch, and it holds on any CPU.
#
# 2. `test_parametric_mc_matches_stored_reference` pins a stored panel, so
#    a change in scipy's sampling internals — `pyproject.toml` bounds
#    scipy only from below — is caught rather than silently absorbed.


def _reference_parametric_mc(
    mu: pd.Series,
    cov: pd.DataFrame,
    iterations: int,
    steps: int,
    correlated: bool,
    distribution: str,
    df: int,
    seed: np.random.Generator,
    precision: str,
) -> np.ndarray:
    """The panel `parametric_mc` produced BEFORE this branch's refactor.

    Transcribed expression for expression from
    `src/quantbox/features/simulations.py` at 58e5e09, covering the
    `mu`/`cov`-supplied call path only, and returning the raw
    `returns_sim` array from just before the reshape (which this branch
    does not touch). Deliberately dumb: its job is to be recognisably the
    old lines, not to be good code.
    """
    frequency = 252
    index = mu.index
    var = pd.Series(np.diag(cov), index=cov.index).loc[index]
    cov = cov.loc[index, index]

    step_frequency = frequency
    mu = mu / step_frequency
    cov = cov / step_frequency
    var = var / step_frequency

    drift = mu - 0.5 * var
    dtype = getattr(np, precision)
    drift = drift.astype(dtype)
    if correlated:
        cov = cov.astype(dtype)
        chol = np.linalg.cholesky(cov)
    else:
        var = var.astype(dtype)

    if distribution == "normal":
        uncorr_x = stats.norm.rvs(size=(len(mu), iterations * steps), random_state=seed).astype(dtype)
    else:
        uncorr_x = stats.t.rvs(df, size=(len(mu), iterations * steps), random_state=seed).astype(dtype)

    if correlated:
        shock = np.dot(chol, uncorr_x).astype(dtype)
    else:
        shock = (uncorr_x * np.tile(np.atleast_2d(np.sqrt(var)).T, uncorr_x.shape[1])).astype(dtype)

    return np.exp(np.atleast_2d(drift).T + shock).astype(dtype) - 1


@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("distribution", ["normal", "student-t"])
@pytest.mark.parametrize("correlated", [True, False])
def test_parametric_mc_matches_reference_implementation(correlated: bool, distribution: str, precision: str) -> None:
    tickers = ["AAA", "BBB"]
    mu = pd.Series([0.05, 0.07], index=tickers)
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=tickers, columns=tickers)
    steps, iterations = 5, 7
    kwargs = dict(
        iterations=iterations,
        steps=steps,
        correlated=correlated,
        distribution=distribution,
        df=3,
        precision=precision,
    )

    got = parametric_mc(mu=mu, cov=cov, seed=np.random.default_rng(2026), **kwargs).to_numpy()

    reference = _reference_parametric_mc(mu=mu, cov=cov, seed=np.random.default_rng(2026), **kwargs)
    # The reshape/concat the function applies to `returns_sim`, unchanged
    # by this branch and reproduced here so the comparison is on panels.
    want = np.hstack([reference[i].reshape(steps, iterations) for i in range(len(tickers))])

    assert got.dtype == getattr(np, precision)
    np.testing.assert_array_equal(got, want)


# Generated at 39cc04b and verified equal to the revision before it. Kept
# tiny and fully written out: a reference you can read is one you can
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
            -0.01000682826373811,
            0.015463354591143341,
            -0.0037696558119070245,
            -0.008715843404286328,
            0.019606705896591414,
            0.006083138915198294,
        ],
        [
            -0.0022668083295737107,
            -0.0008341994945846309,
            -0.010534677379957835,
            0.059497612692870794,
            -0.014679140902960519,
            0.036888500318631445,
        ],
    ],
    ("float32", "normal"): [
        [-0.009824812, 0.0031548738, -0.023492038, -0.008178115, 0.0065398216, -0.010802448],
        [0.017861724, 0.008194208, -0.0035541058, 0.0002859831, 0.015648484, 0.008808851],
    ],
    ("float32", "student-t"): [
        [-0.010006785, 0.015463352, -0.0037696362, -0.008715868, 0.01960659, 0.006083131],
        [-0.0022668242, -0.0008342266, -0.010534644, 0.059497595, -0.014679074, 0.0368886],
    ],
}

# Why this one is `assert_allclose` and not `assert_array_equal`, when
# bit-identity is the whole point of the branch: as an exact assertion it
# went red on GitHub's runners while passing on every dev box, by 1 ULP
# (2.2e-16) in two of twelve float64 elements. `np.dot` here goes to BLAS,
# which selects its kernel from the CPU, so the last bit of a float64
# product is a property of the machine, not of this code —
# `test_parametric_mc_matches_reference_implementation` is what pins the
# refactor, because it compares old and new ON the same machine. This test
# is a drift alarm: the tolerances sit orders of magnitude above one ULP
# and orders of magnitude below any drift worth hearing about.
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
