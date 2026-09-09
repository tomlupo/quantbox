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


# A stored panel, so "this refactor is bit-identical" is a claim the suite
# can fail rather than one a reviewer has to take on trust. Generated at
# 39cc04b, whose output was verified byte-identical to the revision before
# it across correlated x {normal, t} x {float32, float64}, so the fixture
# pins the pre-refactor behaviour just as well as the post-.
#
# Deliberately tiny and fully written out: a reference you can read is a
# reference you can reason about when it fails.
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
    np.testing.assert_array_equal(got, want)
