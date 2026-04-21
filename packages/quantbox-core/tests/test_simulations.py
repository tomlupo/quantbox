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
