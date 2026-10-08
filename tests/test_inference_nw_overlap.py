"""``newey_west_tstat(..., overlap=h)``: the lag floor for overlapping observations (TOM-1644).

A series of overlapping h-period observations (an IC on h-day forward returns,
a rolling h-day spread) shares h-1 periods between neighbours, so it is an
MA(h-1) by construction. The Newey-West lag count must then be at least h-1
(Hansen-Hodrick 1980); the automatic rule alone leaves the long-horizon t-stat
overstated. ADR-0009, amendment of 2026-10-08.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantbox.inference import InferenceInputError, newey_west_auto_lags, newey_west_tstat


def _overlapping_sums(n: int, h: int, seed: int) -> np.ndarray:
    """``n`` overlapping h-period sums of iid N(0, 1) shocks: an MA(h-1) with unit coefficients."""
    e = np.random.default_rng(seed).normal(0.0, 1.0, n + h - 1)
    return np.convolve(e, np.ones(h), mode="valid")


# --- the known answer ---


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_overlap_corrects_the_overstated_tstat_of_an_ma_series(seed):
    """MA(h-1), h=63 (a quarter of trading days), n=20000. True SE of the mean: h/sqrt(n).

    The long-run variance of the series is h^2. Bartlett weights at the auto lag
    count (12 here) recover 19% of it: SE ratio 0.438, a t-stat overstated 2.3x.
    At h-1 = 62 lags they recover 2/3 of it (Bartlett's own downweighting): SE
    ratio 0.817. The bounds sit between those two answers, so the test fails
    when the floor is not applied.
    """
    n, h = 20_000, 63
    x = _overlapping_sums(n, h, seed)
    true_se = h / np.sqrt(n)

    auto = newey_west_tstat(x)
    overlap = newey_west_tstat(x, overlap=h)

    assert auto["nw_lags"] == newey_west_auto_lags(n) == 12
    assert overlap["nw_lags"] == h - 1
    assert auto["nw_se"] / true_se < 0.6  # the auto-lag t is overstated
    assert 0.7 < overlap["nw_se"] / true_se < 1.0  # the overlap t is not
    # Same mean, so the t-stats scale inversely with the SE.
    assert abs(overlap["nw_tstat"]) < abs(auto["nw_tstat"])


# --- the rule: lags = max(auto, h - 1) ---


def test_overlap_equals_explicit_h_minus_1_lags_when_above_auto():
    x = _overlapping_sums(2_000, 63, seed=3)
    by_overlap = newey_west_tstat(x, overlap=63)
    by_lags = newey_west_tstat(x, lags=62)
    assert by_overlap["nw_lags"] == 62
    for key in ("nw_se", "nw_tstat", "nw_pvalue", "mean_return"):
        assert by_overlap[key] == by_lags[key]


def test_short_overlap_keeps_the_auto_lags():
    x = _overlapping_sums(1_000, 3, seed=4)
    auto = newey_west_auto_lags(1_000)
    assert auto > 2
    assert newey_west_tstat(x, overlap=3)["nw_lags"] == auto


def test_overlap_1_is_non_overlapping_and_changes_nothing():
    x = _overlapping_sums(500, 1, seed=5)
    assert newey_west_tstat(x, overlap=1) == {**newey_west_tstat(x), "nw_overlap": 1}


def test_overlap_is_reported():
    x = _overlapping_sums(500, 5, seed=6)
    assert newey_west_tstat(x, overlap=5)["nw_overlap"] == 5
    assert newey_west_tstat(x)["nw_overlap"] is None
    assert newey_west_tstat(x, lags=3)["nw_overlap"] is None


# --- explicit lags win; a conflict is refused ---


def test_explicit_lags_at_or_above_the_floor_win():
    x = _overlapping_sums(1_000, 21, seed=7)
    assert newey_west_tstat(x, lags=20, overlap=21)["nw_lags"] == 20
    assert newey_west_tstat(x, lags=30, overlap=21)["nw_lags"] == 30


def test_explicit_lags_below_the_floor_are_refused():
    x = _overlapping_sums(1_000, 21, seed=8)
    with pytest.raises(InferenceInputError, match="overlap=21"):
        newey_west_tstat(x, lags=5, overlap=21)


# --- invalid overlap ---


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "21"])
def test_invalid_overlap_is_refused(bad):
    x = _overlapping_sums(200, 2, seed=9)
    with pytest.raises(InferenceInputError, match="overlap"):
        newey_west_tstat(x, overlap=bad)


def test_overlap_longer_than_the_sample_is_refused():
    """h-1 lags cannot be honoured on n <= h-1 observations; clamping would hide the overstatement."""
    x = _overlapping_sums(30, 5, seed=10)
    with pytest.raises(InferenceInputError, match="overlap"):
        newey_west_tstat(x, overlap=31)
