"""strategy.momentum_long_short.v1: skip_periods (12-1) and weighting (equal | inverse_vol | rank), TOM-3927."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.config_checks import check_plugin_params
from quantbox.plugins.strategies.momentum_long_short import (
    MomentumLongShortStrategy,
    compute_momentum_signal,
    construct_long_short_weights,
    construct_rank_weights,
    zscore_cross_sectional,
)

NAME = "strategy.momentum_long_short.v1"


def _prices(n: int = 160, n_assets: int = 7, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rets = rng.normal(0.0005, 0.03, (n, n_assets))
    return pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=[f"A{i}" for i in range(n_assets)])


def _run(prices: pd.DataFrame, **params) -> pd.DataFrame:
    base = {
        "enable_trend_filter": False,
        "rebalance_frequency": "D",
        "output_periods": len(prices),
        "exclude_tickers": [],
    }
    return MomentumLongShortStrategy().run({"prices": prices}, {**base, **params})["weights"]


# --- rank weights: Asness, Moskowitz, Pedersen (2013) -----------------------------------------


def test_rank_weights_reproduce_amp_fixture():
    """w_i = c (rank_i - mean rank), $1 long and $1 short; ties take the average rank, NaN is unranked."""
    signals = pd.DataFrame(
        [
            [0.30, -0.10, 0.05, 0.20, -0.40],  # ranks 5 2 3 4 1, mean 3 -> 2 -1 0 1 -2, long sum 3
            [0.10, 0.10, np.nan, -0.20, 0.00],  # ranks 3.5 3.5 - 1 2, mean 2.5 -> 1 1 - -1.5 -0.5, long sum 2
            [np.nan, 0.50, np.nan, np.nan, np.nan],  # one asset: nothing to rank against
        ],
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
        columns=list("ABCDE"),
    )
    expected = np.array(
        [
            [2 / 3, -1 / 3, 0.0, 1 / 3, -2 / 3],
            [0.5, 0.5, 0.0, -0.75, -0.25],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    got = construct_rank_weights(signals)
    np.testing.assert_allclose(got.to_numpy(), expected, rtol=0, atol=1e-12)


def test_rank_legs_are_one_dollar_and_net_exposure_tilts_them():
    signals = compute_momentum_signal(_prices(), [20], skip_periods=5).dropna(how="all")
    w = construct_rank_weights(signals)
    np.testing.assert_allclose(w.clip(lower=0).sum(axis=1), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(w.clip(upper=0).sum(axis=1), -1.0, rtol=0, atol=1e-12)

    tilted = construct_rank_weights(signals, net_exposure=0.2)
    np.testing.assert_allclose(tilted.clip(lower=0).sum(axis=1), 1.2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(tilted.clip(upper=0).sum(axis=1), -0.8, rtol=0, atol=1e-12)
    # Sign follows the rank: the top asset is long, the bottom one short.
    assert (np.sign(w.to_numpy()) == np.sign(tilted.to_numpy())).all()


def test_strategy_12_1_rank_matches_an_independent_reference():
    """Through run(): skip 3 of a 10-bar window, rank-weighted, against a numpy re-derivation."""
    prices = _prices()
    got = _run(prices, momentum_windows=[10], skip_periods=3, weighting="rank")

    p = prices.to_numpy()
    for i in (10, 57, len(p) - 1):
        s = p[i - 3] / p[i - 10] - 1.0  # P[t-skip] / P[t-window] - 1
        ranks = np.argsort(np.argsort(s)) + 1.0  # distinct returns: no ties
        d = ranks - ranks.mean()
        np.testing.assert_allclose(got.iloc[i].to_numpy(), d / d[d > 0].sum(), rtol=0, atol=1e-12)
    assert (got.iloc[:10] == 0).all().all()  # no 10-bar lookback yet


# --- skip_periods -------------------------------------------------------------------------------


def test_skip_periods_is_the_shorter_return_lagged():
    prices = _prices()
    got = compute_momentum_signal(prices, [10], skip_periods=3)
    pd.testing.assert_frame_equal(got, prices.shift(3) / prices.shift(10) - 1.0, rtol=1e-12)
    pd.testing.assert_frame_equal(got, prices.pct_change(7, fill_method=None).shift(3), rtol=1e-12)


def test_skip_periods_zero_is_todays_signal_exactly():
    prices = _prices()
    plain = (prices.ffill().pct_change(21, fill_method=None) + prices.ffill().pct_change(63, fill_method=None)) * 0.5
    pd.testing.assert_frame_equal(compute_momentum_signal(prices, [21, 63], skip_periods=0), plain, check_exact=True)


def test_skip_periods_never_reads_the_skipped_bars():
    """No look-ahead and a real skip: moving the last 3 bars leaves the signal at the last date unchanged."""
    prices = _prices()
    moved = prices.copy()
    moved.iloc[-3:] *= 1.5
    a = compute_momentum_signal(prices, [10], skip_periods=3).iloc[-1]
    b = compute_momentum_signal(moved, [10], skip_periods=3).iloc[-1]
    pd.testing.assert_series_equal(a, b, check_exact=True)


@pytest.mark.parametrize("skip", [-1, 10, 11, 2.5, True])
def test_skip_periods_refuses_what_measures_no_return(skip):
    with pytest.raises(ValueError, match="skip_periods"):
        compute_momentum_signal(_prices(), [10, 20], skip_periods=skip)


# --- weighting: defaults unchanged, equal is the volatility=None path ---------------------------


def test_defaults_are_todays_behaviour():
    s = MomentumLongShortStrategy()
    assert (s.weighting, s.skip_periods) == ("inverse_vol", 0)
    prices = _prices()
    pd.testing.assert_frame_equal(_run(prices), _run(prices, weighting="inverse_vol", skip_periods=0), check_exact=True)


def test_equal_weighting_is_the_unscaled_legs():
    prices = _prices()
    signals = zscore_cross_sectional(compute_momentum_signal(prices, [21, 63]))
    expected = construct_long_short_weights(signals, n_long=2, n_short=2, volatility=None)
    got = _run(prices, weighting="equal", n_long=2, n_short=2)
    pd.testing.assert_frame_equal(got, expected, check_exact=True, check_freq=False)
    assert not got.equals(_run(prices, n_long=2, n_short=2))  # inverse_vol differs


def test_run_refuses_an_unknown_weighting():
    with pytest.raises(ValueError, match="weighting"):
        _run(_prices(), weighting="inverse_atr")


# --- quantbox validate --------------------------------------------------------------------------


def _findings(params: dict) -> list[str]:
    return [f.message for f in check_plugin_params({"strategies": [{"name": NAME, "params": params}]})]


def test_validate_accepts_a_12_1_rank_config():
    assert _findings({"momentum_windows": [252], "skip_periods": 21, "weighting": "rank"}) == []


@pytest.mark.parametrize("params", [{"weighting": "inverse_atr"}, {"skip_periods": -1}, {"skip_periods": "21"}])
def test_validate_refuses_bad_values(params):
    assert any("invalid_param" in m for m in _findings(params))
