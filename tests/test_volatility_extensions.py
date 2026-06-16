"""Tests for the ``returns_method`` param + ``compute_riskmetrics_vol``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.features import (
    compute_ewm_vol,
    compute_riskmetrics_vol,
    compute_rolling_vol,
)


@pytest.fixture
def toy_prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=2500, freq="B")
    tickers = ["AAA", "BBB"]
    rets = rng.standard_normal((len(dates), len(tickers))) * 0.01
    return pd.DataFrame(100 * np.exp(rets.cumsum(axis=0)), index=dates, columns=tickers)


def test_rolling_vol_log_vs_pct_change_differs(toy_prices: pd.DataFrame) -> None:
    out_log = compute_rolling_vol(toy_prices, [63], returns_method="log", factor=252.0)
    out_pct = compute_rolling_vol(toy_prices, [63], returns_method="pct_change", factor=252.0)
    log_tail = out_log["vol_63d"].iloc[-1]
    pct_tail = out_pct["vol_63d"].iloc[-1]
    # Different returns → different std; gap should be small (sub-percent)
    # but non-zero on a non-degenerate sample.
    assert (log_tail - pct_tail).abs().sum() > 0


def test_rolling_vol_pct_change_matches_direct(toy_prices: pd.DataFrame) -> None:
    out = compute_rolling_vol(toy_prices, [21], returns_method="pct_change", factor=252.0)
    expected = toy_prices.pct_change(fill_method=None).rolling(21).std() * np.sqrt(252)
    pd.testing.assert_frame_equal(out["vol_21d"], expected)


def test_rolling_vol_invalid_method(toy_prices: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="returns_method"):
        compute_rolling_vol(toy_prices, [21], returns_method="bogus")


def test_ewm_vol_backwards_compatible_default(toy_prices: pd.DataFrame) -> None:
    """Default ``returns_method="log"`` preserves the prior output."""
    out = compute_ewm_vol(toy_prices, [32], factor=252.0)
    log_ret = np.log(toy_prices / toy_prices.shift(1))
    expected = log_ret.ewm(span=32).std() * np.sqrt(252)
    pd.testing.assert_frame_equal(out["vol_ewm_32"], expected)


def test_riskmetrics_vol_default_matches_manual(toy_prices: pd.DataFrame) -> None:
    """Spot-check RiskMetrics: sqrt(EWM-mean(pct_change^2)), trim ``100/alpha``."""
    alpha = 0.06  # RiskMetrics lambda = 0.94
    out = compute_riskmetrics_vol(toy_prices, [alpha], factor=252.0)
    rets = toy_prices.pct_change(fill_method=None).iloc[1:]
    ewma_var = (rets**2).ewm(alpha=alpha).mean()
    expected = ewma_var.pow(0.5) * np.sqrt(252)
    min_pts = int(100 / alpha)
    expected = expected.iloc[min_pts:]
    pd.testing.assert_frame_equal(out["vol_ewma_alpha6"], expected)


def test_riskmetrics_vol_no_trim(toy_prices: pd.DataFrame) -> None:
    alpha = 0.1
    out = compute_riskmetrics_vol(toy_prices, [alpha], factor=252.0, warmup_trim=False)
    df = out["vol_ewma_alpha10"]
    # Full length (minus the leading .iloc[1:] shift-drop)
    assert len(df) == len(toy_prices) - 1


def test_riskmetrics_vol_distinct_from_ewm_vol(toy_prices: pd.DataFrame) -> None:
    """RiskMetrics (EWM of variance) and ``compute_ewm_vol`` (EWM of std)
    are genuinely different estimators — verify they disagree on a
    non-degenerate input."""
    alpha = 0.06
    rm = compute_riskmetrics_vol(toy_prices, [alpha], factor=252.0, warmup_trim=False)["vol_ewma_alpha6"]
    # Match the span convention: span = 2/alpha - 1.
    span = int(2 / alpha - 1)
    ewm = compute_ewm_vol(toy_prices, [span], returns_method="pct_change", factor=252.0)[f"vol_ewm_{span}"]
    # Align indices; iloc[1:] for the rm side since it has one fewer row.
    common = rm.index.intersection(ewm.index)
    delta = (rm.loc[common] - ewm.loc[common]).abs().sum().sum()
    assert delta > 0
