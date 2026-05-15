"""Tests for :mod:`quantbox.analysis.parameter_grid`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbox.analysis.parameter_grid import _parse_vbt_slice_label, sweep
from quantbox.plugins.strategies.vol_matched_buy_hold import VolMatchedBuyHoldStrategy


def _btc_prices(n: int = 600, daily_vol: float = 0.03, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Strip the .freq attribute so vbt+pandas don't trip on the Day offset.
    # Matches the way real-world parquet-backed prices arrive (no freq).
    dates = pd.DatetimeIndex(pd.date_range("2023-01-01", periods=n, freq="D").values)
    btc = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, daily_vol, n))
    other = pd.Series(100.0, index=dates).values
    return pd.DataFrame({"BTC": btc, "USD": other}, index=dates)


def test_parse_simple_label() -> None:
    assert _parse_vbt_slice_label("vol-50") == {"vol": 50.0}
    assert _parse_vbt_slice_label("trend-on_size-5") == {"trend": "on", "size": 5.0}


def test_parse_non_label() -> None:
    assert _parse_vbt_slice_label("notalabel") == {"slice": "notalabel"}
    assert _parse_vbt_slice_label(("BTC", 5)) == {"slice": ("BTC", 5)}


def test_sweep_vol_matched_btc() -> None:
    """Sweep target_annual_vol on the vol-matched BTC strategy. Realised vol
    should track the target ordering across the grid (monotonic increase)."""
    prices = _btc_prices(n=800, daily_vol=0.04, seed=42)
    targets = [0.10, 0.20, 0.40]
    grid = sweep(
        strategy_cls=VolMatchedBuyHoldStrategy,
        base_params={"ticker": "BTC", "trading_days": 365},
        sweep_params={"target_annual_vol": targets},
        data={"prices": prices},
        backtest_kwargs={"fees": 0.0, "rebalancing_freq": 1},
        metrics=["sharpe_ratio", "annualized_volatility"],
        shift_signal=1,
    )
    assert len(grid) == 3
    assert "target_annual_vol" in grid.columns
    assert "annualized_volatility" in grid.columns

    # Realised vol should match target within 5% (fraction form from deep_getattr).
    grid_sorted = grid.sort_values("target_annual_vol").reset_index(drop=True)
    for i, target in enumerate(targets):
        realised = grid_sorted.loc[i, "annualized_volatility"]
        assert abs(realised - target) / target < 0.05, (
            f"target={target}: realised vol {realised:.4f} drifted >5% from target"
        )


def test_sweep_preserves_sweep_columns() -> None:
    prices = _btc_prices(n=300, seed=7)
    grid = sweep(
        strategy_cls=VolMatchedBuyHoldStrategy,
        base_params={"ticker": "BTC"},
        sweep_params={"target_annual_vol": [0.25, 0.50]},
        data={"prices": prices},
        backtest_kwargs={"fees": 0.0, "rebalancing_freq": 1},
    )
    assert set(grid["target_annual_vol"].tolist()) == {0.25, 0.50}
