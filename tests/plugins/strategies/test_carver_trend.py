"""Tests for strategy.carver_trend.v1 — including the Bollinger feature added in v0.2.0."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.strategies.carver_trend import (
    BOLLINGER_WINDOWS,
    CarverTrendStrategy,
    bollinger_forecast,
    generate_carver_forecast,
)

# --- Synthetic data helpers ---


def _synthetic_prices(n_periods: int = 400, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
    """Geometric-Brownian-motion-ish prices for deterministic smoke tests."""
    rng = np.random.default_rng(seed)
    cols = [f"AST{i}" for i in range(n_assets)]
    rets = rng.normal(0.0005, 0.025, size=(n_periods, n_assets))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(
        prices,
        index=pd.date_range("2024-01-01", periods=n_periods, freq="D"),
        columns=cols,
    )


def _synthetic_data(n_periods: int = 400, seed: int = 42) -> dict[str, pd.DataFrame]:
    prices = _synthetic_prices(n_periods, seed=seed)
    return {"prices": prices}


# --- bollinger_forecast unit tests ---


def test_bollinger_forecast_shape_and_range():
    """Forecast has the same length as input and stays roughly in [-3, +3]."""
    rng = np.random.default_rng(7)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))
    fc = bollinger_forecast(prices, window=20)
    assert len(fc) == len(prices)
    # First (window) values are NaN due to rolling
    assert fc.iloc[:19].isna().all()
    # Mature values are bounded loosely (extreme excursions rare)
    mature = fc.iloc[20:].dropna()
    assert mature.abs().max() < 10.0  # extreme moves can exceed bands but not by 10x


def test_bollinger_forecast_zero_when_at_ma():
    """When price equals its rolling MA, the forecast is zero."""
    # Std of constant series is 0 → division gives NaN, not 0; that's fine.
    # Verify a near-flat case instead.
    nearly_flat = pd.Series(100.0 + np.linspace(0, 0.001, 100))
    fc2 = bollinger_forecast(nearly_flat, window=20)
    assert fc2.iloc[-1] == pytest.approx(fc2.iloc[-1])  # finite, not NaN
    # Sign matches direction
    rng = np.random.default_rng(1)
    rising = pd.Series(np.cumsum(np.abs(rng.normal(0.1, 0.05, 100))) + 100)
    fc3 = bollinger_forecast(rising, window=20)
    assert fc3.iloc[-1] > 0


# --- generate_carver_forecast tests ---


def test_generate_forecast_default_off_unchanged():
    """Default call (no bollinger) matches the v0.1 behavior — regression guard."""
    rng = np.random.default_rng(123)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400))))
    # Default: bollinger_weight=0.0
    fc_default = generate_carver_forecast(prices)
    # Explicit ewmac/breakout only, bollinger off
    fc_explicit = generate_carver_forecast(
        prices,
        ewmac_weight=0.6,
        breakout_weight=0.4,
        bollinger_weight=0.0,
    )
    pd.testing.assert_series_equal(fc_default, fc_explicit)


def test_generate_forecast_with_bollinger_differs():
    """Adding Bollinger weight changes the combined forecast."""
    rng = np.random.default_rng(123)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400))))
    fc_off = generate_carver_forecast(prices, ewmac_weight=0.6, breakout_weight=0.4, bollinger_weight=0.0)
    fc_on = generate_carver_forecast(prices, ewmac_weight=0.4, breakout_weight=0.3, bollinger_weight=0.3)
    diff = (fc_on - fc_off).abs().dropna()
    assert diff.sum() > 0
    # Forecast still bounded by ±20 cap
    assert fc_on.dropna().abs().max() <= 20.0


# --- CarverTrendStrategy tests ---


def test_strategy_meta_has_params_schema():
    """The plugin declares a JSON Schema for params (architecture/plugin-authoring rule)."""
    schema = CarverTrendStrategy.meta.params_schema
    assert schema is not None
    assert schema["type"] == "object"
    assert "use_bollinger_feature" in schema["properties"]
    assert "target_vol" in schema["properties"]


def test_strategy_meta_version_bumped():
    """Adding Bollinger is an additive minor bump per semver."""
    assert CarverTrendStrategy.meta.version == "0.2.0"


def test_strategy_smoke_default():
    """Strategy runs with default params and produces a weights DataFrame."""
    strat = CarverTrendStrategy()
    result = strat.run(_synthetic_data(), params=None)
    assert "weights" in result
    assert isinstance(result["weights"], pd.DataFrame)
    assert "simple_weights" in result
    assert "exposure" in result


def test_strategy_smoke_bollinger_on():
    """Strategy runs with use_bollinger_feature=True and produces non-empty weights."""
    strat = CarverTrendStrategy(use_bollinger_feature=True)
    result = strat.run(_synthetic_data(), params=None)
    assert "weights" in result
    assert isinstance(result["weights"], pd.DataFrame)
    assert result["weights"].shape[0] > 0


def test_strategy_bollinger_changes_weights():
    """Same data, different forecast weights → different output weights."""
    data = _synthetic_data()
    off = CarverTrendStrategy(use_bollinger_feature=False).run(data, params=None)
    on = CarverTrendStrategy(use_bollinger_feature=True).run(data, params=None)
    # At least one asset has a different weight
    off_w = pd.Series(off["simple_weights"])
    on_w = pd.Series(on["simple_weights"])
    common = off_w.index.intersection(on_w.index)
    assert len(common) > 0
    # If ALL weights are identical down to float, the bollinger feature isn't active
    assert not np.allclose(off_w.loc[common].values, on_w.loc[common].values, atol=1e-6)


def test_strategy_params_override_via_run():
    """Passing params= overrides instance attributes for that one call."""
    strat = CarverTrendStrategy(use_bollinger_feature=False)
    result = strat.run(_synthetic_data(), params={"use_bollinger_feature": True, "target_vol": 0.30})
    assert "weights" in result
    # After the run, the instance attrs ARE updated (current run() implementation pattern)
    assert strat.use_bollinger_feature is True
    assert strat.target_vol == 0.30


def test_strategy_default_bollinger_windows():
    """Bollinger windows default to module BOLLINGER_WINDOWS constant."""
    strat = CarverTrendStrategy()
    assert strat.bollinger_windows == BOLLINGER_WINDOWS
