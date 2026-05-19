"""Tests for strategy.carry.v1 — funding-rate carry (Mode A: funding_momentum)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbox.plugins.strategies.carry import CarryStrategy
from quantbox.plugins.strategies.carry import run as carry_run

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_prices(n: int = 120, n_assets: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    cols = ["BTC", "ETH", "SOL", "AVAX", "DOGE", "ARB", "OP", "SUI"][:n_assets]
    rets = rng.normal(0.001, 0.03, (n, n_assets))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=cols)


def _make_funding(
    prices: pd.DataFrame,
    seed: int = 1,
    spread: float = 0.003,
) -> pd.DataFrame:
    """Synthetic daily-summed 8h funding rates with variance across assets."""
    rng = np.random.default_rng(seed)
    n, m = prices.shape
    # Some assets positive, some negative funding to create long/short candidates
    biases = np.linspace(-spread, spread, m)
    rates = rng.normal(biases, 0.0005, (n, m))
    return pd.DataFrame(rates, index=prices.index, columns=prices.columns)


def _make_data(n: int = 120, seed: int = 42) -> dict[str, pd.DataFrame]:
    prices = _make_prices(n, seed=seed)
    funding = _make_funding(prices, seed=seed + 1)
    return {"prices": prices, "funding_rates": funding}


# ---------------------------------------------------------------------------
# Meta tests
# ---------------------------------------------------------------------------


def test_meta_name():
    assert CarryStrategy.meta.name == "strategy.carry.v1"


def test_meta_core_compat():
    assert CarryStrategy.meta.core_compat == ">=0.1,<0.2"


def test_meta_status():
    assert CarryStrategy.meta.status == "research"


def test_meta_has_params_schema():
    assert CarryStrategy.meta.params_schema is not None
    assert "signal_span_days" in CarryStrategy.meta.params_schema["properties"]


def test_registered_in_builtins():
    from quantbox.plugins.builtins import builtins

    b = builtins()
    assert "strategy.carry.v1" in b["strategy"]


# ---------------------------------------------------------------------------
# Output contract tests
# ---------------------------------------------------------------------------


def test_run_returns_required_keys():
    data = _make_data()
    result = CarryStrategy().run(data)
    assert "weights" in result
    assert "simple_weights" in result
    assert "details" in result


def test_weights_is_dataframe():
    data = _make_data()
    result = CarryStrategy().run(data)
    assert isinstance(result["weights"], pd.DataFrame)


def test_weights_shape_respects_output_periods():
    data = _make_data(n=120)
    s = CarryStrategy(output_periods=30)
    result = s.run(data)
    assert len(result["weights"]) <= 30


def test_weights_columns_match_valid_tickers():
    data = _make_data()
    result = CarryStrategy().run(data)
    # All weight columns should come from the funding/prices universe (no stablecoins added)
    assert set(result["weights"].columns).issubset(set(data["prices"].columns))


def test_simple_weights_is_dict():
    data = _make_data()
    result = CarryStrategy().run(data)
    assert isinstance(result["simple_weights"], dict)


def test_module_level_run():
    """The module-level run() function must work identically."""
    data = _make_data()
    result = carry_run(data)
    assert "weights" in result
    assert isinstance(result["weights"], pd.DataFrame)


# ---------------------------------------------------------------------------
# Signal construction tests
# ---------------------------------------------------------------------------


def test_annualized_signal_formula():
    """EMA signal × 365 should match manual calculation."""
    prices = _make_prices(n=50, n_assets=2)
    idx = prices.index
    raw = pd.DataFrame(
        {"A": [0.0001] * 50, "B": [-0.0001] * 50},
        index=idx,
    )
    s = CarryStrategy(signal_span_days=3)
    sig = s._annualized_signal(raw)
    # Constant funding → EMA = same constant, annualized = val × 365
    np.testing.assert_allclose(sig["A"].iloc[-1], 0.0001 * 365, rtol=1e-4)
    np.testing.assert_allclose(sig["B"].iloc[-1], -0.0001 * 365, rtol=1e-4)


# ---------------------------------------------------------------------------
# Portfolio construction tests
# ---------------------------------------------------------------------------


def test_max_n_long_positions():
    """Strategy never takes more than top_n_long long positions."""
    data = _make_data()
    s = CarryStrategy(top_n_long=2, top_n_short=2, min_signal_annualized=0.0)
    result = s.run(data)
    w = result["weights"]
    max_longs = (w > 0).sum(axis=1).max()
    assert max_longs <= 2


def test_max_n_short_positions():
    """Strategy never takes more than top_n_short short positions."""
    data = _make_data()
    s = CarryStrategy(top_n_long=2, top_n_short=2, min_signal_annualized=0.0)
    result = s.run(data)
    w = result["weights"]
    max_shorts = (w < 0).sum(axis=1).max()
    assert max_shorts <= 2


def test_min_signal_floor_filters_positions():
    """With a very high floor, strategy returns zero weights."""
    data = _make_data()
    # Daily funding of ~0.003 annualizes to ~1.0; a floor of 100x that kills all signals
    s = CarryStrategy(min_signal_annualized=100.0)
    result = s.run(data)
    assert (result["weights"] == 0).all().all()


def test_max_concentration_cap():
    """No individual weight exceeds max_concentration in absolute value."""
    data = _make_data()
    s = CarryStrategy(max_concentration=0.15, min_signal_annualized=0.0)
    result = s.run(data)
    assert result["weights"].abs().max().max() <= 0.15 + 1e-9


def test_weight_signs_match_signal_direction():
    """Longs have the highest funding signal, shorts the lowest."""
    # Construct deterministic funding: asset 0 always highest, asset 7 always lowest
    prices = _make_prices(n=60, n_assets=8)
    funding_vals = np.tile(np.linspace(0.003, -0.003, 8), (60, 1))  # col 0 highest, col 7 lowest
    funding = pd.DataFrame(funding_vals, index=prices.index, columns=prices.columns)
    data = {"prices": prices, "funding_rates": funding}
    s = CarryStrategy(top_n_long=1, top_n_short=1, min_signal_annualized=0.0, vol_lookback=5)
    result = s.run(data)
    last_w = result["weights"].iloc[-1]
    assert last_w.iloc[0] > 0, "Highest-funding asset should be long"
    assert last_w.iloc[-1] < 0, "Lowest-funding asset should be short"


# ---------------------------------------------------------------------------
# Fallback: missing funding data
# ---------------------------------------------------------------------------


def test_no_funding_returns_zero_weights():
    """When funding_rates is absent, weights are zero (graceful degrade)."""
    prices = _make_prices()
    data = {"prices": prices}  # no funding_rates key
    result = CarryStrategy().run(data)
    assert (result["weights"] == 0).all().all()
    assert result["simple_weights"] == {}


def test_empty_funding_returns_zero_weights():
    """When funding_rates is an empty DataFrame, weights are zero."""
    prices = _make_prices()
    data = {"prices": prices, "funding_rates": pd.DataFrame()}
    result = CarryStrategy().run(data)
    assert (result["weights"] == 0).all().all()


# ---------------------------------------------------------------------------
# Vol targeting
# ---------------------------------------------------------------------------


def test_vol_targeting_scales_weights():
    """Vol targeting should produce non-trivial scaling (not all 1.0)."""
    data = _make_data(n=200)
    s_no_vt = CarryStrategy(target_vol=1e6, vol_lookback=20, min_signal_annualized=0.0)
    s_vt = CarryStrategy(target_vol=0.10, vol_lookback=20, min_signal_annualized=0.0)
    r_no = s_no_vt.run(data)
    r_vt = s_vt.run(data)
    # With target_vol=0.10, gross exposure should generally be lower than with enormous target
    gross_no = r_no["weights"].abs().sum(axis=1).mean()
    gross_vt = r_vt["weights"].abs().sum(axis=1).mean()
    # They shouldn't be identical
    assert not np.isclose(gross_no, gross_vt, rtol=0.01)


# ---------------------------------------------------------------------------
# Params override
# ---------------------------------------------------------------------------


def test_params_override_at_runtime():
    """Params passed to run() override instance defaults."""
    data = _make_data()
    s = CarryStrategy(top_n_long=3)
    result = s.run(data, params={"top_n_long": 1, "top_n_short": 1, "min_signal_annualized": 0.0})
    w = result["weights"]
    max_longs = (w > 0).sum(axis=1).max()
    assert max_longs <= 1
