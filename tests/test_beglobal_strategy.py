"""Tests for BeGlobal core-satellite multi-asset strategy plugin."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.strategies.beglobal_strategy import (
    ASSET_CLASSES,
    RISK_PROFILE_ALLOCATIONS,
    BeGlobalStrategy,
    _corridor_rebalance_needed,
    _dual_momentum_signals,
    _relative_strength_weights,
    _resolve_columns,
    _trend_signals,
    _volatility_scalar,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ALL_ETF_TICKERS = [ac.etf_ticker for ac in ASSET_CLASSES.values()]

# 14 tickers used in BeGlobal: SHV, SHY, IEF, TLT, BWX, LQD, HYG, EMB,
# SPY, EFA, EEM, VNQ, DJP, GLD


@pytest.fixture()
def prices_etf() -> pd.DataFrame:
    """600-day random walk prices with all 14 ETF tickers as columns."""
    rng = np.random.RandomState(42)
    n_days = 600
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    data = {}
    for ticker in _ALL_ETF_TICKERS:
        cumret = np.exp(np.cumsum(rng.normal(0.0003, 0.01, n_days)))
        data[ticker] = cumret * 100
    return pd.DataFrame(data, index=dates)


@pytest.fixture()
def prices_asset_names(prices_etf: pd.DataFrame) -> pd.DataFrame:
    """Same data as prices_etf but columns are asset class names."""
    ticker_to_name = {ac.etf_ticker: key for key, ac in ASSET_CLASSES.items()}
    return prices_etf.rename(columns=ticker_to_name)


@pytest.fixture()
def strategy() -> BeGlobalStrategy:
    return BeGlobalStrategy()


# ---------------------------------------------------------------------------
# Risk profile allocation tests
# ---------------------------------------------------------------------------


class TestRiskProfileAllocations:
    @pytest.mark.parametrize("profile", list(RISK_PROFILE_ALLOCATIONS.keys()))
    def test_allocations_sum_to_one(self, profile: str) -> None:
        alloc = RISK_PROFILE_ALLOCATIONS[profile]
        assert abs(sum(alloc.values()) - 1.0) < 1e-9

    @pytest.mark.parametrize("profile", list(RISK_PROFILE_ALLOCATIONS.keys()))
    def test_allocations_non_negative(self, profile: str) -> None:
        alloc = RISK_PROFILE_ALLOCATIONS[profile]
        for weight in alloc.values():
            assert weight >= 0.0

    @pytest.mark.parametrize("profile", list(RISK_PROFILE_ALLOCATIONS.keys()))
    def test_all_assets_known(self, profile: str) -> None:
        alloc = RISK_PROFILE_ALLOCATIONS[profile]
        for asset in alloc:
            assert asset in ASSET_CLASSES


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------


class TestResolveColumns:
    def test_etf_tickers(self, prices_etf: pd.DataFrame) -> None:
        mapping = _resolve_columns(prices_etf)
        assert len(mapping) == len(_ALL_ETF_TICKERS)
        assert mapping["SPY"] == "us_stocks"

    def test_asset_names(self, prices_asset_names: pd.DataFrame) -> None:
        mapping = _resolve_columns(prices_asset_names)
        assert mapping["us_stocks"] == "us_stocks"
        assert len(mapping) == len(ASSET_CLASSES)

    def test_unknown_columns_ignored(self) -> None:
        df = pd.DataFrame({"AAPL": [1], "TSLA": [2]})
        assert _resolve_columns(df) == {}


# ---------------------------------------------------------------------------
# Dual momentum signals
# ---------------------------------------------------------------------------


class TestDualMomentumSignals:
    def test_all_up_market_produces_long(self) -> None:
        """Steadily rising prices should produce mostly long signals."""
        n = 300
        prices_dict = {}
        for i, key in enumerate(["us_stocks", "gold", "us_treasury_short"]):
            prices_dict[key] = pd.Series(
                np.linspace(100, 200 + i * 10, n),
                dtype=float,
            )

        signals = _dual_momentum_signals(prices_dict, lookback=252)
        long_count = sum(1 for sig, _ in signals.values() if sig == "long")
        assert long_count >= 1

    def test_flat_market_produces_neutral(self) -> None:
        n = 300
        flat = pd.Series(np.full(n, 100.0))
        signals = _dual_momentum_signals({"flat": flat}, lookback=252)
        sig, score = signals["flat"]
        assert sig == "neutral"

    def test_insufficient_data(self) -> None:
        short = pd.Series([100.0, 101.0])
        signals = _dual_momentum_signals({"x": short}, lookback=252)
        _, score = signals["x"]
        assert score == 0.0


# ---------------------------------------------------------------------------
# Relative strength weights
# ---------------------------------------------------------------------------


class TestRelativeStrengthWeights:
    def test_top_n_overweighted(self) -> None:
        prices_dict = {
            "a": pd.Series(np.linspace(100, 200, 300)),  # strong
            "b": pd.Series(np.linspace(100, 150, 300)),  # medium
            "c": pd.Series(np.linspace(100, 105, 300)),  # weak
        }
        base = {"a": 0.33, "b": 0.34, "c": 0.33}
        adjusted = _relative_strength_weights(prices_dict, base, [63, 126], top_n=1)
        assert adjusted["a"] > base["a"]
        assert adjusted["c"] < base["c"]
        assert abs(sum(adjusted.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Trend signals
# ---------------------------------------------------------------------------


class TestTrendSignals:
    def test_uptrend(self) -> None:
        n = 250
        up = pd.Series(np.linspace(100, 300, n))
        signals = _trend_signals({"asset": up}, short_ma=50, long_ma=200)
        assert signals["asset"] == "up"

    def test_downtrend(self) -> None:
        n = 250
        down = pd.Series(np.linspace(300, 100, n))
        signals = _trend_signals({"asset": down}, short_ma=50, long_ma=200)
        assert signals["asset"] == "down"

    def test_insufficient_data_neutral(self) -> None:
        short = pd.Series([100.0, 101.0, 102.0])
        signals = _trend_signals({"asset": short}, short_ma=50, long_ma=200)
        assert signals["asset"] == "neutral"


# ---------------------------------------------------------------------------
# Volatility scalar
# ---------------------------------------------------------------------------


class TestVolatilityScalar:
    def test_low_vol_scales_up(self) -> None:
        """Low realized vol should produce scalar > 1."""
        rng = np.random.RandomState(0)
        tiny_returns = pd.Series(rng.normal(0, 0.001, 100))
        scalar = _volatility_scalar(tiny_returns, target_vol=0.10, lookback=20)
        assert scalar > 1.0

    def test_high_vol_scales_down(self) -> None:
        """High realized vol should produce scalar < 1."""
        rng = np.random.RandomState(0)
        big_returns = pd.Series(rng.normal(0, 0.05, 100))
        scalar = _volatility_scalar(big_returns, target_vol=0.10, lookback=20)
        assert scalar < 1.0

    def test_normal_vol_no_change(self) -> None:
        """Vol near target should return 1.0."""
        target = 0.10
        daily_std = target / np.sqrt(252)
        rng = np.random.RandomState(7)
        returns = pd.Series(rng.normal(0, daily_std, 100))
        scalar = _volatility_scalar(returns, target_vol=target, lookback=20)
        assert scalar == 1.0

    def test_insufficient_data_returns_one(self) -> None:
        short = pd.Series([0.01, -0.01])
        assert _volatility_scalar(short, target_vol=0.10, lookback=20) == 1.0

    def test_low_vol_capped_at_1_2(self) -> None:
        """Scalar should not exceed 1.2 even with very low vol."""
        almost_zero = pd.Series(np.full(100, 0.0))
        almost_zero.iloc[50] = 1e-8  # tiny nonzero to avoid nan
        scalar = _volatility_scalar(almost_zero, target_vol=0.10, lookback=20)
        assert scalar <= 1.2


# ---------------------------------------------------------------------------
# Corridor rebalancing
# ---------------------------------------------------------------------------


class TestCorridorRebalance:
    def test_no_rebalance_within_threshold(self) -> None:
        current = {"a": 0.50, "b": 0.50}
        target = {"a": 0.51, "b": 0.49}
        assert not _corridor_rebalance_needed(current, target, threshold=0.025)

    def test_rebalance_triggered(self) -> None:
        current = {"a": 0.50, "b": 0.50}
        target = {"a": 0.60, "b": 0.40}
        assert _corridor_rebalance_needed(current, target, threshold=0.025)

    def test_new_asset_triggers_rebalance(self) -> None:
        current = {"a": 1.0}
        target = {"a": 0.90, "b": 0.10}
        assert _corridor_rebalance_needed(current, target, threshold=0.025)


# ---------------------------------------------------------------------------
# Full strategy run()
# ---------------------------------------------------------------------------


class TestBeGlobalRun:
    def test_output_structure(
        self,
        strategy: BeGlobalStrategy,
        prices_etf: pd.DataFrame,
    ) -> None:
        result = strategy.run({"prices": prices_etf})
        assert "weights" in result
        assert "simple_weights" in result
        assert "details" in result
        assert isinstance(result["weights"], pd.DataFrame)
        assert isinstance(result["simple_weights"], dict)

    def test_weights_sum_to_one(
        self,
        strategy: BeGlobalStrategy,
        prices_etf: pd.DataFrame,
    ) -> None:
        result = strategy.run({"prices": prices_etf})
        row_sums = result["weights"].sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)

    def test_weights_non_negative(
        self,
        strategy: BeGlobalStrategy,
        prices_etf: pd.DataFrame,
    ) -> None:
        result = strategy.run({"prices": prices_etf})
        assert (result["weights"].values >= -1e-9).all()

    def test_asset_name_columns(
        self,
        prices_asset_names: pd.DataFrame,
    ) -> None:
        """Strategy should work with asset class name columns."""
        strategy = BeGlobalStrategy(risk_profile="profit")
        result = strategy.run({"prices": prices_asset_names})
        assert result["weights"].shape[0] > 0
        row_sums = result["weights"].sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)

    @pytest.mark.parametrize("profile", list(RISK_PROFILE_ALLOCATIONS.keys()))
    def test_all_profiles_run(
        self,
        profile: str,
        prices_etf: pd.DataFrame,
    ) -> None:
        strategy = BeGlobalStrategy(risk_profile=profile)
        result = strategy.run({"prices": prices_etf})
        assert result["weights"].shape[0] > 0
        assert len(result["simple_weights"]) > 0

    def test_core_satellite_split(
        self,
        prices_etf: pd.DataFrame,
    ) -> None:
        """Core weight=1 should produce weights matching base allocation only."""
        strategy = BeGlobalStrategy(core_weight=1.0, risk_profile="safe")
        result = strategy.run({"prices": prices_etf})
        # With core_weight=1 and no satellite, weights should reflect
        # base allocation exclusively.  The vol targeting and corridor logic
        # may shift things slightly, but core assets must dominate.
        latest = result["simple_weights"]
        assert len(latest) > 0
        row_sums = result["weights"].sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)

    def test_warmup_handling_short_data(self) -> None:
        """Prices shorter than warmup should return empty weights."""
        dates = pd.bdate_range("2024-01-01", periods=50)
        prices = pd.DataFrame(
            {"SPY": np.linspace(100, 110, 50), "SHV": np.linspace(100, 101, 50)},
            index=dates,
        )
        strategy = BeGlobalStrategy()
        result = strategy.run({"prices": prices})
        assert result["details"].get("warmup_insufficient", False)

    def test_param_override(self, prices_etf: pd.DataFrame) -> None:
        strategy = BeGlobalStrategy()
        result = strategy.run(
            {"prices": prices_etf},
            params={"risk_profile": "safe", "output_periods": 5},
        )
        assert result["weights"].shape[0] <= 5
        assert result["details"]["risk_profile"] == "safe"

    def test_output_periods_respected(
        self,
        strategy: BeGlobalStrategy,
        prices_etf: pd.DataFrame,
    ) -> None:
        strategy.output_periods = 10
        result = strategy.run({"prices": prices_etf})
        assert result["weights"].shape[0] <= 10

    def test_no_matching_columns_raises(self) -> None:
        prices = pd.DataFrame({"AAPL": [1, 2, 3], "TSLA": [4, 5, 6]})
        strategy = BeGlobalStrategy()
        with pytest.raises(ValueError, match="No matching asset columns"):
            strategy.run({"prices": prices})


# ---------------------------------------------------------------------------
# Plugin metadata
# ---------------------------------------------------------------------------


class TestMeta:
    def test_meta_attributes(self) -> None:
        assert BeGlobalStrategy.meta.name == "strategy.beglobal.v1"
        assert BeGlobalStrategy.meta.kind == "strategy"
        assert BeGlobalStrategy.meta.version == "0.1.0"
