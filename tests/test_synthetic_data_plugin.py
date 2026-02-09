"""Tests for SyntheticDataPlugin — data.synthetic.v1."""
import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.datasources.synthetic_data import SyntheticDataPlugin


@pytest.fixture
def plugin():
    return SyntheticDataPlugin()


class TestMeta:
    def test_name(self):
        assert SyntheticDataPlugin.meta.name == "data.synthetic.v1"

    def test_kind(self):
        assert SyntheticDataPlugin.meta.kind == "data"


class TestLoadUniverse:
    def test_default_symbols(self, plugin):
        universe = plugin.load_universe({"n_assets": 5})
        assert len(universe) == 5
        assert universe[0] == "SYN_001"

    def test_explicit_symbols(self, plugin):
        universe = plugin.load_universe({"symbols": ["BTC", "ETH", "SOL"]})
        assert universe == ["BTC", "ETH", "SOL"]


class TestLoadMarketData:
    def test_output_format(self, plugin):
        universe = plugin.load_universe({"n_assets": 3})
        data = plugin.load_market_data(universe, "2026-02-01", {
            "n_assets": 3, "n_steps": 50, "random_state": 42,
        })
        assert "prices" in data
        assert "volume" in data
        assert isinstance(data["prices"], pd.DataFrame)
        assert isinstance(data["volume"], pd.DataFrame)

    def test_wide_format(self, plugin):
        universe = ["A", "B", "C"]
        data = plugin.load_market_data(universe, "2026-02-01", {
            "n_steps": 30, "symbols": ["A", "B", "C"], "random_state": 42,
        })
        assert list(data["prices"].columns) == ["A", "B", "C"]
        assert len(data["prices"]) == 31  # n_steps + 1

    def test_date_index(self, plugin):
        universe = plugin.load_universe({"n_assets": 2})
        data = plugin.load_market_data(universe, "2026-02-01", {
            "n_assets": 2, "n_steps": 10, "random_state": 42,
        })
        assert isinstance(data["prices"].index, pd.DatetimeIndex)
        # Last date should be on or before asof
        assert data["prices"].index[-1] <= pd.Timestamp("2026-02-01")

    def test_positive_prices(self, plugin):
        universe = plugin.load_universe({"n_assets": 5})
        data = plugin.load_market_data(universe, "2026-02-01", {
            "n_assets": 5, "n_steps": 100, "random_state": 42,
        })
        assert (data["prices"] > 0).all().all()

    def test_models(self, plugin):
        for model in ["gbm", "jump_diffusion", "mean_reversion"]:
            universe = plugin.load_universe({"n_assets": 2})
            data = plugin.load_market_data(universe, "2026-02-01", {
                "n_assets": 2, "n_steps": 20, "model": model, "random_state": 42,
            })
            assert "prices" in data

    def test_correlation_types(self, plugin):
        for corr_type in ["identity", "random", "stressed"]:
            universe = plugin.load_universe({"n_assets": 3})
            data = plugin.load_market_data(universe, "2026-02-01", {
                "n_assets": 3, "n_steps": 20, "correlation": corr_type, "random_state": 42,
            })
            assert "prices" in data

    def test_deterministic(self, plugin):
        universe = plugin.load_universe({"n_assets": 3})
        params = {"n_assets": 3, "n_steps": 30, "random_state": 42}
        d1 = plugin.load_market_data(universe, "2026-02-01", params)
        d2 = plugin.load_market_data(universe, "2026-02-01", params)
        pd.testing.assert_frame_equal(d1["prices"], d2["prices"])
