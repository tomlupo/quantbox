"""Tests for quantbox.agents and quantbox.introspect modules."""

from __future__ import annotations

import pytest

from quantbox.agents.tools import QuantBoxAgent
from quantbox.introspect import describe_plugin, describe_plugin_class, describe_registry
from quantbox.plugins.builtins import builtins
from quantbox.registry import PluginRegistry


@pytest.fixture
def agent():
    return QuantBoxAgent()


@pytest.fixture
def registry():
    return PluginRegistry.discover()


# --------------------------------------------------------------------------
# introspect.py
# --------------------------------------------------------------------------


class TestDescribePluginClass:
    def test_strategy_class(self):
        plugins = builtins()
        for name, cls in plugins["strategy"].items():
            info = describe_plugin_class(cls)
            assert info["name"] == name
            assert info["kind"] == "strategy"
            assert "description" in info
            assert len(info["description"]) > 0

    def test_broker_class(self):
        plugins = builtins()
        for name, cls in plugins["broker"].items():
            info = describe_plugin_class(cls)
            assert info["name"] == name
            assert info["kind"] == "broker"

    def test_all_plugins_describable(self):
        """Every registered plugin class must be describable."""
        plugins = builtins()
        for kind, plugin_map in plugins.items():
            for name, cls in plugin_map.items():
                info = describe_plugin_class(cls)
                assert "name" in info, f"{name} missing 'name'"
                assert "kind" in info, f"{name} missing 'kind'"
                assert "description" in info, f"{name} missing 'description'"


class TestDescribePlugin:
    def test_strategy_instance(self):
        plugins = builtins()
        # Use a strategy that is cheap to instantiate
        cls = plugins["strategy"]["strategy.weighted_avg.v1"]
        instance = cls()
        info = describe_plugin(instance)
        assert info["name"] == "strategy.weighted_avg.v1"
        assert info["kind"] == "strategy"

    def test_broker_instance(self):
        plugins = builtins()
        cls = plugins["broker"]["sim.paper.v1"]
        instance = cls()
        info = describe_plugin(instance)
        assert info["name"] == "sim.paper.v1"
        assert "parameters" in info
        assert "cash" in info["parameters"]


class TestDescribeRegistry:
    def test_all_kinds_present(self, registry):
        catalog = describe_registry(registry)
        for kind in ("pipeline", "strategy", "data", "broker", "risk", "rebalancing", "publisher"):
            assert kind in catalog, f"Missing kind: {kind}"
            assert len(catalog[kind]) > 0


# --------------------------------------------------------------------------
# agents/tools.py
# --------------------------------------------------------------------------


class TestQuantBoxAgent:
    def test_list_plugins(self, agent):
        result = agent.list_plugins()
        assert "strategy" in result
        assert "broker" in result
        assert len(result["strategy"]) >= 9

    def test_list_plugins_filtered(self, agent):
        result = agent.list_plugins(kind="pipeline")
        assert "pipeline" in result
        assert "strategy" not in result

    def test_plugin_info(self, agent):
        info = agent.plugin_info("strategy.crypto_trend.v1")
        assert info["name"] == "strategy.crypto_trend.v1"
        assert info["kind"] == "strategy"
        assert "methods" in info

    def test_plugin_info_not_found(self, agent):
        info = agent.plugin_info("nonexistent.v1")
        assert "error" in info
        assert "available" in info

    def test_search_plugins(self, agent):
        matches = agent.search_plugins("trend")
        assert len(matches) >= 2
        names = [m["name"] for m in matches]
        assert "strategy.crypto_trend.v1" in names

    def test_search_plugins_by_tag(self, agent):
        matches = agent.search_plugins("futures")
        assert len(matches) >= 1

    def test_build_config(self, agent):
        config = agent.build_config(
            mode="backtest",
            pipeline="backtest.pipeline.v1",
            strategy="strategy.crypto_trend.v1",
            data="binance.live_data.v1",
        )
        assert config["run"]["mode"] == "backtest"
        assert config["plugins"]["pipeline"]["name"] == "backtest.pipeline.v1"
        assert len(config["plugins"]["strategies"]) == 1
        assert config["plugins"]["strategies"][0]["name"] == "strategy.crypto_trend.v1"

    def test_build_config_with_broker(self, agent):
        config = agent.build_config(
            mode="paper",
            pipeline="trade.full_pipeline.v1",
            strategy="strategy.crypto_trend.v1",
            data="binance.live_data.v1",
            broker="sim.paper.v1",
            risk="risk.trading_basic.v1",
        )
        assert config["plugins"]["broker"]["name"] == "sim.paper.v1"
        assert len(config["plugins"]["risk"]) == 1

    def test_validate_config(self, agent):
        config = agent.build_config(
            mode="backtest",
            pipeline="backtest.pipeline.v1",
            data="binance.live_data.v1",
        )
        result = agent.validate_config(config)
        assert "valid" in result
        assert "findings" in result

    def test_list_profiles(self, agent):
        profiles = agent.list_profiles()
        assert "research" in profiles
        assert "trading" in profiles

    def test_inspect_run_not_found(self, agent):
        result = agent.inspect_run("/nonexistent/path")
        assert "error" in result
