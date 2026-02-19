"""Tests for FeaturePlugin, ValidationPlugin, and MonitorPlugin protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quantbox.contracts import (
    FeaturePlugin,
    MonitorPlugin,
    PluginMeta,
    RunResult,
    ValidationPlugin,
)


# ---------------------------------------------------------------------------
# Stub implementations
# ---------------------------------------------------------------------------


@dataclass
class StubFeature:
    meta: PluginMeta = PluginMeta(name="stub.feature.v1", kind="feature", version="0.1.0", core_compat=">=0.1.0")

    def compute(self, data: dict[str, pd.DataFrame], params: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame({"value": [1.0, 2.0]})


@dataclass
class StubValidation:
    meta: PluginMeta = PluginMeta(name="stub.validation.v1", kind="validation", version="0.1.0", core_compat=">=0.1.0")

    def validate(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        benchmark: pd.DataFrame | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        return {"passed": True, "findings": []}


@dataclass
class StubMonitor:
    meta: PluginMeta = PluginMeta(name="stub.monitor.v1", kind="monitor", version="0.1.0", core_compat=">=0.1.0")

    def check(
        self,
        result: RunResult,
        history: list[RunResult] | None,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_feature_plugin_structural_subtype() -> None:
    """StubFeature satisfies the FeaturePlugin protocol."""
    plugin: FeaturePlugin = StubFeature()
    data = {"prices": pd.DataFrame({"A": [1.0]})}
    result = plugin.compute(data, {})
    assert isinstance(result, pd.DataFrame)


def test_validation_plugin_structural_subtype() -> None:
    """StubValidation satisfies the ValidationPlugin protocol."""
    plugin: ValidationPlugin = StubValidation()
    result = plugin.validate(
        returns=pd.DataFrame(),
        weights=pd.DataFrame(),
        benchmark=None,
        params={},
    )
    assert isinstance(result, dict)
    assert result["passed"] is True


def test_monitor_plugin_structural_subtype() -> None:
    """StubMonitor satisfies the MonitorPlugin protocol."""
    plugin: MonitorPlugin = StubMonitor()
    run_result = RunResult(
        run_id="test-001",
        pipeline_name="test.pipe.v1",
        mode="backtest",
        asof="2026-01-01",
        artifacts={},
        metrics={},
        notes={},
    )
    alerts = plugin.check(run_result, history=None, params={})
    assert alerts == []


def test_feature_plugin_meta_kind() -> None:
    """FeaturePlugin stub has kind='feature' in its meta."""
    plugin = StubFeature()
    assert plugin.meta.kind == "feature"


def test_validation_plugin_meta_kind() -> None:
    """ValidationPlugin stub has kind='validation' in its meta."""
    plugin = StubValidation()
    assert plugin.meta.kind == "validation"


def test_monitor_plugin_meta_kind() -> None:
    """MonitorPlugin stub has kind='monitor' in its meta."""
    plugin = StubMonitor()
    assert plugin.meta.kind == "monitor"
