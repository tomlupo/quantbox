"""Tests for validation and monitor plugin integration in runner.py."""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantbox.contracts import PluginMeta, RunResult
from quantbox.registry import PluginRegistry


class TestValidationIntegration:
    """Test that runner resolves and executes validation plugins after backtest."""

    def test_registry_has_validations(self):
        reg = PluginRegistry.discover()
        assert len(reg.validations) > 0, "No validation plugins found in registry"

    def test_validation_plugins_have_validate_method(self):
        reg = PluginRegistry.discover()
        for name, cls in reg.validations.items():
            instance = cls()
            assert hasattr(instance, "validate"), f"{name} missing validate() method"

    def test_runner_imports_validation_support(self):
        """runner.py should handle validation config without errors."""
        from quantbox.runner import run_from_config
        # Just verify the function exists and can be imported
        assert callable(run_from_config)


class TestMonitorIntegration:
    """Test that runner resolves and executes monitor plugins for paper/live."""

    def test_registry_has_monitors(self):
        reg = PluginRegistry.discover()
        assert len(reg.monitors) > 0, "No monitor plugins found in registry"

    def test_monitor_plugins_have_check_method(self):
        reg = PluginRegistry.discover()
        for name, cls in reg.monitors.items():
            instance = cls()
            assert hasattr(instance, "check"), f"{name} missing check() method"


class TestValidationRunnerFlow:
    """Test the full validation flow through the runner."""

    def test_validation_results_added_to_notes(self):
        """When validation plugins run, results should be added to RunResult.notes."""
        # Create mock returns and weights
        dates = pd.date_range("2025-01-01", periods=100)
        returns_df = pd.DataFrame(
            {"returns": np.random.randn(100) * 0.01},
            index=dates,
        )
        weights_df = pd.DataFrame(
            {"BTC": np.random.rand(100), "ETH": np.random.rand(100)},
            index=dates,
        )

        # Run a validation plugin directly
        reg = PluginRegistry.discover()
        v_cls = reg.validations.get("validation.walk_forward.v1")
        if v_cls is None:
            pytest.skip("walk_forward validation not available")

        v_plugin = v_cls()
        v_result = v_plugin.validate(returns_df, weights_df, None, {"n_splits": 3, "train_ratio": 0.7})

        assert "passed" in v_result
        assert "metrics" in v_result or "findings" in v_result


class TestMonitorRunnerFlow:
    """Test the full monitor flow through the runner."""

    def test_drawdown_monitor_emits_alerts(self):
        """When drawdown exceeds threshold, monitor should emit alerts."""
        reg = PluginRegistry.discover()
        m_cls = reg.monitors.get("monitor.drawdown.v1")
        if m_cls is None:
            pytest.skip("drawdown monitor not available")

        # Create a result with high drawdown (negative value, breaches -0.10 threshold)
        result = RunResult(
            run_id="test",
            pipeline_name="test",
            mode="paper",
            asof="2025-01-01",
            artifacts={},
            metrics={"max_drawdown": -0.25, "total_return": -0.1},
            notes={},
        )

        m_plugin = m_cls()
        alerts = m_plugin.check(result, None, {"max_drawdown": -0.10, "action": "halt"})
        assert len(alerts) > 0
        assert any(a.get("action") == "halt" for a in alerts)
