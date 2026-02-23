"""Tests for monitor plugin discovery via builtins registry.

Verifies that both monitor plugins are registered and discoverable.
"""

from __future__ import annotations

from quantbox.plugins.builtins import builtins
from quantbox.plugins.monitor.drawdown import DrawdownMonitor
from quantbox.plugins.monitor.signal_decay import SignalDecayMonitor


class TestMonitorDiscovery:
    """Verify monitor plugins appear in the builtins registry."""

    def test_monitor_section_exists(self) -> None:
        registry = builtins()
        assert "monitor" in registry

    def test_drawdown_monitor_registered(self) -> None:
        registry = builtins()
        assert "monitor.drawdown.v1" in registry["monitor"]
        assert registry["monitor"]["monitor.drawdown.v1"] is DrawdownMonitor

    def test_signal_decay_monitor_registered(self) -> None:
        registry = builtins()
        assert "monitor.signal_decay.v1" in registry["monitor"]
        assert registry["monitor"]["monitor.signal_decay.v1"] is SignalDecayMonitor

    def test_monitor_count(self) -> None:
        registry = builtins()
        assert len(registry["monitor"]) == 2

    def test_monitor_init_exports(self) -> None:
        from quantbox.plugins.monitor import DrawdownMonitor as DD
        from quantbox.plugins.monitor import SignalDecayMonitor as SD

        assert DD is DrawdownMonitor
        assert SD is SignalDecayMonitor
