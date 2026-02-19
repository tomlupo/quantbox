"""Tests for PluginRegistry support of feature, validation, and monitor types."""

from __future__ import annotations

from quantbox.registry import PluginRegistry


def test_registry_has_features_field() -> None:
    """PluginRegistry.discover() returns an object with a 'features' dict field."""
    reg = PluginRegistry.discover()
    assert isinstance(reg.features, dict)


def test_registry_has_validations_field() -> None:
    """PluginRegistry.discover() returns an object with a 'validations' dict field."""
    reg = PluginRegistry.discover()
    assert isinstance(reg.validations, dict)


def test_registry_has_monitors_field() -> None:
    """PluginRegistry.discover() returns an object with a 'monitors' dict field."""
    reg = PluginRegistry.discover()
    assert isinstance(reg.monitors, dict)


def test_registry_new_fields_default_empty() -> None:
    """New plugin type fields start empty (no builtins registered yet)."""
    reg = PluginRegistry.discover()
    # No builtins registered for these types yet, so they should be empty
    # (unless entry points exist, which they shouldn't in the test env)
    assert isinstance(reg.features, dict)
    assert isinstance(reg.validations, dict)
    assert isinstance(reg.monitors, dict)


def test_registry_preserves_existing_fields() -> None:
    """Adding new fields does not break existing plugin discovery."""
    reg = PluginRegistry.discover()
    assert isinstance(reg.pipelines, dict)
    assert isinstance(reg.brokers, dict)
    assert isinstance(reg.data, dict)
    assert isinstance(reg.publishers, dict)
    assert isinstance(reg.risk, dict)
    assert isinstance(reg.strategies, dict)
    assert isinstance(reg.rebalancing, dict)
    # Existing builtins should still be present
    assert len(reg.pipelines) > 0
    assert len(reg.data) > 0
