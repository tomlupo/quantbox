"""Tests that feature plugins are discoverable via PluginRegistry."""

from __future__ import annotations

from quantbox.registry import PluginRegistry


class TestFeatureDiscovery:
    """Verify both feature plugins appear in the registry."""

    def test_technical_features_in_registry(self) -> None:
        reg = PluginRegistry.discover()
        assert "features.technical.v1" in reg.features

    def test_cross_sectional_features_in_registry(self) -> None:
        reg = PluginRegistry.discover()
        assert "features.cross_sectional.v1" in reg.features

    def test_feature_plugins_are_classes(self) -> None:
        reg = PluginRegistry.discover()
        for name, cls in reg.features.items():
            assert hasattr(cls, "meta"), f"{name} missing meta attribute"
            assert hasattr(cls, "compute"), f"{name} missing compute method"
