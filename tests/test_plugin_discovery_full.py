"""Tests for plugin discovery and registry completeness.

Validates that all built-in plugins are properly registered, have correct
metadata, and are discoverable through the PluginRegistry.
"""

from __future__ import annotations

import pytest

from quantbox.contracts import PluginMeta
from quantbox.plugins.builtins import builtins
from quantbox.registry import PluginRegistry

# ---------------------------------------------------------------------------
# Expected plugin types — the keys returned by builtins()
# ---------------------------------------------------------------------------

EXPECTED_PLUGIN_TYPES = frozenset({"pipeline", "data", "broker", "strategy", "rebalancing", "risk", "publisher"})

# ---------------------------------------------------------------------------
# Known plugin names that must always be present
# ---------------------------------------------------------------------------

KNOWN_PLUGIN_NAMES = {
    "trade.full_pipeline.v1",
    "strategy.crypto_trend.v1",
    "local_file_data",
    "sim.paper.v1",
}


def _all_plugins(registry: dict[str, dict[str, type]]):
    """Yield (type_key, meta_name, plugin_class) for every registered plugin."""
    for type_key, plugins in registry.items():
        for name, cls in plugins.items():
            yield type_key, name, cls


# ===================================================================
# TestBuiltins — structure of the builtins() return value
# ===================================================================


class TestBuiltins:
    """Verify the shape and keys of the builtins() dict."""

    def test_builtins_returns_all_plugin_types(self):
        """builtins() must return a dict with exactly the expected plugin-type keys."""
        result = builtins()
        assert isinstance(result, dict)
        assert set(result.keys()) == EXPECTED_PLUGIN_TYPES

    def test_each_plugin_type_has_at_least_one_plugin(self):
        """Every plugin type must contain at least one registered plugin."""
        result = builtins()
        for type_key, plugins in result.items():
            assert len(plugins) >= 1, f"Plugin type '{type_key}' has no plugins"

    def test_plugin_count_sanity(self):
        """There must be at least 20 total plugins registered."""
        result = builtins()
        total = sum(len(plugins) for plugins in result.values())
        assert total >= 20, f"Expected >= 20 total plugins, got {total}"

    def test_known_plugin_names_present(self):
        """Specific well-known plugin names must be found in the registry."""
        result = builtins()
        all_names: set[str] = set()
        for plugins in result.values():
            all_names.update(plugins.keys())
        for expected_name in KNOWN_PLUGIN_NAMES:
            assert expected_name in all_names, (
                f"Known plugin '{expected_name}' not found. Available: {sorted(all_names)}"
            )


# ===================================================================
# TestPluginMeta — metadata quality for every registered plugin
# ===================================================================


class TestPluginMeta:
    """Validate PluginMeta on every built-in plugin class."""

    @pytest.fixture()
    def all_plugins(self):
        return list(_all_plugins(builtins()))

    def test_every_plugin_has_meta_attribute(self, all_plugins):
        """Every plugin class must have a class-level 'meta' attribute."""
        for _type_key, _name, cls in all_plugins:
            assert hasattr(cls, "meta"), f"Plugin {cls.__name__} (type={_type_key}) is missing 'meta' attribute"

    def test_meta_is_plugin_meta_instance(self, all_plugins):
        """Every plugin's meta must be a PluginMeta instance."""
        for _type_key, _name, cls in all_plugins:
            assert isinstance(cls.meta, PluginMeta), (
                f"Plugin {cls.__name__}.meta is {type(cls.meta).__name__}, expected PluginMeta"
            )

    def test_meta_name_is_nonempty_string(self, all_plugins):
        """Every meta.name must be a non-empty string."""
        for _type_key, _name, cls in all_plugins:
            assert isinstance(cls.meta.name, str) and cls.meta.name.strip(), (
                f"Plugin {cls.__name__}.meta.name is empty or not a string"
            )

    def test_meta_version_is_set(self, all_plugins):
        """Every meta.version must be a non-empty string."""
        for _type_key, _name, cls in all_plugins:
            assert isinstance(cls.meta.version, str) and cls.meta.version.strip(), (
                f"Plugin {cls.__name__}.meta.version is empty or not a string"
            )

    def test_meta_description_is_nonempty(self, all_plugins):
        """Every meta.description should be a non-empty string."""
        for _type_key, _name, cls in all_plugins:
            assert isinstance(cls.meta.description, str) and cls.meta.description.strip(), (
                f"Plugin {cls.__name__}.meta.description is empty or not a string"
            )

    def test_no_duplicate_names_within_same_type(self):
        """Within each plugin type, meta.name values must be unique."""
        result = builtins()
        for type_key, plugins in result.items():
            seen_names: dict[str, str] = {}
            for _name, cls in plugins.items():
                meta_name = cls.meta.name
                assert meta_name not in seen_names, (
                    f"Duplicate meta.name '{meta_name}' in type '{type_key}': "
                    f"found on both {seen_names[meta_name]} and {cls.__name__}"
                )
                seen_names[meta_name] = cls.__name__

    def test_dict_key_matches_meta_name(self):
        """The builtins dict key must equal the plugin's meta.name."""
        result = builtins()
        for type_key, plugins in result.items():
            for dict_key, cls in plugins.items():
                assert dict_key == cls.meta.name, (
                    f"Dict key '{dict_key}' != meta.name '{cls.meta.name}' for {cls.__name__} in type '{type_key}'"
                )


# ===================================================================
# TestPluginRegistry — PluginRegistry.discover() completeness
# ===================================================================


class TestPluginRegistry:
    """Verify PluginRegistry.discover() returns the same plugins as builtins()."""

    @pytest.fixture()
    def registry(self):
        return PluginRegistry.discover()

    def test_discover_returns_plugin_registry(self, registry):
        """discover() must return a PluginRegistry instance."""
        assert isinstance(registry, PluginRegistry)

    def test_registry_contains_all_builtin_plugins(self, registry):
        """Every builtin plugin name must appear in the discovered registry."""
        b = builtins()
        # Map builtins type keys to PluginRegistry field names
        field_map = {
            "pipeline": "pipelines",
            "data": "data",
            "broker": "brokers",
            "strategy": "strategies",
            "rebalancing": "rebalancing",
            "risk": "risk",
            "publisher": "publishers",
        }
        for type_key, expected_plugins in b.items():
            field_name = field_map[type_key]
            registry_dict = getattr(registry, field_name)
            for plugin_name in expected_plugins:
                assert plugin_name in registry_dict, (
                    f"Builtin plugin '{plugin_name}' (type={type_key}) "
                    f"not found in registry.{field_name}. "
                    f"Available: {sorted(registry_dict.keys())}"
                )

    def test_registry_plugin_count_matches_builtins(self, registry):
        """When no entry-point plugins are installed, registry counts should
        be at least as large as builtins counts (entry points can only add)."""
        b = builtins()
        field_map = {
            "pipeline": "pipelines",
            "data": "data",
            "broker": "brokers",
            "strategy": "strategies",
            "rebalancing": "rebalancing",
            "risk": "risk",
            "publisher": "publishers",
        }
        for type_key, expected_plugins in b.items():
            field_name = field_map[type_key]
            registry_dict = getattr(registry, field_name)
            assert len(registry_dict) >= len(expected_plugins), (
                f"Registry has fewer {type_key} plugins ({len(registry_dict)}) than builtins ({len(expected_plugins)})"
            )
