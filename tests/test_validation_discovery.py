"""Tests for validation plugin discovery.

Verifies all 5 validation plugins are registered and discoverable
through the PluginRegistry.
"""

from __future__ import annotations

from quantbox.registry import PluginRegistry


EXPECTED_VALIDATION_PLUGINS = [
    "validation.walk_forward.v1",
    "validation.statistical.v1",
    "validation.turnover.v1",
    "validation.regime.v1",
    "validation.benchmark.v1",
]


def test_all_validation_plugins_discoverable() -> None:
    reg = PluginRegistry.discover()

    for name in EXPECTED_VALIDATION_PLUGINS:
        assert name in reg.validations, (
            f"Validation plugin '{name}' not found in registry. "
            f"Available: {list(reg.validations.keys())}"
        )


def test_validation_plugin_count() -> None:
    reg = PluginRegistry.discover()

    assert len(reg.validations) >= len(EXPECTED_VALIDATION_PLUGINS), (
        f"Expected at least {len(EXPECTED_VALIDATION_PLUGINS)} validation plugins, "
        f"found {len(reg.validations)}"
    )


def test_validation_plugins_have_correct_kind() -> None:
    reg = PluginRegistry.discover()

    for name in EXPECTED_VALIDATION_PLUGINS:
        plugin_cls = reg.validations[name]
        assert plugin_cls.meta.kind == "validation", (
            f"Plugin '{name}' has kind '{plugin_cls.meta.kind}', expected 'validation'"
        )


def test_validation_plugins_are_instantiable() -> None:
    reg = PluginRegistry.discover()

    for name in EXPECTED_VALIDATION_PLUGINS:
        plugin_cls = reg.validations[name]
        instance = plugin_cls()
        assert hasattr(instance, "validate"), (
            f"Plugin '{name}' instance does not have a 'validate' method"
        )
