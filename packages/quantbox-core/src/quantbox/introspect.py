"""Plugin introspection utilities for LLM agents.

Provides a universal `describe_plugin()` function that works with any plugin,
whether or not it has a custom `describe()` method. This module makes
every plugin introspectable by LLM agents without requiring changes to
individual plugin files.

Usage:
    from quantbox.introspect import describe_plugin, describe_registry

    # Single plugin
    info = describe_plugin(my_strategy)

    # Full registry
    catalog = describe_registry(registry)
"""

from __future__ import annotations

import dataclasses
from typing import Any


def describe_plugin(plugin: Any) -> dict[str, Any]:
    """Describe any plugin for LLM consumption.

    Tries plugin.describe() first (custom implementation).
    Falls back to universal introspection via PluginMeta + dataclass fields.

    Returns a dict with standardized keys:
        - name: Plugin ID (e.g. "strategy.crypto_trend.v1")
        - kind: Plugin type (strategy, broker, etc.)
        - description: Human-readable description
        - version: Semver version
        - tags: Searchable tags
        - capabilities: Supported modes/features
        - parameters: Current parameter values (from dataclass fields)
        - params_schema: JSON Schema for params (if available)
        - examples: Config snippets (if available)
        - custom: Output of plugin.describe() (if it has one)
    """
    meta = getattr(plugin, "meta", None)
    result: dict[str, Any] = {}

    if meta:
        result["name"] = getattr(meta, "name", "unknown")
        result["kind"] = getattr(meta, "kind", "unknown")
        result["description"] = getattr(meta, "description", "")
        result["version"] = getattr(meta, "version", "0.0.0")
        result["tags"] = list(getattr(meta, "tags", ()))
        result["capabilities"] = list(getattr(meta, "capabilities", ()))
        result["inputs"] = list(getattr(meta, "inputs", ()))
        result["outputs"] = list(getattr(meta, "outputs", ()))

        schema = getattr(meta, "params_schema", None)
        if schema:
            result["params_schema"] = schema

        examples = getattr(meta, "examples", ())
        if examples:
            result["examples"] = list(examples)

    # Extract current parameter values from dataclass fields
    if dataclasses.is_dataclass(plugin) and not isinstance(plugin, type):
        params = {}
        for f in dataclasses.fields(plugin):
            if f.name.startswith("_"):
                continue
            val = getattr(plugin, f.name)
            # Only include JSON-serializable values
            if isinstance(val, (str, int, float, bool, list, tuple, dict, type(None))):
                params[f.name] = val
        if params:
            result["parameters"] = params

    # Include custom describe() output if available
    if hasattr(plugin, "describe") and callable(plugin.describe):
        try:
            result["custom"] = plugin.describe()
        except Exception:
            pass

    return result


def describe_plugin_class(cls: type) -> dict[str, Any]:
    """Describe a plugin class (not instance) for LLM consumption.

    Extracts PluginMeta and field defaults without instantiation.
    """
    meta = getattr(cls, "meta", None)
    result: dict[str, Any] = {}

    if meta:
        result["name"] = getattr(meta, "name", "unknown")
        result["kind"] = getattr(meta, "kind", "unknown")
        result["description"] = getattr(meta, "description", "")
        result["version"] = getattr(meta, "version", "0.0.0")
        result["tags"] = list(getattr(meta, "tags", ()))
        result["capabilities"] = list(getattr(meta, "capabilities", ()))
        result["inputs"] = list(getattr(meta, "inputs", ()))
        result["outputs"] = list(getattr(meta, "outputs", ()))

        schema = getattr(meta, "params_schema", None)
        if schema:
            result["params_schema"] = schema

        examples = getattr(meta, "examples", ())
        if examples:
            result["examples"] = list(examples)

    # Extract field defaults (without instantiating)
    if dataclasses.is_dataclass(cls):
        defaults = {}
        for f in dataclasses.fields(cls):
            if f.name.startswith("_"):
                continue
            if f.default is not dataclasses.MISSING:
                defaults[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:
                try:
                    defaults[f.name] = f.default_factory()
                except Exception:
                    defaults[f.name] = "<factory>"
        if defaults:
            result["parameter_defaults"] = defaults

    # List available methods
    methods = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        attr = getattr(cls, name, None)
        if callable(attr) and name not in ("meta",):
            methods.append(name)
    if methods:
        result["methods"] = methods

    return result


def describe_registry(registry: Any) -> dict[str, list[dict[str, Any]]]:
    """Describe all plugins in a registry for LLM consumption.

    Returns {kind: [plugin_info, ...]} dict.
    """
    catalog: dict[str, list[dict[str, Any]]] = {}

    for kind_attr, kind_label in [
        ("pipelines", "pipeline"),
        ("strategies", "strategy"),
        ("data", "data"),
        ("brokers", "broker"),
        ("risk", "risk"),
        ("rebalancing", "rebalancing"),
        ("publishers", "publisher"),
    ]:
        plugins = getattr(registry, kind_attr, {})
        entries = []
        for name, cls in sorted(plugins.items()):
            info = describe_plugin_class(cls)
            entries.append(info)
        if entries:
            catalog[kind_label] = entries

    return catalog
