"""L1 pipeline smoke tests.

Guards the gap between "pytest is green" and "pipeline works in production":

  1. **Plugin discovery** — every registered plugin loads cleanly
  2. **Metadata schema** — every plugin has the required ``meta`` attributes
  3. **Plugin map invariants** — no duplicate names, builtins all registered
  4. **Negative paths** — unknown plugin names fail loudly, not silently

These tests run in <1 second and catch the most common agent-introduced
failure: "added a strategy file but forgot to wire it into builtins.py /
manifest.yaml". They are the cheapest layer of the pipeline-correctness
ladder.

A future expansion (deferred — requires bundling a tiny parquet fixture
dataset) will add full end-to-end backtest invocations that exercise
strategy → backtest → manifest. That's a stronger guarantee but ~10x
the runtime and one more thing to maintain.

Marked ``@pytest.mark.pipeline_smoke`` so default ``pytest -q`` skips them.
CI runs ``pytest -m pipeline_smoke`` as a separate job.
"""

from __future__ import annotations

import pytest

from quantbox.contracts import PluginMeta
from quantbox.registry import PluginRegistry


@pytest.fixture(scope="module")
def registry() -> PluginRegistry:
    """Single discovery pass shared by all tests in this module."""
    return PluginRegistry.discover()


# ----------------------------------------------------------------------
# Plugin discovery
# ----------------------------------------------------------------------


@pytest.mark.pipeline_smoke
def test_discovery_finds_strategies(registry: PluginRegistry) -> None:
    """The registry must surface strategy plugins."""
    assert registry.strategies, "no strategies registered — builtins wiring broken"
    # Touchstone plugins that should always exist.
    assert "strategy.static_weights.v1" in registry.strategies
    assert "strategy.crypto_regime_trend.v1" in registry.strategies


@pytest.mark.pipeline_smoke
def test_discovery_finds_pipelines(registry: PluginRegistry) -> None:
    """The backtest pipeline is the load-bearing entry point."""
    assert "backtest.pipeline.v1" in registry.pipelines


@pytest.mark.pipeline_smoke
def test_discovery_finds_data_plugins(registry: PluginRegistry) -> None:
    """At least the synthetic + dataset.curated data plugins must register."""
    assert registry.data, "no data plugins registered"
    assert "data.synthetic.v1" in registry.data


# ----------------------------------------------------------------------
# Metadata schema
# ----------------------------------------------------------------------


@pytest.mark.pipeline_smoke
@pytest.mark.parametrize(
    "group_name", ["strategies", "pipelines", "data", "brokers", "rebalancing", "risk", "overlays"]
)
def test_every_plugin_has_well_formed_meta(registry: PluginRegistry, group_name: str) -> None:
    """Every plugin must declare ``meta = PluginMeta(...)`` with required fields.

    Catches: agent adds a new plugin class but forgets the meta attribute,
    or uses a stale meta from a copy-paste. The discovery will succeed
    but downstream consumers (CLI, lifecycle audits, --strict mode) break.
    """
    group = getattr(registry, group_name)
    for name, cls in group.items():
        meta = getattr(cls, "meta", None)
        assert meta is not None, f"{name}: missing meta attribute"
        assert isinstance(meta, PluginMeta), f"{name}: meta is not PluginMeta"
        assert meta.name, f"{name}: meta.name is empty"
        assert meta.kind, f"{name}: meta.kind is empty"
        assert meta.version, f"{name}: meta.version is empty"
        assert isinstance(meta.status, str), f"{name}: meta.status not a string"
        # Plugin's registered key must match its declared name.
        assert meta.name == name, f"registry key {name!r} != meta.name {meta.name!r}"


# ----------------------------------------------------------------------
# Plugin map invariants
# ----------------------------------------------------------------------


@pytest.mark.pipeline_smoke
def test_no_duplicate_plugin_names_across_groups(registry: PluginRegistry) -> None:
    """A plugin name must be unique across all kinds.

    Catches: agent registers ``strategy.foo.v1`` in two builtins maps,
    or accidentally collides with an existing entry-point.
    """
    all_names: list[tuple[str, str]] = []
    for group_name in ("strategies", "pipelines", "data", "brokers", "rebalancing", "risk", "overlays"):
        group = getattr(registry, group_name)
        for name in group:
            all_names.append((name, group_name))

    seen: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []
    for name, group in all_names:
        if name in seen:
            duplicates.append((name, seen[name], group))
        else:
            seen[name] = group
    assert not duplicates, f"duplicate plugin names across groups: {duplicates}"


@pytest.mark.pipeline_smoke
def test_all_builtins_are_discovered(registry: PluginRegistry) -> None:
    """Every plugin in the builtins map must appear in the registry after discover().

    Catches: agent adds a class to plugins/strategies/ but forgets to
    add it to plugins/builtins.py — the registry has no idea it exists,
    runs fail with PluginNotFoundError at runtime instead of CI time.
    """
    from quantbox.plugins.builtins import builtins as builtin_plugins

    builtins = builtin_plugins()
    group_to_attr = {
        "strategy": "strategies",
        "pipeline": "pipelines",
        "data": "data",
        "broker": "brokers",
        "rebalancing": "rebalancing",
        "risk": "risk",
        "publisher": "publishers",
        "feature": "features",
        "validation": "validations",
        "monitor": "monitors",
        "overlay": "overlays",
    }
    for group_key, registry_attr in group_to_attr.items():
        b = builtins.get(group_key, {})
        r = getattr(registry, registry_attr, {})
        missing = set(b.keys()) - set(r.keys())
        assert not missing, f"{group_key}: builtins not in registry: {missing}"


# ----------------------------------------------------------------------
# Negative paths
# ----------------------------------------------------------------------


@pytest.mark.pipeline_smoke
def test_unknown_plugin_raises_loudly(registry: PluginRegistry) -> None:
    """Asking for a nonexistent plugin must raise — never silently return None or empty."""
    from quantbox.exceptions import PluginNotFoundError
    from quantbox.runner import _resolve_plugin_cls

    with pytest.raises(PluginNotFoundError):
        _resolve_plugin_cls(
            {"name": "strategy.does_not_exist.v999"},
            registry.strategies,
            "strategy",
            mode="backtest",
        )


# ----------------------------------------------------------------------
# The three registration places agree (TOM-1341)
# ----------------------------------------------------------------------

# builtins.py map key -> manifest.yaml section. The names differ per kind;
# docs/playbooks/add-a-plugin.md step 3 is the human-facing statement.
_BUILTINS_TO_MANIFEST = {
    "pipeline": "pipelines",
    "data": "data",
    "broker": "brokers",
    "publisher": "publishers",
    "risk": "risk",
    "strategy": "strategies",
    "rebalancing": "rebalancing",
    "feature": "features",
    "monitor": "monitors",
    "validation": "validation",
    "overlay": "overlays",
}


@pytest.mark.pipeline_smoke
def test_every_exported_strategy_is_registered() -> None:
    """Every plugin class the strategies package exports must be in builtins.py.

    Catches: a class exported from ``plugins/strategies/__init__.py`` with a
    ``meta`` but never wired into ``builtins.py`` — importable from Python,
    unreachable from YAML (four sat like that until TOM-1341).
    """
    import quantbox.plugins.strategies as strategies_pkg
    from quantbox.plugins.builtins import builtins as builtin_plugins

    registered = set(builtin_plugins()["strategy"].values())
    exported = {
        name: obj
        for name in strategies_pkg.__all__
        if isinstance(obj := getattr(strategies_pkg, name), type) and isinstance(getattr(obj, "meta", None), PluginMeta)
    }
    assert exported, "no plugin classes found in strategies.__all__ — the check would be vacuous"
    unregistered = sorted(name for name, cls in exported.items() if cls not in registered)
    assert not unregistered, f"exported but not registered in builtins.py: {unregistered}"


@pytest.mark.pipeline_smoke
def test_manifest_lists_exactly_the_registered_builtins() -> None:
    """``manifest.yaml`` ``plugins.builtins`` must equal the builtins.py map, kind by kind.

    Catches: a plugin added to builtins.py but not to the manifest (three
    were, until TOM-1341), a manifest entry naming a plugin that no longer
    exists, and a kind present on one side only.
    """
    from importlib.resources import files

    import yaml

    from quantbox.plugins.builtins import builtins as builtin_plugins

    manifest = yaml.safe_load(files("quantbox.plugins").joinpath("manifest.yaml").read_text(encoding="utf-8"))
    sections = manifest["plugins"]["builtins"]
    builtins = builtin_plugins()

    assert set(builtins) == set(_BUILTINS_TO_MANIFEST), (
        f"unmapped builtins kinds: {set(builtins) ^ set(_BUILTINS_TO_MANIFEST)}"
    )
    assert set(sections) == set(_BUILTINS_TO_MANIFEST.values()), (
        f"manifest sections differ from kinds: {set(sections) ^ set(_BUILTINS_TO_MANIFEST.values())}"
    )
    disagreements = {}
    for kind, section in _BUILTINS_TO_MANIFEST.items():
        registered = set(builtins[kind])
        listed = set(sections[section] or [])
        if registered != listed:
            disagreements[kind] = {
                "not_in_manifest": sorted(registered - listed),
                "not_registered": sorted(listed - registered),
            }
    assert not disagreements, f"registry and manifest disagree: {disagreements}"
