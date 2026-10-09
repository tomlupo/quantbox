from __future__ import annotations

import contextlib
import importlib.metadata
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ._lazy import load
from .contracts import (
    BrokerPlugin,
    DataPlugin,
    FeaturePlugin,
    MonitorPlugin,
    OverlayPlugin,
    PipelinePlugin,
    PublisherPlugin,
    RebalancingPlugin,
    RiskPlugin,
    StrategyPlugin,
    ValidationPlugin,
)
from .plugins.builtins import BUILTIN_PLUGINS

__all__ = ["ENTRYPOINT_GROUPS", "PluginMap", "PluginRegistry"]

ENTRYPOINT_GROUPS = {
    "pipeline": "quantbox.pipelines",
    "broker": "quantbox.brokers",
    "data": "quantbox.data",
    "publisher": "quantbox.publishers",
    "risk": "quantbox.risk",
    "strategy": "quantbox.strategies",
    "rebalancing": "quantbox.rebalancing",
    "feature": "quantbox.features",
    "validation": "quantbox.validations",
    "monitor": "quantbox.monitors",
    "overlay": "quantbox.overlays",
    "dataset": "quantbox.datasets",
    "capability": "quantbox.capabilities",
}


def _load_group(group: str) -> dict[str, Any]:
    eps = importlib.metadata.entry_points(group=group)
    out: dict[str, Any] = {}
    for ep in eps:
        out[ep.name] = ep.load()
    return out


class PluginMap(dict):
    """Plugin name -> plugin class; a builtin is held as ``"module:Class"`` until it is asked for.

    A ``dict``, so every reader keeps working. Reading a value (``m[name]``,
    ``get``, ``items``, ``values``, ``==``, ``{**m}``, ``dict(m)``) imports that
    plugin's module through :func:`quantbox._lazy.load` and stores the class.
    The keys, ``in``, ``len`` and ``sorted(m)`` import nothing, so
    ``quantbox plugins list`` and a lookup by name load only what they name (TOM-1451).
    """

    def _resolve(self, key: Any) -> Any:
        value = dict.__getitem__(self, key)
        if isinstance(value, str):
            value = load(value)
            dict.__setitem__(self, key, value)
        return value

    def _resolve_all(self) -> None:
        for key in list(dict.keys(self)):
            self._resolve(key)

    def __getitem__(self, key: Any) -> Any:
        return self._resolve(key)

    def __iter__(self) -> Iterator[Any]:
        # Overridden so that {**m}, dict(m) and d.update(m) read values through __getitem__,
        # never the stored "module:Class" strings (CPython's fast path skips a subclass's __iter__).
        return dict.__iter__(self)

    def get(self, key: Any, default: Any = None) -> Any:
        return self._resolve(key) if dict.__contains__(self, key) else default

    def items(self):  # type: ignore[override]
        self._resolve_all()
        return dict.items(self)

    def values(self):  # type: ignore[override]
        self._resolve_all()
        return dict.values(self)

    def pop(self, key: Any, *default: Any) -> Any:
        if dict.__contains__(self, key):
            value = self._resolve(key)
            dict.__delitem__(self, key)
            return value
        return dict.pop(self, key, *default)

    def popitem(self) -> tuple[Any, Any]:
        key = next(reversed(list(dict.keys(self))))
        return key, self.pop(key)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        if dict.__contains__(self, key):
            return self._resolve(key)
        dict.__setitem__(self, key, default)
        return default

    def copy(self) -> PluginMap:
        return PluginMap(dict.items(self))

    def __eq__(self, other: object) -> bool:
        self._resolve_all()
        if isinstance(other, PluginMap):
            other._resolve_all()
        return dict.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = None  # type: ignore[assignment]

    def __or__(self, other: Any) -> PluginMap:
        new = self.copy()
        new.update(other)
        return new

    def __ror__(self, other: Any) -> dict:
        self._resolve_all()
        return {**other, **dict(dict.items(self))}

    def __repr__(self) -> str:
        self._resolve_all()
        return dict.__repr__(self)


def _group(kind: str) -> PluginMap:
    """The builtins of *kind* (lazy) with the entry points of its group merged over them."""
    plugins = PluginMap(BUILTIN_PLUGINS.get(kind, {}))
    plugins.update(_load_group(ENTRYPOINT_GROUPS[kind]))
    return plugins


@dataclass
class PluginRegistry:
    pipelines: dict[str, type[PipelinePlugin]]
    brokers: dict[str, type[BrokerPlugin]]
    data: dict[str, type[DataPlugin]]
    publishers: dict[str, type[PublisherPlugin]]
    risk: dict[str, type[RiskPlugin]]
    strategies: dict[str, type[StrategyPlugin]] = field(default_factory=dict)
    rebalancing: dict[str, type[RebalancingPlugin]] = field(default_factory=dict)
    features: dict[str, type[FeaturePlugin]] = field(default_factory=dict)
    validations: dict[str, type[ValidationPlugin]] = field(default_factory=dict)
    monitors: dict[str, type[MonitorPlugin]] = field(default_factory=dict)
    overlays: dict[str, type[OverlayPlugin]] = field(default_factory=dict)
    datasets: dict[str, type] = field(default_factory=dict)
    capabilities: dict[str, type] = field(default_factory=dict)

    @staticmethod
    def discover() -> PluginRegistry:
        capability_classes = dict(_group("capability"))
        # Side-effect: register each capability class instance into the strict registry.
        from .strict import register_capability  # local import to avoid cycle

        for _name, _cls in capability_classes.items():
            with contextlib.suppress(Exception):
                register_capability(_name, _cls())
        return PluginRegistry(
            pipelines=_group("pipeline"),
            brokers=_group("broker"),
            data=_group("data"),
            publishers=_group("publisher"),
            risk=_group("risk"),
            strategies=_group("strategy"),
            rebalancing=_group("rebalancing"),
            features=_group("feature"),
            validations=_group("validation"),
            monitors=_group("monitor"),
            overlays=_group("overlay"),
            datasets=_group("dataset"),
            capabilities=capability_classes,
        )
