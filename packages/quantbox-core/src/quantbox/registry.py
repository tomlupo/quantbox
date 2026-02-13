from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, field
from typing import Any

from .contracts import (
    BrokerPlugin,
    DataPlugin,
    PipelinePlugin,
    PublisherPlugin,
    RebalancingPlugin,
    RiskPlugin,
    StrategyPlugin,
)
from .plugins.builtins import builtins as builtin_plugins

ENTRYPOINT_GROUPS = {
    "pipeline": "quantbox.pipelines",
    "broker": "quantbox.brokers",
    "data": "quantbox.data",
    "publisher": "quantbox.publishers",
    "risk": "quantbox.risk",
    "strategy": "quantbox.strategies",
    "rebalancing": "quantbox.rebalancing",
}


def _load_group(group: str) -> dict[str, Any]:
    eps = importlib.metadata.entry_points(group=group)
    out: dict[str, Any] = {}
    for ep in eps:
        out[ep.name] = ep.load()
    return out


@dataclass
class PluginRegistry:
    pipelines: dict[str, type[PipelinePlugin]]
    brokers: dict[str, type[BrokerPlugin]]
    data: dict[str, type[DataPlugin]]
    publishers: dict[str, type[PublisherPlugin]]
    risk: dict[str, type[RiskPlugin]]
    strategies: dict[str, type[StrategyPlugin]] = field(default_factory=dict)
    rebalancing: dict[str, type[RebalancingPlugin]] = field(default_factory=dict)

    @staticmethod
    def discover() -> PluginRegistry:
        builtins = builtin_plugins()
        return PluginRegistry(
            pipelines={**builtins["pipeline"], **_load_group(ENTRYPOINT_GROUPS["pipeline"])},
            brokers={**builtins["broker"], **_load_group(ENTRYPOINT_GROUPS["broker"])},
            data={**builtins["data"], **_load_group(ENTRYPOINT_GROUPS["data"])},
            publishers={**builtins["publisher"], **_load_group(ENTRYPOINT_GROUPS["publisher"])},
            risk={**builtins["risk"], **_load_group(ENTRYPOINT_GROUPS["risk"])},
            strategies={**builtins.get("strategy", {}), **_load_group(ENTRYPOINT_GROUPS["strategy"])},
            rebalancing={**builtins.get("rebalancing", {}), **_load_group(ENTRYPOINT_GROUPS["rebalancing"])},
        )
