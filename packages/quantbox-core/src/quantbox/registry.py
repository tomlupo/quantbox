from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Type
import importlib.metadata

from .contracts import PipelinePlugin, BrokerPlugin, DataPlugin, PublisherPlugin, RiskPlugin, StrategyPlugin, RebalancingPlugin
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

def _load_group(group: str) -> Dict[str, Any]:
    eps = importlib.metadata.entry_points(group=group)
    out: Dict[str, Any] = {}
    for ep in eps:
        out[ep.name] = ep.load()
    return out

@dataclass
class PluginRegistry:
    pipelines: Dict[str, Type[PipelinePlugin]]
    brokers: Dict[str, Type[BrokerPlugin]]
    data: Dict[str, Type[DataPlugin]]
    publishers: Dict[str, Type[PublisherPlugin]]
    risk: Dict[str, Type[RiskPlugin]]
    strategies: Dict[str, Type[StrategyPlugin]] = field(default_factory=dict)
    rebalancing: Dict[str, Type[RebalancingPlugin]] = field(default_factory=dict)

    @staticmethod
    def discover() -> "PluginRegistry":
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
