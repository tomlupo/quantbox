from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Type
import importlib.metadata

from .contracts import PipelinePlugin, BrokerPlugin, DataPlugin, PublisherPlugin, RiskPlugin

ENTRYPOINT_GROUPS = {
    "pipeline": "quantbox.pipelines",
    "broker": "quantbox.brokers",
    "data": "quantbox.data",
    "publisher": "quantbox.publishers",
    "risk": "quantbox.risk",
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

    @staticmethod
    def discover() -> "PluginRegistry":
        return PluginRegistry(
            pipelines=_load_group(ENTRYPOINT_GROUPS["pipeline"]),
            brokers=_load_group(ENTRYPOINT_GROUPS["broker"]),
            data=_load_group(ENTRYPOINT_GROUPS["data"]),
            publishers=_load_group(ENTRYPOINT_GROUPS["publisher"]),
            risk=_load_group(ENTRYPOINT_GROUPS["risk"]),
        )
