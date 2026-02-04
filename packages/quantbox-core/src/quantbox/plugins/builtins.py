"""Built-in plugin registry.

These plugins are shipped inside the core package. External plugins can still be
installed via entry points and will be merged by the PluginRegistry.
"""

from __future__ import annotations

from typing import Dict, Type

from ..contracts import PipelinePlugin, BrokerPlugin, DataPlugin, PublisherPlugin, RiskPlugin
from .pipeline import AllocationsToOrdersPipeline, FundSelectionPipeline
from .data import DuckDBParquetData
from .broker import (
    BinanceBroker,
    BinancePaperBrokerStub,
    IBKRBroker,
    IBKRPaperBrokerStub,
    SimPaperBroker,
)


def _map(*classes):
    return {c.meta.name: c for c in classes}


def builtins() -> Dict[str, Dict[str, Type]]:
    return {
        "pipeline": _map(FundSelectionPipeline, AllocationsToOrdersPipeline),
        "data": _map(DuckDBParquetData),
        "broker": _map(
            SimPaperBroker,
            IBKRPaperBrokerStub,
            BinancePaperBrokerStub,
            IBKRBroker,
            BinanceBroker,
        ),
        "publisher": {},
        "risk": {},
    }
