from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Literal
import pandas as pd

Mode = Literal["backtest", "paper", "live"]
PluginKind = Literal["pipeline", "broker", "data", "publisher", "risk"]
PipelineKind = Literal["research", "trading"]

@dataclass(frozen=True)
class PluginMeta:
    name: str
    kind: PluginKind
    version: str
    core_compat: str
    description: str = ""
    tags: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    schema_version: str = "v1"

@dataclass
class RunResult:
    run_id: str
    pipeline_name: str
    mode: Mode
    asof: str
    artifacts: Dict[str, str]
    metrics: Dict[str, float]
    notes: Dict[str, Any]

class ArtifactStore(Protocol):
    def put_parquet(self, name: str, df: pd.DataFrame) -> str: ...
    def put_json(self, name: str, obj: Dict[str, Any]) -> str: ...
    def get_path(self, name: str) -> str: ...
    @property
    def run_id(self) -> str: ...

class DataPlugin(Protocol):
    meta: PluginMeta
    def load_universe(self, params: Dict[str, Any]) -> pd.DataFrame: ...
    def load_prices(self, universe: pd.DataFrame, asof: str, params: Dict[str, Any]) -> pd.DataFrame: ...
    def load_fx(self, asof: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]: ...

class BrokerPlugin(Protocol):
    meta: PluginMeta
    def get_positions(self) -> pd.DataFrame: ...
    def get_cash(self) -> Dict[str, float]: ...
    def get_market_snapshot(self, symbols: List[str]) -> pd.DataFrame: ...
    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame: ...
    def fetch_fills(self, since: str) -> pd.DataFrame: ...

class PublisherPlugin(Protocol):
    meta: PluginMeta
    def publish(self, result: RunResult, params: Dict[str, Any]) -> None: ...

class RiskPlugin(Protocol):
    meta: PluginMeta
    def check_targets(self, targets: pd.DataFrame, params: Dict[str, Any]) -> List[Dict[str, Any]]: ...
    def check_orders(self, orders: pd.DataFrame, params: Dict[str, Any]) -> List[Dict[str, Any]]: ...

class PipelinePlugin(Protocol):
    meta: PluginMeta
    kind: PipelineKind
    def run(
        self,
        *,
        mode: Mode,
        asof: str,
        params: Dict[str, Any],
        data: DataPlugin,
        store: ArtifactStore,
        broker: Optional[BrokerPlugin],
        risk: List[RiskPlugin],
    ) -> RunResult: ...
