"""Plugin protocol contracts for Quantbox.

Defines the interfaces that all plugins must implement. Each protocol
specifies the minimal method signatures — plugins are ``@dataclass``
classes with a class-level ``meta = PluginMeta(...)`` attribute.

## LLM Quick Reference

**DataPlugin** — loads market data:
    load_universe(params) → DataFrame[symbol]
    load_market_data(universe, asof, params) → {"prices": wide_df, "volume": wide_df, ...}
    load_fx(asof, params) → DataFrame | None

**BrokerPlugin** — executes orders:
    get_positions() → DataFrame[symbol, qty]
    get_cash() → {"USD": float}
    place_orders(orders_df) → fills_df

**PipelinePlugin** — orchestrates a workflow:
    run(mode, asof, params, data, store, broker, risk, ...) → RunResult

**StrategyPlugin** — computes target weights:
    run(data, params) → {"weights": DataFrame, ...}

**RiskPlugin** — validates targets/orders:
    check_targets(targets, params) → [findings]
    check_orders(orders, params) → [findings]

**RebalancingPlugin** — generates orders from weights:
    generate_orders(weights, broker, params) → {"orders": df, ...}

**PublisherPlugin** — sends notifications:
    publish(result, params) → None
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Literal
import pandas as pd

Mode = Literal["backtest", "paper", "live"]
PluginKind = Literal["pipeline", "broker", "data", "publisher", "risk", "strategy", "rebalancing"]
PipelineKind = Literal["research", "trading"]

@dataclass(frozen=True)
class PluginMeta:
    """Metadata describing a plugin for discovery and documentation.

    Attributes:
        name: Unique plugin identifier (e.g. "binance.live_data.v1").
        kind: Plugin type — determines which protocol it implements.
        version: Semver version of this plugin.
        core_compat: Semver range of compatible quantbox-core versions.
        description: Human/LLM-readable description of what this plugin does.
        tags: Searchable tags (e.g. ("crypto", "futures")).
        capabilities: Supported modes/features (e.g. ("paper", "live")).
        schema_version: Version of the artifact schema this plugin produces.
        params_schema: JSON Schema for plugin parameters (LLM-friendly).
        inputs: Artifact names this plugin expects as input.
        outputs: Artifact names this plugin produces.
        examples: Minimal YAML config snippets showing usage.
    """
    name: str
    kind: PluginKind
    version: str
    core_compat: str
    description: str = ""
    tags: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    schema_version: str = "v1"
    params_schema: Optional[Dict[str, Any]] = None
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()

@dataclass
class RunResult:
    """Result returned by ``PipelinePlugin.run()``.

    Attributes:
        run_id: Unique identifier for this run.
        pipeline_name: Name of the pipeline that produced this result.
        mode: Execution mode (backtest/paper/live).
        asof: Reference date (ISO format).
        artifacts: Map of artifact name → file path.
        metrics: Numeric metrics (e.g. portfolio_value, n_orders).
        notes: Freeform metadata (risk findings, debug info, etc.).
    """
    run_id: str
    pipeline_name: str
    mode: Mode
    asof: str
    artifacts: Dict[str, str]
    metrics: Dict[str, float]
    notes: Dict[str, Any]

class ArtifactStore(Protocol):
    """Stores pipeline artifacts (Parquet files, JSON) with run-level grouping."""
    def put_parquet(self, name: str, df: pd.DataFrame) -> str: ...
    def put_json(self, name: str, obj: Dict[str, Any]) -> str: ...
    def get_path(self, name: str) -> str: ...
    def read_parquet(self, name: str) -> pd.DataFrame: ...
    def read_json(self, name: str) -> Dict[str, Any]: ...
    def list_artifacts(self) -> List[str]: ...
    @property
    def run_id(self) -> str: ...

class DataPlugin(Protocol):
    """Loads market data for pipelines and strategies.

    All data is returned in **wide format**: DataFrames with a DatetimeIndex
    and one column per symbol.

    Methods:
        load_universe: Returns DataFrame with ``symbol`` column.
        load_market_data: Returns dict of wide DataFrames. Required key:
            ``"prices"`` (close prices). Optional: ``"volume"``,
            ``"market_cap"``, ``"funding_rates"``.
        load_fx: Returns FX rate DataFrame, or None if not applicable.

    Example:
        >>> data = plugin.load_market_data(universe, "2026-02-01", {"lookback_days": 365})
        >>> data["prices"]  # DataFrame: date index x symbol columns
        >>> data["volume"]  # DataFrame: date index x symbol columns
    """
    meta: PluginMeta
    def load_universe(self, params: Dict[str, Any]) -> pd.DataFrame: ...
    def load_market_data(self, universe: pd.DataFrame, asof: str, params: Dict[str, Any]) -> Dict[str, pd.DataFrame]: ...
    def load_fx(self, asof: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]: ...

class BrokerPlugin(Protocol):
    """Manages positions and executes orders.

    Methods:
        get_positions: Current holdings as DataFrame[symbol, qty].
        get_cash: Cash balances as {currency: amount}.
        get_market_snapshot: Current prices/info for symbols.
        place_orders: Submit orders, returns fills DataFrame.
        fetch_fills: Historical fills since a timestamp.

    Optional methods (checked via hasattr):
        get_equity: Total account value in USD. For derivatives brokers
            this is the authoritative portfolio value (margin + unrealized PnL).
            Pipelines prefer this over cash + sum(qty * price) when available,
            since the latter is incorrect for short/futures positions.
    """
    meta: PluginMeta
    def get_positions(self) -> pd.DataFrame: ...
    def get_cash(self) -> Dict[str, float]: ...
    def get_market_snapshot(self, symbols: List[str]) -> pd.DataFrame: ...
    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame: ...
    def fetch_fills(self, since: str) -> pd.DataFrame: ...

class PublisherPlugin(Protocol):
    """Sends run results to external destinations (Telegram, Slack, etc.)."""
    meta: PluginMeta
    def publish(self, result: RunResult, params: Dict[str, Any]) -> None: ...

class RiskPlugin(Protocol):
    """Validates portfolio targets and orders against risk limits.

    Returns a list of findings (dicts with severity, message, etc.).
    Empty list = all checks passed.
    """
    meta: PluginMeta
    def check_targets(self, targets: pd.DataFrame, params: Dict[str, Any]) -> List[Dict[str, Any]]: ...
    def check_orders(self, orders: pd.DataFrame, params: Dict[str, Any]) -> List[Dict[str, Any]]: ...

class StrategyPlugin(Protocol):
    """Computes target portfolio weights from market data.

    Input ``data`` dict contains wide DataFrames: ``prices``, ``volume``,
    ``market_cap``, ``universe``, and optionally ``funding_rates``.

    Returns dict with at minimum ``"weights"`` (DataFrame: date index x symbol columns).
    """
    meta: PluginMeta
    def run(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]: ...

class RebalancingPlugin(Protocol):
    """Generates executable orders from target weights + current broker state."""
    meta: PluginMeta
    def generate_orders(
        self,
        *,
        weights: Dict[str, float],
        broker: BrokerPlugin,
        params: Dict[str, Any],
    ) -> Dict[str, Any]: ...

class PipelinePlugin(Protocol):
    """Top-level orchestrator: data loading → strategy → risk → execution → artifacts."""
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
        strategies: Optional[List["StrategyPlugin"]] = None,
        rebalancer: Optional["RebalancingPlugin"] = None,
        **kwargs,
    ) -> RunResult: ...
