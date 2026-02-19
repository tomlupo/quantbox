# Plugin Protocols Reference

All plugins implement Protocols defined in `packages/quantbox-core/src/quantbox/contracts.py`.
Every plugin is a `@dataclass` with a class-level `meta = PluginMeta(...)` attribute.

## PluginMeta

```python
@dataclass(frozen=True)
class PluginMeta:
    name: str                              # Unique ID, e.g. "strategy.crypto_trend.v1"
    kind: PluginKind                       # "pipeline"|"broker"|"data"|"publisher"|"risk"|"strategy"|"rebalancing"
    version: str                           # Semver, e.g. "0.1.0"
    core_compat: str                       # Semver range, e.g. ">=0.1,<0.2"
    description: str = ""                  # Human/LLM-readable description
    tags: tuple[str, ...] = ()             # Searchable tags
    capabilities: tuple[str, ...] = ()     # Supported modes/features
    schema_version: str = "v1"             # Artifact schema version
    params_schema: dict | None = None      # JSON Schema for parameters
    inputs: tuple[str, ...] = ()           # Expected input artifact names
    outputs: tuple[str, ...] = ()          # Produced artifact names
    examples: tuple[str, ...] = ()         # Minimal YAML config snippets
```

## RunResult

Returned by `PipelinePlugin.run()`:

```python
@dataclass
class RunResult:
    run_id: str                    # Unique run identifier
    pipeline_name: str             # Name of pipeline that produced this
    mode: Mode                     # "backtest"|"paper"|"live"
    asof: str                      # Reference date (ISO format)
    artifacts: dict[str, str]      # artifact_name -> file_path
    metrics: dict[str, float]      # Numeric metrics (portfolio_value, n_orders, etc.)
    notes: dict[str, Any]          # Freeform metadata (risk findings, debug info)
```

## ArtifactStore Protocol

```python
class ArtifactStore(Protocol):
    def put_parquet(self, name: str, df: pd.DataFrame) -> str: ...
    def put_json(self, name: str, obj: dict) -> str: ...
    def get_path(self, name: str) -> str: ...
    def read_parquet(self, name: str) -> pd.DataFrame: ...
    def read_json(self, name: str) -> dict: ...
    def list_artifacts(self) -> list[str]: ...
    @property
    def run_id(self) -> str: ...
```

## DataPlugin Protocol

Loads market data in **wide format** (DatetimeIndex x symbol columns).

```python
class DataPlugin(Protocol):
    meta: PluginMeta

    def load_universe(self, params: dict) -> pd.DataFrame:
        """Returns DataFrame with 'symbol' column."""
        ...

    def load_market_data(
        self, universe: pd.DataFrame, asof: str, params: dict
    ) -> dict[str, pd.DataFrame]:
        """Returns {"prices": wide_df, "volume": wide_df, "market_cap": wide_df, ...}
        Only "prices" is required; others may be empty DataFrames."""
        ...

    def load_fx(self, asof: str, params: dict) -> pd.DataFrame | None:
        """FX rates, or None for crypto."""
        ...
```

## BrokerPlugin Protocol

```python
class BrokerPlugin(Protocol):
    meta: PluginMeta

    def get_positions(self) -> pd.DataFrame:
        """Current holdings: DataFrame[symbol, qty, value]."""
        ...

    def get_cash(self) -> dict[str, float]:
        """Cash balances: {"USD": 100000.0}."""
        ...

    def get_market_snapshot(self, symbols: list[str]) -> pd.DataFrame:
        """Current prices/info for given symbols."""
        ...

    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Execute orders (columns: symbol, side, qty). Returns fills DataFrame."""
        ...

    def fetch_fills(self, since: str) -> pd.DataFrame:
        """Historical fills since timestamp."""
        ...
```

Optional methods (checked via `hasattr`):
- `get_equity() -> float` - Total account value (margin + unrealized PnL). Preferred over cash + positions for derivatives.
- `describe() -> dict` - Structured state snapshot for LLM inspection.

## StrategyPlugin Protocol

```python
class StrategyPlugin(Protocol):
    meta: PluginMeta

    def run(
        self,
        data: dict[str, Any],        # {"prices": df, "volume": df, "market_cap": df, "universe": df}
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Returns dict with at minimum "weights" (wide DataFrame: date x symbols)."""
        ...
```

## PipelinePlugin Protocol

```python
class PipelinePlugin(Protocol):
    meta: PluginMeta
    kind: PipelineKind              # "research" or "trading"

    def run(
        self,
        *,
        mode: Mode,                 # "backtest"|"paper"|"live"
        asof: str,                  # Reference date "YYYY-MM-DD"
        params: dict[str, Any],
        data: DataPlugin,
        store: ArtifactStore,
        broker: BrokerPlugin | None,
        risk: list[RiskPlugin],
        strategies: list[StrategyPlugin] | None = None,
        rebalancer: RebalancingPlugin | None = None,
        **kwargs,
    ) -> RunResult: ...
```

## RebalancingPlugin Protocol

```python
class RebalancingPlugin(Protocol):
    meta: PluginMeta

    def generate_orders(
        self,
        *,
        weights: dict[str, float],  # {symbol: target_weight}
        broker: BrokerPlugin,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Returns dict with "orders" DataFrame (symbol, side, qty, order_type)."""
        ...
```

## RiskPlugin Protocol

```python
class RiskPlugin(Protocol):
    meta: PluginMeta

    def check_targets(
        self, targets: pd.DataFrame, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Validate weight targets. Returns list of findings (empty = all clear)."""
        ...

    def check_orders(
        self, orders: pd.DataFrame, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Validate generated orders. Returns list of findings."""
        ...
```

## PublisherPlugin Protocol

```python
class PublisherPlugin(Protocol):
    meta: PluginMeta

    def publish(self, result: RunResult, params: dict[str, Any]) -> None:
        """Send run results to external destination (Telegram, Slack, etc.)."""
        ...
```

## Registration

### Built-in (recommended for core plugins)

1. Create module in `plugins/<type>/<name>.py`
2. Export from `plugins/<type>/__init__.py`
3. Import in `plugins/builtins.py` and add to `builtins()` dict:
   ```python
   from .pipeline import MyPipeline
   # In builtins():
   "pipeline": _map(..., MyPipeline),
   ```

### External (separate package)

Register via entry point in `pyproject.toml`:
```toml
[project.entry-points."quantbox.pipelines"]
"my.pipeline.v1" = "my_pkg.pipeline:MyPipeline"
```

Entry point groups: `quantbox.pipelines`, `quantbox.brokers`, `quantbox.data`,
`quantbox.publishers`, `quantbox.risk`, `quantbox.strategies`, `quantbox.rebalancing`.

## Plugin Naming Convention

Format: `<namespace>.<descriptive_name>.v<version>`

Examples:
- `strategy.crypto_trend.v1`
- `binance.live_data.v1`
- `risk.trading_basic.v1`
- `rebalancing.futures.v1`

Never rename existing IDs. Create a new version instead.
