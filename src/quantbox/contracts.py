"""Plugin protocol contracts for Quantbox.

Defines the interfaces that all plugins must implement. Each protocol
specifies the minimal method signatures — plugins are ``@dataclass``
classes with a class-level ``meta = PluginMeta(...)`` attribute.

## LLM Quick Reference

**DataPlugin** — loads market data:
    load_universe(params) → DataFrame[symbol]
    load_market_data(universe, asof, params) → {"prices": wide_df, "volume": wide_df,
        "high": wide_df, "low": wide_df, "market_cap": wide_df, "funding_rates": wide_df,
        "eligibility_mask": wide_df, ...}
    load_fx(asof, params) → DataFrame | None

**BrokerPlugin** — executes orders:
    get_positions() → DataFrame[symbol, qty]
    get_cash() → {"USD": float}
    place_orders(orders_df) → fills_df

**PipelinePlugin** — orchestrates a workflow:
    run(mode, asof, params, data, store, broker, risk, ...) → RunResult

**StrategyPlugin** — computes target weights:
    run(data, params, context=None) → {"weights": DataFrame, ...}
    ``context`` is a :class:`StrategyContext` (bars_per_year, mode, asof,
    calendar, frequency) handed by the one strategy runner
    (``quantbox.strategy_runner``), identically in backtest and trading.

**RiskPlugin** — validates targets/orders:
    check_targets(targets, params) → [findings]
    check_orders(orders, params) → [findings]

**RebalancingPlugin** — generates orders from weights:
    generate_orders(weights, broker, params) → {"orders": df, ...}

**PublisherPlugin** — sends notifications:
    publish(result, params) → None

**FeaturePlugin** — computes derived features:
    compute(data, params) → DataFrame

**ValidationPlugin** — validates returns/weights:
    validate(returns, weights, benchmark, params) → dict

**MonitorPlugin** — checks run results for anomalies:
    check(result, history, params) → [alerts]

**OverlayPlugin** — modifies decided weights before execution (chained):
    apply(weights, data, params) → DataFrame (same index and columns)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol

import pandas as pd

__all__ = [
    "BROKER_CAPABILITY_MEMBERS",
    "AcceptsFundingRates",
    "AcceptsPositionLimits",
    "AcceptsPrices",
    "AccruesFees",
    "AppliesFunding",
    "ArtifactStore",
    "BrokerCapability",
    "BrokerPlugin",
    "HasCostModel",
    "ReadsFundingHistory",
    "ResolvesOrders",
    "SupportsEquity",
    "ValuationBasis",
    "DataPlugin",
    "FeaturePlugin",
    "Mode",
    "MonitorPlugin",
    "OverlayPlugin",
    "PipelineKind",
    "PipelinePlugin",
    "PluginKind",
    "PluginMeta",
    "PluginStatus",
    "PublisherPlugin",
    "RebalancingPlugin",
    "RiskPlugin",
    "RunResult",
    "StrategyContext",
    "StrategyPlugin",
    "ValidationPlugin",
]

Mode = Literal["backtest", "paper", "live"]
PluginKind = Literal[
    "pipeline",
    "broker",
    "data",
    "publisher",
    "risk",
    "strategy",
    "rebalancing",
    "feature",
    "validation",
    "monitor",
    "overlay",
    "dataset",
]
PipelineKind = Literal["research", "trading"]
PluginStatus = Literal["research", "locked", "production"]
"""Lifecycle status of a plugin/methodology.

- ``research`` — default; plugin is in development or its methodology spec is not LOCKED.
  Acceptable for ad-hoc backtests and notebooks. ``--strict`` mode will refuse these.
- ``locked`` — methodology spec frozen with ``status: LOCKED`` frontmatter and a
  paired ``docs/methodology/{name}.md`` doc. Validation evidence recorded.
  Not yet tagged for production.
- ``production`` — has a ``prod-{subsystem}-vX.Y.Z-YYYYMMDD`` git tag. Reproducibility
  pins required (uv.lock + dataset content-hashes + seeds). Subject to monthly
  revalidation cron.

State transitions are human-driven via ``/promote-lock`` and ``/promote``. The runtime
never auto-promotes. See ``docs/architecture/lifecycle.md``.
"""


@dataclass(frozen=True)
class StrategyContext:
    """What a strategy knows about the run it is in — built once, by the one strategy runner.

    ``quantbox.strategy_runner.build_strategy_context`` derives it from the
    run's params, so backtest and trading hand the SAME context to the same
    strategy and config (they differ only in ``mode``). TOM-1448.

    Attributes:
        bars_per_year: Annualisation factor of the run's bars (365.0 for daily
            24/7, ~250 for daily NYSE, 8760.0 for hourly 24/7).
        mode: ``backtest`` | ``paper`` | ``live``.
        asof: The run's as-of date, as the run was given it.
        calendar: The trading calendar (``24/7``, ``NYSE``, ...).
        frequency: The bar size (``1d``, ``4h``, ``1h``, ...).
    """

    bars_per_year: float
    mode: str
    asof: str
    calendar: str
    frequency: str


@dataclass(frozen=True)
class PluginMeta:
    """Metadata describing a plugin for discovery and documentation.

    Attributes:
        name: Unique plugin identifier (e.g. "binance.live_data.v1").
        kind: Plugin type — determines which protocol it implements.
        version: Semver version of this plugin.
        core_compat: Semver range of compatible quantbox-core versions.
        status: Lifecycle status — ``research`` | ``locked`` | ``production``.
            Defaults to ``"research"``. ``--strict`` mode rejects below ``locked``.
            See ``docs/architecture/lifecycle.md``.
        description: Human/LLM-readable description of what this plugin does.
        tags: Searchable tags (e.g. ("crypto", "futures")).
        capabilities: Supported modes/features (e.g. ("paper", "live")).
        schema_version: Version of the artifact schema this plugin produces.
        params_schema: JSON Schema for plugin parameters (LLM-friendly). Required
            for every registered plugin: each key a config may set is a property
            with a description. ``quantbox.params_schema.resolve_params_schema``
            completes constructor params (type, default) from the signature.
        inputs: Artifact names this plugin expects as input.
        outputs: Artifact names this plugin produces.
        examples: Minimal YAML config snippets showing usage.
    """

    name: str
    kind: PluginKind
    version: str
    core_compat: str
    status: PluginStatus = "research"
    description: str = ""
    tags: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    schema_version: str = "v1"
    params_schema: dict[str, Any] | None = None
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
        metrics: Numeric metrics; None where a value could not be MEASURED (#92) —
        an explicit null is a signal, whereas an absent key is indistinguishable
        from a metric that was never emitted and lets every consumer default it
        to 0 (e.g. portfolio_value, n_orders).
        notes: Freeform metadata (risk findings, debug info, etc.).
    """

    run_id: str
    pipeline_name: str
    mode: Mode
    asof: str
    artifacts: dict[str, str]
    metrics: dict[str, float | None]
    notes: dict[str, Any]


class ArtifactStore(Protocol):
    """Stores pipeline artifacts (Parquet files, JSON) with run-level grouping."""

    def put_parquet(self, name: str, df: pd.DataFrame) -> str: ...
    def put_json(self, name: str, obj: dict[str, Any]) -> str: ...
    def put_text(self, name: str, content: str) -> str: ...
    def get_path(self, name: str) -> str: ...
    def read_parquet(self, name: str) -> pd.DataFrame: ...
    def read_json(self, name: str) -> dict[str, Any]: ...
    def list_artifacts(self) -> list[str]: ...
    @property
    def run_id(self) -> str: ...


class DataPlugin(Protocol):
    """Loads market data for pipelines and strategies.

    All data is returned in **wide format**: DataFrames with a DatetimeIndex
    and one column per symbol.

    Methods:
        load_universe: Returns DataFrame with ``symbol`` column.
        load_market_data: Returns dict of wide DataFrames. Required key:
            ``"prices"`` (close prices). Recognised optional keys (the engine
            will ``setdefault`` each to an empty DataFrame if the plugin omits
            it, so strategies can always ``data.get(key)`` safely):
              - ``"volume"`` — quote-currency dollar volume
              - ``"high"`` — daily high (for ATR / true-range calcs)
              - ``"low"`` — daily low
              - ``"market_cap"`` — monthly mcap snapshots (typically forward-filled)
              - ``"funding_rates"`` — perp funding (futures datasets)
              - ``"eligibility_mask"`` — boolean wide DataFrame, top-N-by-mcap
                gate. Strategies that consume it get notebook-style daily
                universe rotation; strategies that ignore it keep working.
            Data plugins may emit additional keys, but only the canonical set
            above is guaranteed to be present in ``market_data``.
        load_fx: Returns FX rate DataFrame, or None if not applicable.

    Optional, read before any data is loaded (``quantbox config explain`` and the
    backtest pipeline): ``planned_paths(params)`` — ``{"prices", "funding_rates"}``,
    the files the plugin will read; ``planned_market()`` — the dataset's ``market``
    (its manifest), which the funding guard reads (:mod:`quantbox.funding_guard`);
    ``planned_funding()`` — for a plugin that plans no files, whether the dataset it
    will serve carries a funding series (True/False), or None when it cannot say.

    Example:
        >>> data = plugin.load_market_data(universe, "2026-02-01", {"lookback_days": 365})
        >>> data["prices"]  # DataFrame: date index x symbol columns
        >>> data["volume"]  # DataFrame: date index x symbol columns
    """

    meta: PluginMeta

    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame: ...
    def load_market_data(
        self, universe: pd.DataFrame, asof: str, params: dict[str, Any]
    ) -> dict[str, pd.DataFrame]: ...
    def load_fx(self, asof: str, params: dict[str, Any]) -> pd.DataFrame | None: ...


ValuationBasis = Literal["mark_to_market", "margin_balance"]
"""How a venue values a book: ``BrokerPlugin.valuation_basis`` (see :mod:`quantbox.trade.book`)."""


class BrokerCapability(StrEnum):
    """An optional thing a broker can do, which the trade path uses only when DECLARED.

    A broker declares its capabilities in one class attribute,
    ``broker_capabilities`` (a ``frozenset`` of these members). The pipeline
    reads that declaration and never probes a broker for a method
    (TOM-1459). A broker that declares nothing supports none of them, and the
    trade path behaves as it did for a broker without those methods.

    Each member names the members the broker must then have
    (:data:`BROKER_CAPABILITY_MEMBERS`) and the typed Protocol they form:

    * ``EQUITY`` — :class:`SupportsEquity`: ``get_equity()``, the venue's own
      account value. It MUST raise rather than return a value that omits a held
      position it could not price (see :mod:`quantbox.trade.book`).
    * ``FUNDING_APPLIED`` — :class:`AppliesFunding`: ``apply_funding()``, a
      simulated book that books funding itself and returns the signed charge.
    * ``FUNDING_HISTORY`` — :class:`ReadsFundingHistory`:
      ``fetch_funding_payments(since, until)``, a live venue that reads back
      the funding it took; ``None`` means unmeasured.
    * ``FEES_ACCRUED`` — :class:`AccruesFees`: ``cumulative_fees``, the fees
      the book has paid so far.
    * ``COST_MODEL`` — :class:`HasCostModel`: ``spread_bps``,
      ``slippage_bps``, ``maker_fee_bps``, ``taker_fee_bps``, the simulated
      execution costs, recorded in the trade artifact.
    * ``PRICE_INJECTION`` — :class:`AcceptsPrices`: ``set_prices(prices)``.
    * ``FUNDING_RATE_INJECTION`` — :class:`AcceptsFundingRates`:
      ``set_funding_rates(rates)``.
    * ``POSITION_LIMIT_INJECTION`` — :class:`AcceptsPositionLimits`:
      ``set_position_limits(limits)``.
    * ``ORDER_RESULT`` — :class:`ResolvesOrders`:
      ``fetch_order_result(order_id, symbol)``, the venue's answer for an order
      a previous cycle left working.

    A third-party broker declares, for example::

        from quantbox.contracts import BrokerCapability

        class MyBroker:
            valuation_basis = "margin_balance"
            broker_capabilities = frozenset({BrokerCapability.EQUITY, BrokerCapability.ORDER_RESULT})

    ``quantbox plugins doctor`` warns when a broker's declaration and its
    members disagree, in either direction.
    """

    EQUITY = "equity"
    FUNDING_APPLIED = "funding_applied"
    FUNDING_HISTORY = "funding_history"
    FEES_ACCRUED = "fees_accrued"
    COST_MODEL = "cost_model"
    PRICE_INJECTION = "price_injection"
    FUNDING_RATE_INJECTION = "funding_rate_injection"
    POSITION_LIMIT_INJECTION = "position_limit_injection"
    ORDER_RESULT = "order_result"


#: The members each capability obliges a broker to have. The one table that
#: ``quantbox plugins doctor`` and the per-broker tests check a declaration
#: against. ``COST_MODEL`` lists attributes; every other entry lists methods or
#: a property.
BROKER_CAPABILITY_MEMBERS: dict[BrokerCapability, tuple[str, ...]] = {
    BrokerCapability.EQUITY: ("get_equity",),
    BrokerCapability.FUNDING_APPLIED: ("apply_funding",),
    BrokerCapability.FUNDING_HISTORY: ("fetch_funding_payments",),
    BrokerCapability.FEES_ACCRUED: ("cumulative_fees",),
    BrokerCapability.COST_MODEL: ("spread_bps", "slippage_bps", "maker_fee_bps", "taker_fee_bps"),
    BrokerCapability.PRICE_INJECTION: ("set_prices",),
    BrokerCapability.FUNDING_RATE_INJECTION: ("set_funding_rates",),
    BrokerCapability.POSITION_LIMIT_INJECTION: ("set_position_limits",),
    BrokerCapability.ORDER_RESULT: ("fetch_order_result",),
}


class SupportsEquity(Protocol):
    """``BrokerCapability.EQUITY``."""

    def get_equity(self) -> float: ...


class AppliesFunding(Protocol):
    """``BrokerCapability.FUNDING_APPLIED``."""

    def apply_funding(self) -> float | None: ...


class ReadsFundingHistory(Protocol):
    """``BrokerCapability.FUNDING_HISTORY``."""

    def fetch_funding_payments(self, since: str, until: str) -> float | None: ...


class AccruesFees(Protocol):
    """``BrokerCapability.FEES_ACCRUED``."""

    @property
    def cumulative_fees(self) -> float: ...


class HasCostModel(Protocol):
    """``BrokerCapability.COST_MODEL``."""

    spread_bps: float
    slippage_bps: float
    maker_fee_bps: float
    taker_fee_bps: float


class AcceptsPrices(Protocol):
    """``BrokerCapability.PRICE_INJECTION``."""

    def set_prices(self, prices: dict[str, float]) -> None: ...


class AcceptsFundingRates(Protocol):
    """``BrokerCapability.FUNDING_RATE_INJECTION``."""

    def set_funding_rates(self, rates: dict[str, float]) -> None: ...


class AcceptsPositionLimits(Protocol):
    """``BrokerCapability.POSITION_LIMIT_INJECTION``."""

    def set_position_limits(self, limits: dict[str, float]) -> None: ...


class ResolvesOrders(Protocol):
    """``BrokerCapability.ORDER_RESULT``."""

    def fetch_order_result(self, order_id: str, symbol: str) -> dict[str, Any] | None: ...


class BrokerPlugin(Protocol):
    """Manages positions and executes orders.

    Methods (required; the trade path calls them without asking):
        get_positions: Current holdings as DataFrame[symbol, qty].
        get_cash: Cash balances as {currency: amount}.
        get_market_snapshot: Current prices/info for symbols.
        place_orders: Submit orders, returns fills DataFrame.
        fetch_fills: Historical fills since a timestamp.

    Optional capabilities are DECLARED, never probed: ``broker_capabilities``
    is a ``frozenset`` of :class:`BrokerCapability`, and each member names the
    typed Protocol the broker then implements (``get_equity`` is
    :class:`SupportsEquity`, and so on). A broker that omits the attribute
    declares nothing. :mod:`quantbox.trade.capabilities` reads it.

    Class attributes:
        broker_capabilities: The capabilities this broker declares. A plain
            class attribute, free of API calls; it must not vary with account
            state.
        valuation_basis: HOW THIS VENUE VALUES A BOOK — the one declaration that
            decides pre-trade sizing. ``"mark_to_market"`` for a spot/cash venue
            (value is cash + sum(qty * price), so every held position must be
            markable); ``"margin_balance"`` for a derivatives venue (value is
            the margin balance plus unrealised PnL, and leveraged positions do
            not add to it). Use the :data:`~quantbox.trade.book.BASIS_MARK`
            / :data:`~quantbox.trade.book.BASIS_MARGIN` constants.

            It is a plain class attribute, not a ``describe()`` key: reading it
            must be free of API calls and must not vary with account state.

            A broker that declares neither cannot be valued, and a LIVE run
            refuses rather than guessing. That is not defensive
            over-engineering — the live ``crypto-trend-kraken`` book sized every
            target off cash because its config named ``rebalancing.futures.v1``
            against a spot broker, and the margin-balance rule was applied
            faithfully to a book where it is wrong by the whole value of the
            positions held.
        lot_venue: Optional. The venue whose lot rules the broker sends with
            (``"hyperliquid"``, ``"kraken"``, ``"binance_futures"``). An order
            generator reads it with :func:`quantbox.trade.lots.lot_venue`, so
            it plans the quantity the broker sends (TOM-2207). A broker that
            omits it gets no venue facts: no reduce-only exemption, no
            fallback step.
    """

    meta: PluginMeta
    valuation_basis: ValuationBasis
    broker_capabilities: frozenset[BrokerCapability]

    def get_positions(self) -> pd.DataFrame: ...
    def get_cash(self) -> dict[str, float]: ...
    def get_market_snapshot(self, symbols: list[str]) -> pd.DataFrame: ...
    def place_orders(self, orders: pd.DataFrame) -> pd.DataFrame: ...
    def fetch_fills(self, since: str) -> pd.DataFrame: ...


class PublisherPlugin(Protocol):
    """Sends run results to external destinations (Telegram, Slack, etc.)."""

    meta: PluginMeta

    def publish(self, result: RunResult, params: dict[str, Any]) -> None: ...


class RiskPlugin(Protocol):
    """Validates portfolio targets and orders against risk limits.

    Returns a list of findings (dicts with severity, message, etc.).
    Empty list = all checks passed.
    """

    meta: PluginMeta

    def check_targets(self, targets: pd.DataFrame, params: dict[str, Any]) -> list[dict[str, Any]]: ...
    def check_orders(self, orders: pd.DataFrame, params: dict[str, Any]) -> list[dict[str, Any]]: ...


class StrategyPlugin(Protocol):
    """Computes target portfolio weights from market data.

    Input ``data`` dict contains wide DataFrames: ``prices``, ``volume``,
    ``market_cap``, ``universe``, and optionally ``funding_rates``.
    ``context`` is the run's :class:`StrategyContext`; a strategy reads its
    annualisation from ``context.bars_per_year`` (via
    ``quantbox.strategy_runner.resolve_annualize``), never from its own default.
    A strategy whose ``run`` takes no ``context`` still runs, but is handed the
    deprecated ``_pipeline_annualize`` param instead, with a DeprecationWarning.

    Returns dict with at minimum ``"weights"`` (DataFrame: date index x symbol columns).
    """

    meta: PluginMeta

    def run(
        self, data: dict[str, Any], params: dict[str, Any], context: StrategyContext | None = None
    ) -> dict[str, Any]: ...


class RebalancingPlugin(Protocol):
    """Generates executable orders from target weights + current broker state."""

    meta: PluginMeta

    def generate_orders(
        self,
        *,
        weights: dict[str, float],
        broker: BrokerPlugin,
        params: dict[str, Any],
    ) -> dict[str, Any]: ...


class FeaturePlugin(Protocol):
    """Computes derived features from market data (e.g. momentum, volatility)."""

    meta: PluginMeta

    def compute(self, data: dict[str, pd.DataFrame], params: dict[str, Any]) -> pd.DataFrame: ...


class ValidationPlugin(Protocol):
    """Validates portfolio returns/weights against benchmarks and constraints."""

    meta: PluginMeta

    def validate(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        benchmark: pd.DataFrame | None,
        params: dict[str, Any],
    ) -> dict[str, Any]: ...


class MonitorPlugin(Protocol):
    """Checks run results for anomalies and generates alerts."""

    meta: PluginMeta

    def check(
        self,
        result: RunResult,
        history: list[RunResult] | None,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]: ...


class OverlayPlugin(Protocol):
    """Modifies a base strategy's DECIDED weights before execution (ADR-0004).

    ``weights`` is the aggregated decided book (date x symbol); ``data`` is the
    market-data dict the strategies saw. Row ``t`` of the result may use data up
    to and including bar ``t`` and must stay on row ``t``: an overlay never
    shifts its own output. The run's execution convention
    (``execution.lag_bars``) moves the whole overlaid book onto the fill bar
    afterwards, once, for every overlay alike. The result has the input's
    index and columns; overlays chain in config order.
    """

    meta: PluginMeta

    def apply(self, weights: pd.DataFrame, data: dict[str, Any], params: dict[str, Any]) -> pd.DataFrame: ...


class PipelinePlugin(Protocol):
    """Top-level orchestrator: data loading → strategy → risk → execution → artifacts."""

    meta: PluginMeta
    kind: PipelineKind

    def run(
        self,
        *,
        mode: Mode,
        asof: str,
        params: dict[str, Any],
        data: DataPlugin,
        store: ArtifactStore,
        broker: BrokerPlugin | None,
        risk: list[RiskPlugin],
        strategies: list[StrategyPlugin] | None = None,
        rebalancer: RebalancingPlugin | None = None,
        **kwargs,
    ) -> RunResult: ...
