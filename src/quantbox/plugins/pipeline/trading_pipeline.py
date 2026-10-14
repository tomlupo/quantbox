"""Full trading pipeline plugin.

Orchestrates: strategy execution -> aggregation -> the decision (final
target weights: short clip, gross cap, group limits, ``venue.leverage``
normalisation; :mod:`quantbox.decision`, the backtest's own transform,
TOM-1520) -> the rebalancing policy -> order generation (the injected
rebalancing plugin) -> execution (:mod:`quantbox.trade.executor`) ->
artifact storage.

The pipeline sizes no order itself (TOM-1458): a paper or live run without
``plugins.rebalancing`` is refused, and the orders go to the venue through
:func:`quantbox.trade.executor.execute_orders`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from quantbox.contracts import (
    ArtifactStore,
    BrokerCapability,
    BrokerPlugin,
    DataPlugin,
    Mode,
    PluginMeta,
    RebalancingPlugin,
    RiskPlugin,
    RunResult,
    StrategyPlugin,
)
from quantbox.decision import DecisionRules, final_targets, gross_cap, log_normalisation
from quantbox.engine.groups import GROUP_LIMITS_SCHEMA, GroupLimits, resolve_group_limits
from quantbox.engine.policy import (
    POLICY_SCHEMA,
    LiveDecision,
    RebalancePolicy,
    decide_rebalance,
    next_execution_bar,
    resolve_policy,
    trades_every_bar,
    tranches_alias,
)
from quantbox.financing import resolve_leverage
from quantbox.frequency import _parse_bar_size
from quantbox.strategy_runner import build_strategy_context, decided_weights, latest_weights, run_strategies
from quantbox.trade.book import (
    DEFAULT_RECONCILIATION_TOLERANCE,
    REASON_UNPRICED,
    PortfolioValuation,
    PortfolioValuationError,
    read_book,
)
from quantbox.trade.capabilities import supports
from quantbox.trade.executor import execute_orders, to_fee
from quantbox.trade.notifier import Alerts, Notifier
from quantbox.trade.reconciliation.cycle import open_cycle
from quantbox.trade.reconciliation.ledger import EXEC_STATUS_TO_LEDGER
from quantbox.trade.reconciliation.working_orders import (
    DEFAULT_MAX_AGE_DAYS,
    WorkingOrderStore,
    resolve_working_orders,
)

logger = logging.getLogger(__name__)


def _target_dict(weights: dict[str, float]) -> dict[str, float]:
    """Target weights as the order generators take them: ``{symbol: weight}``, zeros dropped, largest first."""
    s = pd.Series(weights, dtype=float)
    s = s[s != 0].sort_values(ascending=False)
    return {str(k): float(v) for k, v in s.items()}


def _valuation_metrics(valuation: PortfolioValuation | None) -> dict[str, float | None]:
    """Valuation metrics for the run record.

    A missing valuation is reported as ``None`` (this repo's "could not be
    MEASURED" convention, #92) rather than omitted or zeroed: a metric that
    disappears reads as "fine" to every consumer downstream, and "not measured"
    must not look like "measured and clean".
    """
    if valuation is None:
        return {
            "portfolio_value_cash_usd": None,
            "portfolio_value_marked_usd": None,
            "portfolio_value_n_holdings": None,
            "portfolio_value_n_unpriced": None,
            "portfolio_valuation_complete": None,
            "portfolio_value_broker_equity_usd": None,
            "portfolio_value_reconciled": None,
        }
    return dict(valuation.as_metrics())


def _valuation_notes(valuation: PortfolioValuation | None) -> dict[str, Any]:
    """The readable half of the valuation, for the run's notes."""
    if valuation is None:
        return {"portfolio_valuation_state": "unknown"}
    return dict(valuation.as_notes())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CAPITAL_AT_RISK = 1.0
DEFAULT_STABLE_COIN = "USDC"

#: Pipeline-level key only the deleted private order generator read as a floor
#: (TOM-1458). The rebalancing plugin reads its own ``rebalancing.params``, as it
#: always did when one was injected; a run that sets it here is told so. Its
#: siblings ``min_notional`` and ``scaling_factor_min`` left the schema, so
#: ``quantbox validate`` refuses them as unknown.
UNREAD_ORDER_KEYS = ("min_trade_size",)

#: The universe screen's inputs, stored next to ``prices.parquet`` in every run
#: dir (TOM-1420), so a run's selection can be replayed config-only (TOM-1722).
SCREEN_INPUTS = ("volume", "market_cap", "screen_volume")

# Ledger result status mapped from an execution ``orders_details`` row status.
# Backwards-compatible alias. The map itself now lives in
# ``reconciliation.ledger`` beside the vocabulary it targets, because the
# out-of-cycle working-order resolver must translate into the same map and a
# private second copy is how two callers start disagreeing.
_EXEC_STATUS_TO_LEDGER = EXEC_STATUS_TO_LEDGER


def _expand_price_fetch_universe(
    universe: pd.DataFrame,
    broker: Any,
    mode: str,
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    """Union today's screened ``universe`` with current broker holdings.

    ``universe`` is only the NEW, freshly-screened universe for this run. A
    symbol the book still HOLDS can rotate out of it on any given day. If the
    strategy/rebalance price-fetch (``DataPlugin.load_market_data``) only
    covers the screened universe, that held symbol never gets a price: the
    paper/live broker's market snapshot built from that data comes back with
    no quote for it, the rebalancer reads the missing quote as NaN, and the
    (correct) exit order for that symbol is skipped. If enough held symbols
    rotate out this way, EVERY order gets suppressed and the book freezes on
    stale positions (quantbox#120). This mirrors the holdings ∪ universe
    pattern already used by quantbox-live's ``dump_kraken_prices.py``.

    Returns a new DataFrame (or ``universe`` unchanged if there is nothing to
    add) — never mutates the input. Callers should keep using the ORIGINAL
    ``universe`` for eligibility/weights/store, so this cannot change strategy
    sizing or signal logic — it only widens what gets a price fetched.
    """
    if broker is None or mode not in ("paper", "live"):
        return universe
    try:
        # A required BrokerPlugin method: called, never probed (TOM-1459). A
        # broker without it fails here and keeps the universe, as before.
        positions_df = broker.get_positions()
    except Exception:
        return universe
    if positions_df is None or positions_df.empty or "symbol" not in positions_df.columns:
        return universe

    held_symbols = {str(s) for s in positions_df["symbol"].tolist() if s}
    existing = set(universe["symbol"].astype(str)) if not universe.empty and "symbol" in universe.columns else set()
    missing = held_symbols - existing
    if not missing:
        return universe

    extra_rows = pd.DataFrame({"symbol": sorted(missing)})
    for col in universe.columns:
        if col != "symbol":
            extra_rows[col] = pd.NA
    expanded = pd.concat([universe, extra_rows[universe.columns]], ignore_index=True)
    if log is not None:
        log.info(
            "Price-fetch universe expanded with %d held-but-rotated-out symbol(s) so exit orders get a real price: %s",
            len(missing),
            sorted(missing),
        )
    return expanded


def _compute_data_age(prices: pd.DataFrame, asof: str) -> tuple[float | None, float | None]:
    """Return ``(data_age_seconds, bar_interval_seconds)`` for a wide price frame.

    ``data_age_seconds`` is ``asof - latest bar timestamp`` — how stale the feed
    is as of the run. ``bar_interval_seconds`` is the median spacing of the price
    index (the expected bar cadence). Either may be ``None`` when it cannot be
    determined (empty feed, non-datetime index, unparseable ``asof``). Used to
    surface the data-staleness exception signal the notifier consumes (issue
    #62 — the HL 429 feed-gap class).
    """
    if prices is None or getattr(prices, "empty", True):
        return None, None
    idx = prices.index
    if not isinstance(idx, pd.DatetimeIndex):
        # Only coerce a string/object index (e.g. ISO date labels). A numeric
        # index must NOT be coerced — pandas would read ints as 1970-epoch
        # nanoseconds and fabricate a huge, false staleness age.
        if idx.dtype != object:
            return None, None
        # An object-dtype index can still hold NUMERIC labels (Python or numpy
        # ints/floats) — pandas would read those as 1970-epoch nanoseconds and
        # fabricate a huge false staleness. Reject any numeric label element-wise;
        # only genuine date-like labels (strings, dates) are ageable.
        import numbers

        if any(isinstance(x, numbers.Number) and not isinstance(x, bool) for x in idx):
            return None, None
        try:
            idx = pd.DatetimeIndex(pd.to_datetime(idx, errors="raise"))
        except Exception:  # noqa: BLE001 - unparseable labels: cannot age the feed
            return None, None
    if len(idx) == 0:
        return None, None
    try:
        asof_ts = pd.Timestamp(asof)
    except Exception:  # noqa: BLE001 - unparseable asof
        return None, None

    latest = idx.max()
    # Compare naive-to-naive so a tz-aware feed and a naive asof don't raise.
    if latest.tzinfo is not None and asof_ts.tzinfo is None:
        latest = latest.tz_localize(None)
    elif latest.tzinfo is None and asof_ts.tzinfo is not None:
        asof_ts = asof_ts.tz_localize(None)

    age = float((asof_ts - latest).total_seconds())
    interval: float | None = None
    if len(idx) >= 2:
        diffs = pd.Series(idx).diff().dropna()
        if not diffs.empty:
            interval = float(diffs.median().total_seconds())
    return age, interval


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


# Length of the funding reporting window. The pipeline runs daily, so the window
# ENDING at `asof` starts 24h earlier; Hyperliquid settles funding hourly, giving
# ~24 payments per open position per run. See _funding_window for why the end
# bound is `asof` and not the moment of the run.
FUNDING_LOOKBACK = pd.Timedelta(hours=24)


def _resolve_funding_charge(broker: Any, asof: str) -> float | None:
    """The run's funding cost as a SIGNED cash delta, or ``None`` when UNKNOWN.

    Negative means the book paid (see ``broker/_funding.py`` for why that sign,
    and why it is the same one the simulated book already uses).

    Two sources, one meaning:

    * a SIMULATED book applies funding itself and returns what it booked;
    * a LIVE broker reads back what the venue already took over the window.

    A broker with neither yields ``None`` — never 0.0. That distinction is the
    whole of #92: for 165 days a ``hasattr`` miss on a live run rendered as
    ``Funding charge $0.00`` on a perps book that pays funding hourly on its
    full notional, and nothing downstream could tell that from a free window.

    Which source a broker has is its DECLARATION (``FUNDING_APPLIED``,
    ``FUNDING_HISTORY``; TOM-1459), not a probe for the method.
    """
    if broker is None:
        return None

    if supports(broker, BrokerCapability.FUNDING_APPLIED):
        charge = broker.apply_funding()
        logger.info("Applied funding charge: %.2f", charge or 0.0)
        return charge

    if supports(broker, BrokerCapability.FUNDING_HISTORY):
        window_start, window_end = _funding_window(asof)
        charge = broker.fetch_funding_payments(window_start, window_end)
        if charge is None:
            logger.error(
                "Funding UNMEASURED for the window [%s, %s) — reported as UNKNOWN, not $0.00",
                window_start,
                window_end,
            )
        else:
            logger.info("Realised funding in [%s, %s): %.4f (negative = paid)", window_start, window_end, charge)
        return charge

    logger.warning(
        "Broker %s declares no funding source (neither funding_applied nor "
        "funding_history) — funding reported UNKNOWN, not $0.00",
        type(broker).__name__,
    )
    return None


def _funding_window(asof: str) -> tuple[str, str]:
    """The funding reporting window as ISO ``(start, end)``, half-open at the end.

    CLOSED at both ends, because a funding total is only a measurement of the
    window it claims. Asking the venue for everything since the start and
    letting it answer through wall-clock now would report days of payments for
    a historical ``asof`` as that day's funding.

    ``end`` is ``asof`` itself, not the moment of the run: funding accruing
    between ``asof`` and a cron that fires hours later belongs to the NEXT
    report, and the half-open interval means consecutive daily windows abut
    exactly — no gap, no overlap, no payment counted twice.
    """
    try:
        ts = pd.Timestamp(asof)
    except Exception:  # noqa: BLE001 - unparseable asof: fall back to a live window
        ts = None
    if ts is None or pd.isna(ts):
        logger.warning("Unparseable asof %r for the funding window — using the last %s", asof, FUNDING_LOOKBACK)
        ts = pd.Timestamp.utcnow()
    return (ts - FUNDING_LOOKBACK).isoformat(), ts.isoformat()


#: The orders artifact's columns. One definition, because an "empty orders"
#: frame that disagrees with the real one makes every downstream reader raise
#: KeyError on exactly the runs that produced no orders.
ORDER_COLUMNS: tuple[str, ...] = (
    "Asset",
    "Symbol",
    "Action",
    "Raw Quantity",
    "Adjusted Quantity",
    "Price",
    "Notional Value",
    "Min Notional",
    "Min Qty",
    "Step Size",
    "Scaling Factor",
    "Order Status",
    "Reason",
    "Executable",
)


def _empty_order_result() -> dict[str, Any]:
    """What a cycle that refused to size anything produces: no orders, no plan,
    zero notional. Shaped exactly like a real ``generate_orders`` result so every
    reader below (`rebalancing`/`orders`/`total_value`) keeps working."""
    return {
        "rebalancing": pd.DataFrame(),
        "orders": pd.DataFrame(columns=list(ORDER_COLUMNS)),
        "total_value": 0.0,
    }


@dataclass
class TradingPipeline:
    meta = PluginMeta(
        name="trade.full_pipeline.v1",
        kind="pipeline",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Full trading pipeline: strategy execution -> aggregation -> "
            "risk transforms -> order generation -> execution. "
            "Ported from quantlab trading workflow."
        ),
        tags=("trading", "full-pipeline", "crypto"),
        capabilities=("paper", "live", "crypto"),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "strategies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "weight": {"type": "number", "default": 1.0},
                            "params": {"type": "object"},
                        },
                        "required": ["name"],
                    },
                    "description": "List of strategy configs to run.",
                },
                "frequency": {
                    "type": ["string", "object"],
                    "description": (
                        "Bar frequency: '1h' or {bar_size, calendar}. Wins over prices.frequency + "
                        "market_calendar; its bars_per_year is the strategy annualize."
                    ),
                },
                "market_calendar": {
                    "type": "string",
                    "default": "24/7",
                    "description": "Calendar used with prices.frequency when `frequency` is absent (e.g. NYSE).",
                },
                "strategy_weights": {
                    "type": "object",
                    "description": "Override strategy-level weights {name: weight}.",
                },
                "capital_at_risk": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 1.0,
                    "description": "Fraction of portfolio value the target weights are sized against.",
                },
                "stable_coin_symbol": {
                    "type": "string",
                    "default": "USDC",
                    "description": "Cash asset of the book; never traded, excluded from orders.",
                },
                "equity_reconciliation_tolerance": {
                    "type": "number",
                    "default": DEFAULT_RECONCILIATION_TOLERANCE,
                    "description": (
                        "Relative tolerance for the LIVE pre-trade reconciliation between the "
                        "pipeline's own valuation and the broker's get_equity(). A live run whose "
                        "two views of the book disagree by more than this REFUSES to trade. "
                        "Default 0.005 (0.5%) absorbs bid/ask and timing drift between two reads "
                        "of the same book, but not a missing position."
                    ),
                },
                "require_equity_reconciliation": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "Whether a long-only spot book must reconcile against broker equity before "
                        "trading. Turning this off does NOT disable the completeness gate: an "
                        "unmarkable holding still fails a live run."
                    ),
                },
                "risk": {
                    "type": "object",
                    "description": (
                        "The decision's short clip (allow_short) and gross cap (max_leverage) when no rebalancer "
                        "plugin is injected, a backtest-mode run (paper and live need one, TOM-1458; its own "
                        "params are the rules then); also "
                        "handed to risk plugins. tranches: N is DEPRECATED (TOM-1513, TOM-1518): it is "
                        "rebalancing_policy {cadence: tranche, tranches: N}, and warns."
                    ),
                    "properties": {
                        "tranches": {"type": "integer", "minimum": 1, "default": 1},
                        "max_leverage": {"type": "number", "minimum": 0, "default": 1},
                        "allow_short": {"type": "boolean", "default": False},
                    },
                },
                "rebalancing_policy": {
                    **POLICY_SCHEMA,
                    "description": (
                        "The backtest's rebalancing policy, applied live by the same code (TOM-1518, "
                        "quantbox.engine.policy.decide_rebalance): on each run, whether a rebalance is due on the "
                        "last decided bar and to which targets, given the broker's held book. "
                        + POLICY_SCHEMA["description"]
                        + " Live: frequency must be daily, a calendar frequency or a list of dates (an int above 1 "
                        "and null are refused: the live history window moves); with a declared policy its "
                        "min_trade replaces min_trade_size (declaring both is refused). Absent = every run trades "
                        "the last decided row, as before."
                    ),
                },
                "venue": {
                    "type": "object",
                    "description": "The venue, as the decision and the rebalancing policy read it.",
                    "properties": {
                        "leverage": {
                            "type": "string",
                            "enum": ["normalize", "borrow"],
                            "default": "normalize",
                            "description": (
                                "How the decision is normalised (TOM-1520, the backtest's own transform, "
                                "quantbox.decision): normalize (the default) scales a target row whose NET exposure "
                                "is above 1 down to net 1, and caps the rebalancing_policy's buys at the cash; "
                                "borrow keeps the levered row (a margined or perps book declares it)."
                            ),
                        },
                    },
                },
                "group_limits": {
                    **GROUP_LIMITS_SCHEMA,
                    "description": (
                        "The backtest's group limits, in the same decision (TOM-1520): "
                        + GROUP_LIMITS_SCHEMA["description"]
                    ),
                },
                "min_trade_size": {
                    "type": "number",
                    "minimum": 0,
                    "default": 0.01,
                    "description": (
                        "Not a floor here since TOM-1458 (a run that sets it is warned): the order-level floor "
                        "is the rebalancing plugin's rebalancing.params.min_trade_size. Beside a declared "
                        "rebalancing_policy it is refused (its min_trade is the floor)."
                    ),
                },
                "trading_enabled": {
                    "type": "boolean",
                    "default": True,
                    "description": "False computes targets and orders but sends nothing to the broker.",
                },
                "universe": {
                    "type": "object",
                    "default": {},
                    "description": (
                        "Universe selection params for the data plugin's load_universe(); also carries "
                        "token_policy / token_policy_file."
                    ),
                },
                "prices": {
                    "type": "object",
                    "default": {"lookback_days": 365},
                    "description": "Market-data request passed to the data plugin's load_market_data().",
                },
                "book_key": {
                    "type": "string",
                    "description": "Book identity namespacing the token-policy store, working orders and recon ledger.",
                },
                "data_dir": {
                    "type": "string",
                    "default": "data",
                    "description": "Root for per-book state (working orders) when reconciliation.data_dir is unset.",
                },
                "staleness_factor": {
                    "type": "number",
                    "default": 2.0,
                    "description": "Market data older than this many bar intervals marks the run data_stale.",
                },
                "reconciliation": {
                    "type": "object",
                    "description": (
                        "Order/fill reconciliation block: book_key, data_dir, mode (observe|enforce), "
                        "tolerances, enforce_acknowledged. Absent = no reconciliation."
                    ),
                },
                "exclusions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Assets to exclude from trading.",
                },
            },
            "required": ["strategies"],
        },
        inputs=(),
        outputs=(
            "strategy_weights",
            "aggregated_weights",
            "targets",
            "rebalancing",
            "orders",
            "fills",
            "portfolio_daily",
            "trade_history",
        ),
        examples=(
            "plugins:\n  pipeline:\n    name: trade.full_pipeline.v1\n    params:\n"
            "      strategies:\n        - name: crypto_trend\n          weight: 1.0\n"
            "      risk:\n        tranches: 1\n        max_leverage: 1\n        allow_short: false\n"
            "      capital_at_risk: 1.0\n      stable_coin_symbol: USDC",
        ),
    )
    kind = "trading"

    # The one alert channel's notifier (TOM-1720): injected per run as
    # `run(notifier=...)`, or on an instance by setting `pipeline.notifier` (a fake in
    # tests); None reads TELEGRAM_TOKEN / TELEGRAM_CHAT_ID. ClassVar only so it is not
    # a constructor param that YAML could set; the class default is never assigned.
    # Every alert goes through `_alert`, on every venue — never through the broker.
    notifier: ClassVar[Notifier | None] = None
    _alerts: ClassVar[Alerts | None] = None

    def _alert_channel(self) -> Alerts:
        if self._alerts is None:
            self._alerts = Alerts(self.notifier) if self.notifier is not None else Alerts.from_env()
        return self._alerts

    def _alert(self, message: str, *, site: str) -> bool:
        """Send one alert; never raises (a failed send is logged and recorded)."""
        return self._alert_channel().send(message, site=site)

    # ==================================================================
    # Main entry point
    # ==================================================================
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
    ) -> RunResult:
        if mode in ("paper", "live") and broker is None:
            raise ValueError("broker_required_for_paper_or_live")
        if mode in ("paper", "live") and rebalancer is None:
            # Q4 (Tom, 2026-10-10, TOM-1458): the pipeline's private order
            # generator is deleted and there is no default plugin. Refused
            # before anything is read or stored, so the error is the run's only
            # outcome.
            raise ValueError(
                f"{self.meta.name} in {mode} mode needs plugins.rebalancing: the pipeline sizes no "
                "order itself (TOM-1458). Declare rebalancing.standard.v1 for a spot book or "
                "rebalancing.futures.v1 for a perps or margin book."
            )
        unread = [key for key in UNREAD_ORDER_KEYS if key in params]
        if unread:
            logger.warning(
                "pipeline params %s: not a floor here; the rebalancing plugin sizes the orders from its own "
                "rebalancing.params (TOM-1458)",
                ", ".join(unread),
            )
        # A fresh alert record per run; the channel opens on the live path below. A
        # `run(notifier=...)` serves THIS run only — it never sticks to the pipeline.
        injected = kwargs.get("notifier")
        self._alerts = Alerts(injected) if injected is not None else None

        # Run-state exception inputs the Tier-2 notifier consumes (issue #62):
        # API errors caught (but survived) during the run. Data-staleness and
        # pipeline-failure are derived later from the feed + the run completing.
        api_errors: list[dict[str, Any]] = []
        # Set when a refused portfolio valuation cost us this cycle's orders. It
        # is NOT a clean run: `pipeline_ok` below reads it, so the Tier-2
        # pipeline-failure alert fires even though run() returned normally.
        valuation_refusal: PortfolioValuationError | None = None

        # Resolve strategies config: from injected plugins or from pipeline params
        strategies_cfg = params.get("_strategies_cfg", params.get("strategies", []))
        if not strategies and not strategies_cfg:
            raise ValueError("params.strategies is required and must be non-empty")

        # Resolve aggregator and rebalancer config from runner injection
        aggregator_cfg = params.get("_aggregator_cfg", {})
        rebalancer_cfg = params.get("_rebalancer_cfg", {})

        # --- Stage 1: Universe & Prices ---
        universe_params = params.get("universe", {})
        # Copy + inject run mode so mode-aware data sources (universe-screen
        # market_cap / screen_volume) pick the live snapshot vs the point-in-time
        # backtest path. Run mode is authoritative; never mutate the config dict.
        prices_params = {**params.get("prices", {"lookback_days": 365}), "mode": mode}
        universe = data.load_universe(universe_params)

        # Token policy filtering (universe scope). Derive book_key the same way
        # the reconciliation block does (line ~2034) so the seen-token store is
        # namespaced per book (issue #86).
        book_key = str(params.get("book_key") or "default")
        token_policy = self._load_token_policy(universe_params, book_key)
        token_policy_notes: dict[str, Any] = {}
        if token_policy:
            all_symbols = universe["symbol"].tolist()

            # Detect new tokens BEFORE filtering (uses raw universe)
            new_tokens = self._detect_new_tokens(token_policy, universe)
            if new_tokens:
                alert_text = token_policy.format_new_token_alert(new_tokens)
                logger.warning("NEW TOKENS DETECTED:\n%s", alert_text)
                token_policy_notes = {
                    "new_tokens": new_tokens,
                    "new_token_alert": alert_text,
                }

            # Filter universe to allowed tokens only
            allowed = token_policy.filter_allowed(all_symbols)
            denied = token_policy.filter_denied(all_symbols)

            if denied:
                logger.info(
                    "TokenPolicy denied %d symbols: %s",
                    len(denied),
                    [d[0] for d in denied[:5]],
                )

            universe = universe[universe["symbol"].isin(allowed)].reset_index(drop=True)
            logger.info(
                "TokenPolicy: %d/%d symbols allowed (mode=%s)",
                len(allowed),
                len(all_symbols),
                token_policy.mode,
            )

        # Widen the PRICE-FETCH universe (not the strategy-facing `universe`) to
        # also cover current broker holdings — see `_expand_price_fetch_universe`
        # for why (quantbox#120). Only `fetch_universe` is widened; `universe`
        # (used for eligibility/weights/store below) is untouched, so this
        # cannot change strategy sizing or signal logic.
        fetch_universe = _expand_price_fetch_universe(universe, broker, mode, logger)

        market_data_dict = data.load_market_data(fetch_universe, asof, prices_params)
        store.put_parquet("universe", universe)
        # Store prices, and the universe screen's inputs beside them (TOM-1420),
        # as wide-format parquet. With `volume`, `market_cap` and `screen_volume`
        # a run's top-N selection can be replayed from its run dir alone: Kraken
        # serves only 720 hourly candles, so a screen older than ~30 days could
        # not be rebuilt from the venue (TOM-1411). A key the data plugin did not
        # return is stored empty, as `prices` always was.
        for key in ("prices", *SCREEN_INPUTS):
            wide = market_data_dict.get(key, pd.DataFrame())
            if not isinstance(wide, pd.DataFrame) or wide.empty:
                store.put_parquet(key, pd.DataFrame())
            else:
                store.put_parquet(key, wide.reset_index() if isinstance(wide.index, pd.DatetimeIndex) else wide)

        # Build data dict for strategies (quantlab convention)
        market_data = self._build_market_data(market_data_dict, universe)

        # --- Stage 2: Strategy Execution ---
        # The ONE strategy runner, shared with the backtest pipeline: the same
        # StrategyContext (bars_per_year, as-of, calendar, frequency) reaches the
        # strategy here as in its backtest, so vol sizing cannot differ by
        # sqrt(365/252) between them (TOM-1338, TOM-1448).
        context = build_strategy_context(mode, asof, params, prices_params)
        strategy_results = run_strategies(market_data, strategies_cfg, context, plugins=strategies)

        # Save per-strategy weights
        strat_weights_records: list[dict[str, Any]] = []
        for sname, sinfo in strategy_results.items():
            w = sinfo["result"].get("weights")
            if w is not None and not w.empty:
                last_row = w.iloc[-1] if isinstance(w, pd.DataFrame) else w
                for ticker, wt in last_row.items():
                    strat_weights_records.append({"strategy": sname, "symbol": str(ticker), "weight": float(wt)})
        a_strat_w = store.put_parquet("strategy_weights", pd.DataFrame(strat_weights_records))

        # --- Stage 3: Strategy Aggregation ---
        # The decided weights HISTORY: the strategy runner's one aggregation (TOM-1712), or an
        # injected aggregator's `weights` frame. The rebalancing policy reads the history
        # (TOM-1518); orders trade to its last row.
        _aggregator = kwargs.get("aggregator")
        if _aggregator is not None:
            agg_data = {**market_data, "strategy_results": strategy_results}
            agg_result = _aggregator.run(data=agg_data, params=aggregator_cfg.get("params", {}))
            w_df = agg_result.get("weights")
            agg_history = w_df if isinstance(w_df, pd.DataFrame) else pd.DataFrame()
            final_weights = agg_result.get("simple_weights") or latest_weights(agg_history)
        else:
            agg_history = decided_weights(strategy_results, params.get("strategy_weights"))
            final_weights = latest_weights(agg_history)
        agg_records = [{"symbol": str(k), "weight": float(v)} for k, v in final_weights.items()]
        a_agg_w = store.put_parquet("aggregated_weights", pd.DataFrame(agg_records))
        logger.info("Aggregated weights: %d assets", len(final_weights))

        # --- Inject latest prices into paper broker (if it declares PRICE_INJECTION) ---
        if supports(broker, BrokerCapability.PRICE_INJECTION):
            wide_prices = market_data.get("prices", pd.DataFrame())
            if not wide_prices.empty:
                latest = wide_prices.iloc[-1].dropna()
                broker.set_prices({str(k): float(v) for k, v in latest.items()})
                logger.info("Injected %d prices into broker", len(latest))

        # --- Inject funding rates into broker (if it declares FUNDING_RATE_INJECTION) ---
        wide_funding = market_data.get("funding_rates", pd.DataFrame())
        if not wide_funding.empty and supports(broker, BrokerCapability.FUNDING_RATE_INJECTION):
            latest_funding = wide_funding.iloc[-1].dropna()
            broker.set_funding_rates({str(k): float(v) for k, v in latest_funding.items()})
            logger.info("Injected %d funding rates into broker", len(latest_funding))

        # --- Inject position limits into broker (if it declares POSITION_LIMIT_INJECTION) ---
        # The `_fetcher` probes below are on the DATA plugin, not the broker.
        if supports(broker, BrokerCapability.POSITION_LIMIT_INJECTION) and hasattr(data, "_fetcher"):
            fetcher = data._fetcher
            if hasattr(fetcher, "get_position_limits"):
                limit_symbols = universe["symbol"].tolist() if not universe.empty else []
                if limit_symbols:
                    limits = fetcher.get_position_limits(limit_symbols)
                    if limits:
                        broker.set_position_limits(limits)
                        logger.info("Injected %d position limits into broker", len(limits))

        # --- Stage 4: the decision + the rebalancing policy + Stage 5: Order Generation ---
        # The decision is the backtest's own transform (quantbox.decision, TOM-1520): the
        # short clip, the gross cap, the group limits, then venue.leverage normalisation —
        # the target weights are FINAL before the policy reads them. The policy is the
        # backtest's too (TOM-1518): cadence x trigger + min_trade, on the final history
        # and the broker's held book. A policy that trades every bar to the last decided
        # row (the default, and every config without `rebalancing_policy` or tranches)
        # decides on that row alone.
        policy = self._live_policy(params, rebalancer_cfg)
        leverage = resolve_leverage((params.get("venue") or {}).get("leverage"))
        rebal_params = (
            self._rebalancer_params(
                rebalancer_cfg=rebalancer_cfg,
                params=params,
                strategy_results=strategy_results,
                mode=mode,
                policy=policy,
            )
            if rebalancer is not None
            else {}
        )
        rules = self._decision_rules(params, rebalancer, rebal_params, leverage, universe)
        policy_decision: LiveDecision | None = None
        decision_report: dict[str, Any] | None = None
        live_book = mode not in ("backtest",) and broker is not None
        order_result: dict[str, Any] | None = None
        if not trades_every_bar(policy):
            history = agg_history
            if history.empty:
                raise ValueError(
                    f"rebalancing_policy {policy.policy!r} reads the decided weights HISTORY, and the aggregation "
                    "returned none (an injected aggregator must return a `weights` frame)"
                )
            history, decision_report = final_targets(history.sort_index(), rules)
            log_normalisation(decision_report)
            next_bar = next_execution_bar(
                pd.Timestamp(history.index[-1]),
                _parse_bar_size(context.frequency),
                policy.calendar or context.calendar,
            )
            if live_book and rebalancer is not None:

                def generate(weights: dict[str, float]) -> dict[str, Any]:
                    return rebalancer.generate_orders(weights=weights, broker=broker, params=rebal_params)

                book_capital = float(rebal_params.get("capital_at_risk", DEFAULT_CAPITAL_AT_RISK))
                try:
                    policy_decision, order_result = self._policy_orders(
                        policy=policy,
                        history=history,
                        generate=generate,
                        capital_at_risk=book_capital,
                        next_bar=next_bar,
                        leverage=leverage,
                    )
                    final_weights = order_result.get("weights", final_weights)
                except PortfolioValuationError as exc:  # see the rebalancer arm below
                    valuation_refusal = exc
                    api_errors.append({"stage": "portfolio_valuation", "error": str(exc)})
                    logger.error(
                        "Portfolio valuation REFUSED (%s) -- sizing nothing this cycle; bookkeeping stages still run",
                        exc,
                    )
                    order_result = _empty_order_result()
            else:
                # No broker: no held book. The cadence and the frequency still decide the targets.
                policy_decision = decide_rebalance(policy, history, None, next_bar=next_bar, leverage=leverage)
                held_or_targets = policy_decision.targets if policy_decision.targets is not None else pd.Series()
                final_weights = _target_dict({str(k): float(v) for k, v in held_or_targets.items()})
        else:
            # Every bar trades to the last decided row: the decision on that row alone.
            final_weights, decision_report = self._final_row(final_weights, rules)
        if trades_every_bar(policy) and rebalancer is not None and live_book:
            # The injected rebalancer turns the final targets into orders (its own caps re-applied: a no-op).
            try:
                order_result = rebalancer.generate_orders(
                    weights=final_weights,
                    broker=broker,
                    params=rebal_params,
                )
                final_weights = order_result.get("weights", final_weights)
            except PortfolioValuationError as exc:
                # A refused valuation means we must not SIZE anything -- that is
                # the whole point of the gate. It must not mean the run dies
                # here: Stage 6c (resolving orders a previous cycle left WORKING)
                # and Stage 7b (recon-state persist) both sit BELOW this line,
                # and 6c's own guard cannot catch a raise that never reaches it.
                # A live book whose limit order filled overnight would then never
                # book the fill, and working_orders.DEFAULT_MAX_AGE_DAYS drops the
                # record after 7 days -- losing a real fill, which that queue
                # exists to prevent.
                #
                # So: no orders, loudly, and the rest of the cycle still runs.
                valuation_refusal = exc
                api_errors.append({"stage": "portfolio_valuation", "error": str(exc)})
                logger.error(
                    "Portfolio valuation REFUSED (%s) -- sizing nothing this cycle; bookkeeping stages still run",
                    exc,
                )
                order_result = _empty_order_result()
        policy_notes: dict[str, Any] = (
            {"rebalancing_policy": policy_decision.record()} if policy_decision is not None else {}
        )
        if decision_report is not None:
            # What the decision did to the target weights (TOM-1520): the rules and the rows it scaled.
            policy_notes["decision"] = decision_report

        if mode == "backtest" or broker is None:
            # In backtest mode, just save targets with no execution
            targets = pd.DataFrame([{"symbol": s, "weight": w, "asof": asof} for s, w in final_weights.items()])
            a_targets = store.put_parquet("targets", targets)
            empty_orders = pd.DataFrame(columns=["symbol", "side", "qty", "price", "asof"])
            a_orders = store.put_parquet("orders", empty_orders)
            a_fills = store.put_parquet("fills", pd.DataFrame(columns=["symbol", "side", "qty", "price"]))
            a_rebal = store.put_parquet("rebalancing", pd.DataFrame())
            portfolio_daily = pd.DataFrame([{"asof": asof, "cash_usd": 0.0, "portfolio_value_usd": 0.0}])
            a_port = store.put_parquet("portfolio_daily", portfolio_daily)
            a_trade = store.put_json("trade_history", {"trades": []})

            return RunResult(
                run_id=store.run_id,
                pipeline_name=self.meta.name,
                mode=mode,
                asof=asof,
                artifacts={
                    "strategy_weights": a_strat_w,
                    "aggregated_weights": a_agg_w,
                    "targets": a_targets,
                    "rebalancing": a_rebal,
                    "orders": a_orders,
                    "fills": a_fills,
                    "portfolio_daily": a_port,
                    "trade_history": a_trade,
                },
                metrics={
                    "n_strategies": float(len(strategy_results)),
                    "n_assets": float(len(final_weights)),
                },
                notes={"kind": "trading", **token_policy_notes, **policy_notes},
            )

        # Live / paper execution path. Open the alert channel now, so "alerts off"
        # is logged once at the start of every run that could alert, not at the
        # first alert (which a quiet day never reaches).
        self._alert_channel()
        stable_coin = str(params.get("stable_coin_symbol", DEFAULT_STABLE_COIN))

        if order_result is None:
            # The rebalancer sized the orders above on every paper or live run;
            # nothing here sizes them instead (TOM-1458).
            raise ValueError(f"{self.meta.name}: no order generator ran for mode {mode!r}; declare plugins.rebalancing")
        rebalancing_df = order_result["rebalancing"]
        orders_df = order_result["orders"]
        total_value = order_result["total_value"]

        a_targets_data = []
        for sym, wt in final_weights.items():
            a_targets_data.append({"symbol": sym, "weight": wt, "asof": asof})
        targets = pd.DataFrame(a_targets_data)
        a_targets = store.put_parquet("targets", targets)
        a_rebal = store.put_parquet("rebalancing", rebalancing_df)
        a_orders = store.put_parquet("orders", orders_df)

        # --- Stage 6: Risk Checks ---
        risk_findings: list[dict[str, Any]] = []
        for rp in risk:
            try:
                risk_params = params.get("_risk_cfg", params.get("risk", {}))
                risk_findings.extend(rp.check_targets(targets, risk_params))
                exec_orders = (
                    orders_df[orders_df.get("Executable", pd.Series(dtype=bool))].copy()
                    if "Executable" in orders_df.columns
                    else orders_df
                )
                risk_findings.extend(rp.check_orders(exec_orders, risk_params))
            except Exception as exc:
                logger.warning("Risk check failed: %s", exc)

        # --- Stage 7a: Reconciliation PRE-EXECUTION gate (issue #90) ---
        # Load the per-cycle recon context (if a `reconciliation` block is set) and
        # read the persisted prior-cycle state to derive the pre-execution gate. In
        # enforce mode this gate can HALT (send nothing) or FLATTEN (reduce-only)
        # THIS cycle's orders BEFORE they are executed — the real, crash-durable
        # enforcement touchpoint. In observe mode the gate is computed but never
        # applied. Guarded so a recon fault fails safe to no-gate (never fatal),
        # and CLOSED under acknowledged enforce (quantbox.trade.reconciliation.cycle).
        recon = open_cycle(params, run_id=store.run_id, asof=asof, alert=self._alert)

        # --- Stage 6c: resolve orders left WORKING by a previous cycle ---
        # A priced order goes to the venue as a LIMIT order and may rest on the
        # book long after the run that placed it exited. Those are queued rather
        # than alarmed; this is where the book finds out what became of them, so a
        # fill that landed minutes (or hours) later still reaches the accounts.
        # Guarded: resolution is bookkeeping and must never block trading.
        working_store = None
        resolved_working: list[dict[str, Any]] = []
        try:
            working_store = WorkingOrderStore.for_book(params)
            if working_store is not None:
                # The semantics (unreadable venue keeps the order queued, still
                # working keeps it, terminal books it against the carried
                # order_ref, past the cap drops it loudly) are the queue's own,
                # shared with quantbox-live's out-of-cycle job.
                resolved_working = resolve_working_orders(
                    working_store,
                    broker,
                    ledger=recon.ledger,
                    max_age_days=DEFAULT_MAX_AGE_DAYS,
                ).resolved
        except Exception as exc:  # noqa: BLE001 - bookkeeping must never be fatal
            logger.error("Working-order resolution failed (non-fatal): %s", exc)

        # --- Stage 7: Execution (quantbox.trade.executor, TOM-1458) ---
        trading_enabled = bool(params.get("trading_enabled", True))
        execution_report = execute_orders(
            broker=broker,
            orders_df=orders_df,
            trading_enabled=trading_enabled,
            alert=self._alert,
            # Submission-time intent capture: thread the SAME ledger so intent is
            # recorded at the broker call (crash-durable), and Stage 7b reads it back.
            ledger=recon.ledger,
            cycle_id=recon.cycle_id,
            # Enforce-mode pre-execution gate (permissive defaults when not applied).
            gate_orders_allowed=recon.orders_allowed,
            gate_reduce_only=recon.reduce_only,
            # Under enforce, a failed intent write drops the order (fail closed) —
            # never trade live without the durable intent record.
            capture_fail_closed=recon.capture_fail_closed,
            working_store=working_store,
        )

        fills_data = []
        for detail in execution_report.get("orders_details", []):
            if detail.get("status") in ("FILLED", "PARTIAL"):
                fills_data.append(
                    {
                        "symbol": detail.get("symbol", ""),
                        "side": str(detail.get("side", detail.get("action", ""))).lower(),
                        "qty": float(detail.get("executed_quantity", 0)),
                        "price": float(detail.get("executed_price", 0)),
                    }
                )
        fills = pd.DataFrame(fills_data) if fills_data else pd.DataFrame(columns=["symbol", "side", "qty", "price"])
        a_fills = store.put_parquet("fills", fills)

        # Funding — a first-order cost on a perps book, charged every hour on
        # the full notional.
        #
        # #92, same defect as cumulative_fees below: `apply_funding` exists ONLY
        # on futures_paper, so every LIVE run fabricated a $0.00 funding charge.
        # Fixing that to None stopped the lie but measured nothing; a LIVE
        # broker that can read the venue's funding history is now asked for it.
        # Both paths return the same thing: a SIGNED cash delta on the account,
        # negative when the book paid (see broker/_funding.py). Unmeasured stays
        # None, never 0.0 — the report renders None as UNKNOWN.
        funding_charge = _resolve_funding_charge(broker, asof)

        # Portfolio snapshot -- prefer broker.get_equity() for derivatives
        # brokers where cash + sum(qty * price) is wrong for short positions.
        portfolio_value_post = total_value
        cash_usd_post = 0.0
        try:
            # CASH FIRST, on both arms. `get_equity()` now RAISES on a book it
            # cannot fully mark, and this whole block is best-effort — so
            # reading equity first left `cash_usd_post` at its 0.0 initialiser
            # and wrote that false zero to `portfolio_daily`, where nothing
            # downstream can tell it from a genuinely empty account.
            # `alloc2orders` was reordered for exactly this; this twin was not.
            cash2 = broker.get_cash() or {}
            cash_usd_post = sum(float(v) for v in cash2.values())
            if supports(broker, BrokerCapability.EQUITY):
                portfolio_value_post = float(broker.get_equity())
            else:
                # The marked book after the orders (lots summed, one snapshot of
                # the held names). An unmarkable holding is worth an UNKNOWN
                # amount, not nothing: it is reported, and this NAV is the
                # marked part. A snapshot with no `mid` used to make the NAV
                # plain cash, silently; it now reports every holding unmarked.
                post_book = replace(read_book(broker), cash=cash_usd_post)
                if post_book.held and post_book.snapshot is None:
                    # The snapshot itself failed: that is a failed read, not a
                    # book of unmarked names. Recording cash alone would write a
                    # false NAV, so the pre-trade value stays, as it always did
                    # on a failed read, and the error is recorded below.
                    raise RuntimeError("post-trade market snapshot failed; NAV not re-marked")
                post_snapshot = post_book.marked(
                    # Same two arguments as the PRE-trade call. Without them the
                    # stable coin and every excluded asset read as unmarkable,
                    # and a healthy run would log an understatement and push an
                    # api_error every time.
                    stable_coin=stable_coin,
                    exclusions=list(params.get("exclusions", [])),
                )
                portfolio_value_post = post_snapshot.value
                if post_snapshot.unpriced:
                    logger.error(
                        "Post-trade NAV is UNDERSTATED: %d of %d holding(s) unmarkable (%s).",
                        len(post_snapshot.unpriced),
                        post_snapshot.n_holdings,
                        ", ".join(post_snapshot.unpriced),
                    )
                    api_errors.append(
                        {
                            "stage": "portfolio_snapshot",
                            "error": f"unmarkable holdings: {', '.join(post_snapshot.unpriced)}",
                        }
                    )
        except Exception as exc:  # noqa: BLE001 - snapshot is best-effort, never fatal
            # A broker read failing here is a (survived) API error — surface it as
            # an exception input for the notifier rather than swallowing it (#62).
            logger.warning("Portfolio snapshot read failed: %s", exc)
            api_errors.append({"stage": "portfolio_snapshot", "error": str(exc)})

        portfolio_daily = pd.DataFrame(
            [
                {
                    "asof": asof,
                    "cash_usd": float(cash_usd_post),
                    "portfolio_value_usd": float(portfolio_value_post),
                }
            ]
        )
        a_port = store.put_parquet("portfolio_daily", portfolio_daily)

        # --- Stage 7b: Reconciliation break evaluation + state persist (issue #87/#90) ---
        # Classifies breaks + computes the NORMAL→DEGRADED→HALT/FLATTEN transition
        # and PERSISTS the resulting state. Observe by default (logs/alerts only);
        # in enforce mode the persisted state is what NEXT cycle's pre-execution
        # gate (Stage 7a) reads back to actually halt/flatten. Reads the ledger
        # captured at submission (Stage 7). No-op unless a `reconciliation` block is
        # present. Runs after the portfolio snapshot so drift/phantom are computed
        # against the reconciled real book (authority: exchange = truth for holdings).
        # Never fatal, except an unwritable enforcement authority under enforce.
        recon_notes = recon.close(
            broker=broker,
            final_weights=final_weights,
            orders_df=orders_df,
            execution_report=execution_report,
            portfolio_value=portfolio_value_post,
            stable_coin=stable_coin,
            alert=self._alert,
        )

        # Collect fee/funding metrics from broker.
        #
        # #92: this used to initialise to 0.0 and only overwrite it when the
        # broker carried `_cumulative_fees` — an attribute only the SIM brokers
        # have. Every LIVE run therefore reported a FABRICATED $0.00: 165 days
        # of it on a book that turned over ~24x its equity, which made the
        # TOM-33 loss attribution unresolvable. An unmeasured cost is UNKNOWN,
        # not free, so it is None here and null downstream. A broker that
        # measures its fees DECLARES it (FEES_ACCRUED, TOM-1459).
        cumulative_fees: float | None = None
        if supports(broker, BrokerCapability.FEES_ACCRUED):
            cumulative_fees = float(broker.cumulative_fees)

        # --- Stage 8: Artifacts ---
        # The simulated cost model, recorded only for a broker that declares one.
        broker_costs: dict[str, Any] = {}
        if supports(broker, BrokerCapability.COST_MODEL):
            broker_costs = {
                "spread_bps": float(broker.spread_bps),
                "slippage_bps": float(broker.slippage_bps),
                "maker_fee_bps": float(broker.maker_fee_bps),
                "taker_fee_bps": float(broker.taker_fee_bps),
            }

        artifact_payload = self._build_artifact_payload(
            rebalancing_df=rebalancing_df,
            orders_df=orders_df,
            execution_report=execution_report,
            final_weights=final_weights,
            total_value=total_value,
            mode=mode,
            funding_charge=funding_charge,
            cumulative_fees=cumulative_fees,
            broker_costs=broker_costs,
        )
        a_trade = store.put_json(
            "trade_history",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": mode,
                "trades": artifact_payload.get("execution_summary", {}).get("executed_orders", []),
            },
        )

        # --- Exception inputs for the Tier-2 notifier (issue #62) ---
        # Three previously-dormant alert conditions need their input signals
        # produced by the run: data-staleness (feed older than Nx bar interval —
        # the HL 429 feed-gap class), API-error (a survived broker/data API
        # failure), and pipeline-failure (a run crash; surfaced by run() raising
        # before this point, so a returned RunResult always has pipeline_ok=True).
        api_errors.extend(execution_report.get("api_errors", []))
        prices_for_age = market_data.get("prices", pd.DataFrame())
        data_age_seconds, bar_interval_seconds = _compute_data_age(prices_for_age, asof)
        staleness_factor = float(params.get("staleness_factor", 2.0))
        data_stale = (
            data_age_seconds is not None
            and bar_interval_seconds is not None
            and bar_interval_seconds > 0
            and data_age_seconds > staleness_factor * bar_interval_seconds
        )
        exception_signals: dict[str, Any] = {
            "pipeline_ok": valuation_refusal is None,
            "data_age_seconds": data_age_seconds,
            "bar_interval_seconds": bar_interval_seconds,
            "staleness_factor": staleness_factor,
            "data_stale": bool(data_stale),
            "api_errors": list(api_errors),
            "api_error_count": len(api_errors),
        }
        if data_stale:
            logger.warning(
                "DATA STALE: feed latest bar is %.0fs old (> %.1fx the %.0fs bar interval). "
                "Possible feed gap (e.g. rate-limit 429) — exception signal raised.",
                data_age_seconds,
                staleness_factor,
                bar_interval_seconds,
            )

        metrics = {
            "n_strategies": float(len(strategy_results)),
            "n_assets": float(len(final_weights)),
            "portfolio_value_usd_pre": float(total_value),
            "portfolio_value_usd_post": float(portfolio_value_post),
            # Keeps "nothing held", "fully marked" and "could not price" apart
            # in the run record. Before this, all three produced the same
            # number and nothing said which one had happened.
            **_valuation_metrics(order_result.get("valuation")),
            "n_orders": float(len(orders_df)),
            "n_fills": float(len(fills)),
            "total_executed": float(execution_report.get("summary", {}).get("total_executed", 0)),
            "total_partial": float(execution_report.get("summary", {}).get("total_partial", 0)),
            "total_failed": float(execution_report.get("summary", {}).get("total_failed", 0)),
            "total_working": float(execution_report.get("summary", {}).get("total_working", 0)),
            "resolved_working": float(len(resolved_working)),
            "funding_charge": (float(funding_charge) if funding_charge is not None else None),
            "cumulative_fees": cumulative_fees,
            # Dead-man health signal: 1.0 means the strategy wanted to rebalance
            # but every order was suppressed (book frozen on stale positions).
            # Monitors/dashboards should alarm on rebalance_frozen == 1.
            "rebalance_frozen": 1.0 if execution_report.get("frozen") else 0.0,
            # Quiet-day signal: 1.0 means all-cash with every target leg sub-min
            # (staying in cash). This is INFO-level, NOT a freeze — monitors must
            # NOT alarm on it.
            "quiet_day": 1.0 if execution_report.get("quiet_day") else 0.0,
            # Residual-day signal (TOM-1778): nothing sent because every exit was a
            # partial reduce below the venue minimum. INFO-level, NOT a freeze.
            "residual_day": 1.0 if execution_report.get("residual_day") else 0.0,
            "residual_count": float(len(execution_report.get("residuals", []))),
            # Tier-2 exception inputs (#62).
            "data_stale": 1.0 if data_stale else 0.0,
            "data_age_seconds": float(data_age_seconds) if data_age_seconds is not None else -1.0,
            "api_error_count": float(len(api_errors)),
        }

        return RunResult(
            run_id=store.run_id,
            pipeline_name=self.meta.name,
            mode=mode,
            asof=asof,
            artifacts={
                "strategy_weights": a_strat_w,
                "aggregated_weights": a_agg_w,
                "targets": a_targets,
                "rebalancing": a_rebal,
                "orders": a_orders,
                "fills": a_fills,
                "portfolio_daily": a_port,
                "trade_history": a_trade,
            },
            metrics=metrics,
            notes={
                "kind": "trading",
                "risk_findings": risk_findings,
                "artifact_payload": artifact_payload,
                **_valuation_notes(order_result.get("valuation")),
                "rebalance_frozen": bool(execution_report.get("frozen", False)),
                "freeze_reasons": execution_report.get("freeze_reasons", {}),
                # Per-order freeze detail (symbol/side/notional/min/shortfall).
                # freeze_reasons alone is a histogram — it says three orders died,
                # never which or by how much, so every page started with a log dig.
                "frozen_orders": execution_report.get("frozen_orders", []),
                "quiet_day": bool(execution_report.get("quiet_day", False)),
                "quiet_reasons": execution_report.get("quiet_reasons", {}),
                # Partial reduces below the venue minimum, not sent (TOM-1778).
                "residual_day": bool(execution_report.get("residual_day", False)),
                "residuals": execution_report.get("residuals", []),
                # This cycle's order failures (#87) — a first-class exception input
                # the notifier must alert on, from every exit of execute_orders.
                "order_failures": int(execution_report.get("order_failures", 0)),
                "order_failure_detail": execution_report.get("order_failure_detail", ""),
                # Tier-2 exception inputs the notifier threads into BookContext (#62).
                "exceptions": exception_signals,
                # Reconciliation-break enforcement (issue #87). Observe-mode by
                # default: populated with the classified breaks + would-be action,
                # but no order gating happens here.
                "reconciliation": recon_notes,
                # The alert channel's record (TOM-1720): on/off and why, every
                # alert sent, failed or dropped — an alert is never silent.
                "alerts": self._alert_channel().record(),
                **token_policy_notes,
                # What the rebalancing policy decided this run (TOM-1518): absent when
                # the policy trades every bar to the last decided row.
                **policy_notes,
            },
        )

    # ==================================================================
    # Stage 1 helper: build market data dict for strategies
    # ==================================================================
    def _build_market_data(
        self,
        market_data_dict: dict[str, pd.DataFrame],
        universe: pd.DataFrame,
    ) -> dict[str, Any]:
        """Build the data dict that strategies expect.

        Takes the wide-format dict from ``DataPlugin.load_market_data()``
        and adds universe + defaults for missing keys.
        """
        result: dict[str, Any] = {"universe": universe}
        result.update(market_data_dict)
        for key in ("prices", "volume", "market_cap", "funding_rates"):
            result.setdefault(key, pd.DataFrame())
        return result

    # ==================================================================
    # Stage 4: Rebalancing policy (TOM-1518)
    # ==================================================================
    def _live_policy(self, params: dict[str, Any], rebalancer_cfg: dict[str, Any]) -> RebalancePolicy:
        """The run's rebalancing policy: ``rebalancing_policy``, with the deprecated ``tranches`` keys folded in.

        One tranche concept (TOM-1513): ``risk.tranches`` and the rebalancers'
        ``tranches`` are the policy's tranche cadence
        (:func:`quantbox.engine.policy.tranches_alias`, which the backtest calls
        too); declaring tranches twice is refused. With a declared policy its
        ``min_trade`` is the trade floor, so ``min_trade_size`` beside it is
        refused: one floor, stated once.
        """
        declared = params.get("rebalancing_policy")
        policy = resolve_policy(declared) if declared is not None else RebalancePolicy(declared=False)
        rebal_cfg_params = dict(rebalancer_cfg.get("params") or {})
        risk_n = int((params.get("risk") or {}).get("tranches", 1) or 1)
        rebal_n = int(rebal_cfg_params.get("tranches", 1) or 1)
        if risk_n > 1 and rebal_n > 1:
            raise ValueError(
                f"risk.tranches ({risk_n}) and rebalancing.params.tranches ({rebal_n}) both declare tranches; "
                "declare rebalancing_policy {cadence: tranche, tranches: N} only (both keys are deprecated, TOM-1518)"
            )
        policy = tranches_alias(policy, risk_n, key="risk.tranches")
        policy = tranches_alias(policy, rebal_n, key="rebalancing.params.tranches")
        if policy.declared:
            both = [
                where
                for where, present in (
                    ("pipeline params", "min_trade_size" in params),
                    ("rebalancing.params", "min_trade_size" in rebal_cfg_params),
                )
                if present
            ]
            if both:
                raise ValueError(
                    f"min_trade_size ({', '.join(both)}) and rebalancing_policy both declare a trade floor; "
                    "declare rebalancing_policy.min_trade only (the backtest's floor, TOM-1518)"
                )
        return policy

    @staticmethod
    def _risk_rules(params: dict[str, Any]) -> dict[str, Any]:
        """The decision's short clip and gross cap from the pipeline's own ``risk`` block (no shorts by default;
        the gross cap's default is :func:`quantbox.decision.gross_cap`'s, the backtest's too)."""
        risk_cfg = params.get("risk") or {}
        return {
            "allow_short": bool(risk_cfg.get("allow_short", False)),
            "max_leverage": gross_cap(risk_cfg),
        }

    def _decision_rules(
        self,
        params: dict[str, Any],
        rebalancer: RebalancingPlugin | None,
        rebal_params: dict[str, Any],
        leverage: str,
        universe: pd.DataFrame,
    ) -> DecisionRules:
        """The run's decision rules (:class:`quantbox.decision.DecisionRules`, TOM-1520).

        The short clip and the gross cap are the ORDER GENERATOR's: an injected
        rebalancer's own (``risk_rules``, the caps it re-applies when it sizes,
        so re-applying them changes nothing), else the pipeline's ``risk``
        block. A third-party rebalancer that declares no ``risk_rules`` keeps
        its short side (no clip here), and gets the gross cap every other door
        has: :func:`quantbox.decision.gross_cap` of its params, default 1
        (TOM-1526; before, it got no cap). ``group_limits`` and
        ``venue.leverage`` are the pipeline's, as in the backtest.
        """
        spec = params.get("group_limits")
        groups: GroupLimits | None = resolve_group_limits(spec).bind(universe) if spec is not None else None
        if rebalancer is None:
            caps = self._risk_rules(params)
        else:
            own = getattr(rebalancer, "risk_rules", None)
            caps = (
                own(rebal_params) if callable(own) else {"allow_short": True, "max_leverage": gross_cap(rebal_params)}
            )
        return DecisionRules(
            allow_short=bool(caps["allow_short"]),
            max_leverage=caps["max_leverage"],
            groups=groups,
            leverage=leverage,
        )

    @staticmethod
    def _final_row(weights: dict[str, float], rules: DecisionRules) -> tuple[dict[str, float], dict[str, Any]]:
        """The decision on ONE decided row (the last), as ``{symbol: weight}`` without zeros, largest first."""
        if not weights:
            return {}, final_targets(pd.DataFrame(), rules)[1]
        row = pd.DataFrame([{str(k): float(v) for k, v in weights.items()}])
        final, report = final_targets(row, rules)
        log_normalisation(report)
        return _target_dict({str(k): float(v) for k, v in final.iloc[0].items()}), report

    @staticmethod
    def _held_weights(rebalancing_df: Any, capital_at_risk: float) -> pd.Series:
        """The broker's held book as target-space weights: ``Current Weight`` / ``capital_at_risk``.

        Read from the order generator's own valuation (the venue's basis), so
        the policy compares the held book to the targets exactly as the orders
        will size them. Cash and the excluded assets are the remainder.

        A held name with no mark has an UNKNOWN weight (NaN in ``Current
        Weight``). It used to be read as 0, so on a margin-basis book an
        unpriced name reached the trigger and ``min_trade`` as flat. It is now a
        valuation refusal (:class:`PortfolioValuationError`): the policy cannot
        judge a book it cannot see, and the caller sizes nothing this cycle
        (TOM-1455).
        """
        if not isinstance(rebalancing_df, pd.DataFrame) or rebalancing_df.empty:
            return pd.Series(dtype=float)
        if not {"Asset", "Current Weight"} <= set(rebalancing_df.columns):
            raise ValueError("rebalancing_policy: the order generator returned no Asset / Current Weight columns")
        weights = pd.to_numeric(rebalancing_df["Current Weight"], errors="coerce")
        unknown = sorted(rebalancing_df.loc[weights.isna().to_numpy(), "Asset"].astype(str))
        if unknown:
            raise PortfolioValuationError(
                f"rebalancing_policy: the held weight of {', '.join(unknown)} is UNKNOWN (no mark), "
                "so the trigger and min_trade cannot be judged. Sizing nothing this cycle.",
                reason=REASON_UNPRICED,
                unpriced=unknown,
            )
        held = pd.Series(weights.to_numpy(dtype=float), index=rebalancing_df["Asset"].astype(str))
        held = held.groupby(level=0).sum()
        return held / capital_at_risk if capital_at_risk > 0 else held

    @staticmethod
    def _hold_untraded(orders_df: pd.DataFrame, trades: tuple[str, ...]) -> pd.DataFrame:
        """Orders on an instrument the policy does not trade on this bar are never executable."""
        if orders_df is None or orders_df.empty or "Asset" not in orders_df.columns:
            return orders_df
        out = orders_df.copy()
        held = ~out["Asset"].astype(str).isin(set(trades))
        if held.any():
            out.loc[held, "Order Status"] = "Held by policy"
            out.loc[held, "Reason"] = "rebalancing_policy: not traded on this bar (a trade under min_trade)"
            out.loc[held, "Adjusted Quantity"] = 0.0
            out["Executable"] = out.get("Executable", pd.Series(True, index=out.index)).astype(bool) & ~held
        return out

    def _policy_orders(
        self,
        *,
        policy: RebalancePolicy,
        history: pd.DataFrame,
        generate: Callable[[dict[str, float]], dict[str, Any]],
        capital_at_risk: float,
        next_bar: pd.Timestamp | None,
        leverage: str,
    ) -> tuple[LiveDecision, dict[str, Any]]:
        """The policy's decision for this run, then the orders that carry it out.

        1. When the trigger or ``min_trade`` reads the held book, or nothing is
           placed, the order generator values the book first on the last decided
           row (*generate*; nothing is sent): its ``Current Weight`` column is
           the held book (:meth:`_held_weights`).
        2. :func:`quantbox.engine.policy.decide_rebalance` — the backtest's own
           cadence, trigger and ``min_trade``.
        3. Placed: *generate* on the decided targets (a traded cell at its
           target, every other cell at its held weight); an order on an
           untraded cell is held (:meth:`_hold_untraded`). Not placed: no
           orders, and the targets are the held book.

        A :class:`PortfolioValuationError` propagates: the caller sizes nothing.
        """
        last_row = {str(k): float(v) for k, v in history.sort_index().ffill().fillna(0.0).iloc[-1].items()}
        needs_held = policy.trigger != "none" or policy.min_trade > 0
        valued: dict[str, Any] | None = generate(last_row) if needs_held else None
        held = self._held_weights(valued.get("rebalancing"), capital_at_risk) if valued is not None else None
        decision = decide_rebalance(policy, history, held, next_bar=next_bar, leverage=leverage)
        if not decision.placed:
            if valued is None:
                valued = generate(last_row)  # values the book for the run record; nothing is sent
                held = self._held_weights(valued.get("rebalancing"), capital_at_risk)
            logger.info("REBALANCING POLICY: no rebalance this run (%s)", decision.reason)
            return decision, {
                **valued,
                "orders": pd.DataFrame(columns=list(ORDER_COLUMNS)),
                "weights": {str(k): float(v) for k, v in held.items() if v != 0},
            }
        assert decision.targets is not None
        targets = {str(k): float(v) for k, v in decision.targets.items() if v != 0}
        result = generate(targets)
        result["orders"] = self._hold_untraded(result.get("orders"), decision.trades)
        logger.info(
            "REBALANCING POLICY: rebalance placed on %d instrument(s) (%s)", len(decision.trades), decision.reason
        )
        return decision, result

    # ==================================================================
    # Stage 5: Order generation
    # ==================================================================
    def _rebalancer_params(
        self,
        *,
        rebalancer_cfg: dict[str, Any],
        params: dict[str, Any],
        strategy_results: Any,
        mode: Mode,
        policy: RebalancePolicy | None = None,
    ) -> dict[str, Any]:
        """Build the params handed to an INJECTED rebalancer plugin.

        Extracted from ``run()`` so the one rule here that is not a default can
        be tested: ``mode`` is ASSIGNED, never ``setdefault``-ed. The rebalancer
        owns the pre-trade valuation, and it decides from ``mode`` whether an
        unmarkable book is fatal -- so a config that says ``mode: paper`` under a
        live run would ungate a live book. That is the incident's own shape: a
        declaration honoured from the wrong source. The real run mode wins.

        Everything else stays ``setdefault``: those are defaults a config is
        entitled to override.

        The rebalancing policy (TOM-1518) owns the tranches and, when declared,
        the trade floor: ``tranches`` is removed (:meth:`_live_policy` folded it
        into the policy) and a declared policy sets ``min_trade_size`` to 0, so
        the rebalancer turns the policy's targets into orders and nothing else.
        """
        rebal_params = dict(rebalancer_cfg.get("params", {}))
        rebal_params.pop("tranches", None)
        if policy is not None and policy.declared:
            rebal_params["min_trade_size"] = 0.0
        rebal_params["strategy_results"] = strategy_results
        rebal_params.setdefault("capital_at_risk", params.get("capital_at_risk", DEFAULT_CAPITAL_AT_RISK))
        rebal_params.setdefault("stable_coin_symbol", params.get("stable_coin_symbol", DEFAULT_STABLE_COIN))
        rebal_params.setdefault("exclusions", params.get("exclusions", []))
        # Threaded like every other pipeline-level default. Both keys are
        # declared in this pipeline's OWN config schema, and both are read by
        # the rebalancers off `params` -- but an injected rebalancer (the
        # production path: `crypto_trend_kraken.yaml` names one) never saw the
        # pipeline's copy, so a tolerance declared where the schema documents it
        # was inert and the built-in default silently won.
        rebal_params.setdefault(
            "equity_reconciliation_tolerance",
            params.get("equity_reconciliation_tolerance", DEFAULT_RECONCILIATION_TOLERANCE),
        )
        rebal_params.setdefault(
            "require_equity_reconciliation",
            params.get("require_equity_reconciliation", True),
        )
        rebal_params["mode"] = mode
        return rebal_params

    # ==================================================================
    # Stage 7: Execution -- quantbox.trade.executor (TOM-1458)
    # ==================================================================
    def _execute_orders(self, *, broker: BrokerPlugin, orders_df: pd.DataFrame, trading_enabled: bool, **kwargs):
        """The old private name, kept ONLY for quantbox-live's ``scripts/shadow_run.py`` execution replay.

        That script calls ``pipeline._execute_orders(broker=, orders_df=, stable_coin=,
        trading_enabled=, mode=)`` at both refs of a shadow run. The execution is
        :func:`quantbox.trade.executor.execute_orders`; ``stable_coin`` and
        ``mode`` were never read. Delete this once the script calls the module.
        """
        kwargs.pop("stable_coin", None)
        kwargs.pop("mode", None)
        return execute_orders(
            broker=broker, orders_df=orders_df, trading_enabled=trading_enabled, alert=self._alert, **kwargs
        )

    # ==================================================================
    # Artifact payload builder (for publisher plugins)
    # ==================================================================
    def _build_artifact_payload(
        self,
        rebalancing_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        execution_report: dict[str, Any],
        final_weights: dict[str, float],
        total_value: float,
        mode: str,
        funding_charge: float | None = None,
        cumulative_fees: float | None = None,
        broker_costs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build structured artifact payload for publishers.

        Ported from quantlab ``trading.py:_build_artifact_payload()``.
        """
        paper_trading = mode == "paper"

        # Rebalancing table
        rebalancing_table: list[dict[str, Any]] = []
        if rebalancing_df is not None and not rebalancing_df.empty:
            col_map = {
                "Asset": "asset",
                "Current Quantity": "current_qty",
                "Current Value": "current_value",
                "Current Weight": "current_weight",
                "Target Quantity": "target_qty",
                "Target Value": "target_value",
                "Target Weight": "target_weight",
                "Weight Delta": "weight_delta",
                "Delta Quantity": "delta_qty",
                "Price": "price",
                "Trade Action": "trade_action",
            }
            for _, row in rebalancing_df.iterrows():
                entry: dict[str, Any] = {}
                for src, dst in col_map.items():
                    val = row.get(src)
                    if val is not None and isinstance(val, (int, float, np.floating)):
                        entry[dst] = round(float(val), 6)
                    elif val is not None:
                        entry[dst] = str(val)
                rebalancing_table.append(entry)

        # Execution summary
        summary = execution_report.get("summary", {})
        executed_orders: list[dict[str, Any]] = []
        failed_orders: list[dict[str, Any]] = []
        # #92: None once ANY executed order's fee is unknown. A partial sum
        # presented as the run's cost is a fabricated number — the same defect
        # as the fabricated zero, wearing a more plausible value.
        total_order_fees: float | None = 0.0
        for detail in execution_report.get("orders_details", []):
            detail_status = detail.get("status")
            if detail_status in ("FILLED", "PARTIAL"):
                order_fee = to_fee(detail.get("fee"))
                if order_fee is None or total_order_fees is None:
                    total_order_fees = None
                else:
                    total_order_fees += order_fee
                executed_orders.append(
                    {
                        "symbol": str(detail.get("symbol", "")),
                        "action": str(detail.get("action", "")),
                        "quantity": detail.get("executed_quantity"),
                        "executed_price": detail.get("executed_price"),
                        "reference_price": detail.get("reference_price"),
                        "spread_bps": round(float(detail.get("spread_pct", 0) or 0) * 10000, 1),
                        "fee": (round(order_fee, 4) if order_fee is not None else None),
                        "status": detail_status,
                    }
                )
            elif detail_status == "FAILED":
                failed_orders.append(
                    {
                        "symbol": str(detail.get("symbol", "")),
                        "action": str(detail.get("action", "")),
                        "error": str(detail.get("error", "")),
                        "status": "FAILED",
                    }
                )

        return {
            "portfolio_value": round(float(total_value), 2),
            "paper_trading": paper_trading,
            "strategy_weights": {k: round(v, 6) for k, v in final_weights.items()},
            "rebalancing_table": rebalancing_table,
            "execution_summary": {
                "total_executed": summary.get("total_executed", 0),
                "total_failed": summary.get("total_failed", 0),
                "total_value_traded": round(float(summary.get("total_value", 0)), 2),
                "total_cost": round(float(summary.get("total_cost", 0)), 4),
                "executed_orders": executed_orders,
                "failed_orders": failed_orders,
            },
            "trading_costs": {
                "fees_this_run": (round(total_order_fees, 4) if total_order_fees is not None else None),
                # None (not 0.0) when the cost could not be measured — see #92.
                "cumulative_fees": (round(cumulative_fees, 4) if cumulative_fees is not None else None),
                "funding_charge": (round(funding_charge, 4) if funding_charge is not None else None),
                **(broker_costs or {}),
            },
        }

    # ==================================================================
    # Token policy helpers
    # ==================================================================
    def _load_token_policy(self, universe_params: dict[str, Any], book_key: str = "default"):
        """Create TokenPolicy from universe params or config file.

        ``book_key`` namespaces the seen-token store per book (issue #86). The
        inline (``from_dict``) path has no YAML anchor, so passing ``book_key``
        here is what keeps its seen-token set per-book instead of a shared
        cwd-relative default. Returns ``None`` when no token-policy configuration
        is present (backward-compatible).
        """
        from quantbox.plugins.trading.token_policy import TokenPolicy

        tp_cfg = universe_params.get("token_policy")
        tp_file = universe_params.get("token_policy_file")

        if tp_cfg:
            return TokenPolicy.from_dict({"token_policy": tp_cfg}, book_key=book_key)
        elif tp_file:
            return TokenPolicy.from_config(tp_file, book_key=book_key)
        return None

    def _detect_new_tokens(self, policy, universe: pd.DataFrame) -> list:
        """Build a minimal rankings DF from universe and run detection."""
        if not policy.alert_on_new or universe.empty:
            return []

        symbols = universe["symbol"].tolist()
        rankings_data = [{"symbol": sym, "rank": rank, "market_cap": 0} for rank, sym in enumerate(symbols, 1)]
        rankings_df = pd.DataFrame(rankings_data)
        return policy.detect_new_tokens(rankings_df, policy.top_n_monitor)
