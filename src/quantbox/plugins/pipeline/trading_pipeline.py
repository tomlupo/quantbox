"""Full trading pipeline plugin.

Orchestrates: strategy execution -> aggregation -> the decision (final
target weights: short clip, gross cap, group limits, ``venue.leverage``
normalisation; :mod:`quantbox.decision`, the backtest's own transform,
TOM-1520) -> the rebalancing policy -> order generation -> execution ->
artifact storage.

Ported from the quantlab ``trading.py`` workflow, ``orders.py``, and
``portfolio.py`` into a single PipelinePlugin that the quantbox runner
can invoke via ``pipeline.run()``.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal, getcontext
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from quantbox.contracts import (
    ArtifactStore,
    BrokerPlugin,
    DataPlugin,
    Mode,
    PluginMeta,
    RebalancingPlugin,
    RiskPlugin,
    RunResult,
    StrategyPlugin,
)
from quantbox.decision import DecisionRules, final_targets, gross_cap, log_normalisation, risk_caps_row
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
from quantbox.plugins.rebalancing.futures_rebalancer import RESIDUAL_STATUS
from quantbox.strategy_runner import build_strategy_context, decided_weights, latest_weights, run_strategies
from quantbox.trade.book import (
    BASIS_MARK,
    DEFAULT_RECONCILIATION_TOLERANCE,
    REASON_UNPRICED,
    MarkedBook,
    PortfolioValuation,
    PortfolioValuationError,
    nan_if_unknown,
    read_book,
)
from quantbox.trade.notifier import Alerts, Notifier, escape_html
from quantbox.trade.reconciliation.cycle import filter_reduce_only, open_cycle
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
# Constants (from quantlab orders.py)
# ---------------------------------------------------------------------------
DEFAULT_CAPITAL_AT_RISK = 1.0
DEFAULT_MIN_NOTIONAL = 1.0
DEFAULT_MIN_TRADE_SIZE = 0.01
DEFAULT_STABLE_COIN = "USDC"

#: The universe screen's inputs, stored next to ``prices.parquet`` in every run
#: dir (TOM-1420), so a run's selection can be replayed config-only (TOM-1722).
SCREEN_INPUTS = ("volume", "market_cap", "screen_volume")

# ---------------------------------------------------------------------------
# Low-level helpers (ported from quantlab orders.py / portfolio.py)
# ---------------------------------------------------------------------------


def _safe_float(v: Any) -> float | None:
    """Best-effort float coercion for ledger records; None on failure/None."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# Ledger result status mapped from an execution ``orders_details`` row status.
# Backwards-compatible alias. The map itself now lives in
# ``reconciliation.ledger`` beside the vocabulary it targets, because the
# out-of-cycle working-order resolver must translate into the same map and a
# private second copy is how two callers start disagreeing.
_EXEC_STATUS_TO_LEDGER = EXEC_STATUS_TO_LEDGER

# The statuses a broker row may carry that the execution loop acts on. Anything
# else — absent, empty, or a word outside this set — is UNKNOWN (TOM-1336).
_EXEC_STATUSES = frozenset({"FILLED", "PARTIAL", "WORKING", "FAILED", "SKIPPED", "UNKNOWN"})


def _adjust_quantity(qty: float, step_size: float) -> float:
    """Round *qty* down to the nearest *step_size* (Binance-style)."""
    if step_size <= 0:
        return qty
    step_str = f"{step_size:.8f}"
    decimal_places = step_str.rstrip("0").split(".")[-1]
    precision = len(decimal_places)
    getcontext().rounding = ROUND_DOWN
    return float(Decimal(qty).quantize(Decimal("1." + "0" * precision)))


def _suppressed_order_records(suppressed: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-order detail for a freeze, as structured records.

    A freeze alert used to carry only a status HISTOGRAM ("Below min notional=2,
    Below threshold=1"). That says how many orders died and nothing about which,
    how big, or how far under the bar they were — so the first question a human
    asks on being paged ("what could not trade, and by how much?") required
    opening the run log every time. Everything needed is already on the row; it
    just never left the pipeline.

    ``shortfall_usd`` is what makes it actionable: it is the gap between the
    order and the venue minimum, i.e. exactly how much the book would have to
    grow for this leg to become executable.
    """
    records: list[dict[str, Any]] = []
    for _, row in suppressed.iterrows():
        notional = float(row.get("Notional Value", 0) or 0)
        min_notional = float(row.get("Min Notional", 0) or 0)
        record = {
            "symbol": str(row.get("Asset", "") or row.get("Symbol", "")),
            "action": str(row.get("Action", "")).upper(),
            "notional_usd": notional,
            "min_notional_usd": min_notional,
            "status": str(row.get("Order Status", "")),
            "reason": str(row.get("Reason", "")),
        }
        if min_notional > 0 and 0 < notional < min_notional:
            record["shortfall_usd"] = min_notional - notional
        records.append(record)
    return records


def _format_suppressed_line(rec: dict[str, Any]) -> str:
    """One human-readable line per suppressed order, for chat alerts."""
    line = f"{rec['action']} {rec['symbol']} ${rec['notional_usd']:,.2f}"
    shortfall = rec.get("shortfall_usd")
    if shortfall:
        line += f" (min ${rec['min_notional_usd']:,.2f}, short ${shortfall:,.2f})"
    detail = rec.get("reason") or rec.get("status")
    if detail:
        line += f" — {detail}"
    return line


def _get_lot_size_and_min_notional(
    symbol_info: dict | None,
) -> tuple[float, float, float]:
    """Extract (min_qty, step_size, min_notional) from Binance symbol info."""
    min_qty, step_size, min_notional = 0.0, 0.0, 0.0
    if not symbol_info:
        return min_qty, step_size, min_notional
    for f in symbol_info.get("filters", []):
        if f.get("filterType") == "LOT_SIZE":
            min_qty = float(f.get("minQty", 0))
            step_size = float(f.get("stepSize", 0))
        if f.get("filterType") == "NOTIONAL":
            min_notional = float(f.get("minNotional", 0))
    return min_qty, step_size, min_notional


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
    if broker is None or mode not in ("paper", "live") or not hasattr(broker, "get_positions"):
        return universe
    try:
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
    """
    if broker is None:
        return None

    if hasattr(broker, "apply_funding"):
        charge = broker.apply_funding()
        logger.info("Applied funding charge: %.2f", charge or 0.0)
        return charge

    if hasattr(broker, "fetch_funding_payments"):
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
        "Broker %s exposes no funding source (neither apply_funding nor "
        "fetch_funding_payments) — funding reported UNKNOWN, not $0.00",
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


def _to_fee(value: Any) -> float | None:
    """A fee that was not reported is UNKNOWN, never 0.0 (#92).

    The `float(x or 0)` idiom this replaces mapped both "no fee key" and
    "fee was genuinely zero" onto 0.0. Only the second is data. Conflating them
    is what produced 165 days of `Fees (cumulative) $0.00` on a book with ~24x
    equity turnover.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN is not a measurement


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
                        "plugin is injected (an injected rebalancer's own params are the rules then); also "
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
                        "Min abs(weight delta) to consider a trade (the order-level floor). With a declared "
                        "rebalancing_policy its min_trade is the floor instead and this key is refused."
                    ),
                },
                "min_notional": {
                    "type": "number",
                    "minimum": 0,
                    "default": 1.0,
                    "description": "Orders below this notional (quote currency) are dropped.",
                },
                "scaling_factor_min": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.9,
                    "description": "Lowest scale-down applied to buys when cash is short before orders are dropped.",
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

        # --- Inject latest prices into paper broker (if supported) ---
        if broker is not None and hasattr(broker, "set_prices"):
            wide_prices = market_data.get("prices", pd.DataFrame())
            if not wide_prices.empty:
                latest = wide_prices.iloc[-1].dropna()
                broker.set_prices({str(k): float(v) for k, v in latest.items()})
                logger.info("Injected %d prices into broker", len(latest))

        # --- Inject funding rates into broker (if supported) ---
        wide_funding = market_data.get("funding_rates", pd.DataFrame())
        if broker is not None and not wide_funding.empty and hasattr(broker, "set_funding_rates"):
            latest_funding = wide_funding.iloc[-1].dropna()
            broker.set_funding_rates({str(k): float(v) for k, v in latest_funding.items()})
            logger.info("Injected %d funding rates into broker", len(latest_funding))

        # --- Inject position limits into broker (if supported) ---
        if broker is not None and hasattr(broker, "set_position_limits") and hasattr(data, "_fetcher"):
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
            elif live_book:
                internal_params = {**params, "min_trade_size": 0.0} if policy.declared else params

                def generate(weights: dict[str, float]) -> dict[str, Any]:
                    out = self._generate_orders(
                        broker=broker,
                        weights=self._apply_risk_transforms(weights, params),
                        capital_at_risk=float(params.get("capital_at_risk", DEFAULT_CAPITAL_AT_RISK)),
                        stable_coin=str(params.get("stable_coin_symbol", DEFAULT_STABLE_COIN)),
                        params=internal_params,
                        mode=mode,
                    )
                    out["weights"] = self._apply_risk_transforms(weights, params)
                    return out

                book_capital = float(params.get("capital_at_risk", DEFAULT_CAPITAL_AT_RISK))
            if live_book:
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
        capital_at_risk = float(params.get("capital_at_risk", DEFAULT_CAPITAL_AT_RISK))

        if order_result is not None:
            # Already ran the rebalancer (or the policy's order generation) above; reuse result
            pass
        else:
            try:
                order_result = self._generate_orders(
                    broker=broker,
                    weights=final_weights,
                    capital_at_risk=capital_at_risk,
                    stable_coin=stable_coin,
                    params=params,
                    mode=mode,
                )
            except PortfolioValuationError as exc:  # see the rebalancer arm above
                valuation_refusal = exc
                api_errors.append({"stage": "portfolio_valuation", "error": str(exc)})
                logger.error(
                    "Portfolio valuation REFUSED (%s) -- sizing nothing this cycle; bookkeeping stages still run",
                    exc,
                )
                order_result = _empty_order_result()
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

        # --- Stage 7: Execution ---
        trading_enabled = bool(params.get("trading_enabled", True))
        execution_report = self._execute_orders(
            broker=broker,
            orders_df=orders_df,
            stable_coin=stable_coin,
            trading_enabled=trading_enabled,
            mode=mode,
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
            if hasattr(broker, "get_equity"):
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
        # not free, so it is None here and null downstream.
        cumulative_fees: float | None = None
        if broker is not None and hasattr(broker, "_cumulative_fees"):
            cumulative_fees = float(broker._cumulative_fees)

        # --- Stage 8: Artifacts ---
        # Collect broker cost config for artifact
        broker_costs: dict[str, Any] = {}
        if broker is not None:
            for attr in ("spread_bps", "slippage_bps", "maker_fee_bps", "taker_fee_bps"):
                if hasattr(broker, attr):
                    broker_costs[attr] = float(getattr(broker, attr))

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
                # the notifier must alert on, from every exit of _execute_orders.
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
    # Stage 4: Risk transforms
    # ==================================================================
    def _apply_risk_transforms(
        self,
        weights: dict[str, float],
        params: dict[str, Any],
    ) -> dict[str, float]:
        """The short clip, then the gross cap (:func:`quantbox.decision.risk_caps_row`), on the targets the
        internal order generator sizes; zeros dropped, largest first.

        The decision (:func:`quantbox.decision.final_targets`) already applied
        both, in the same order, so this re-application changes nothing. Tranching
        is the rebalancing policy's tranche cadence (:meth:`_live_policy`, TOM-1518).
        """
        return _target_dict(risk_caps_row(weights, **self._risk_rules(params)))

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

    def _generate_orders(
        self,
        broker: BrokerPlugin,
        weights: dict[str, float],
        capital_at_risk: float,
        stable_coin: str,
        params: dict[str, Any],
        mode: Mode,
    ) -> dict[str, Any]:
        """Generate rebalancing DataFrame and executable orders.

        Ported from quantlab ``orders.py:generate_portfolio_orders()``.

        ``mode`` drives the pre-trade valuation gate: in ``live`` a book that
        cannot be fully marked refuses instead of sizing off an understated
        portfolio value. It is REQUIRED rather than defaulting to ``"paper"``:
        a default would let a future caller ungate a live book by forgetting an
        argument, and forgetting is not a declaration that this is a simulation.
        """
        min_notional_cfg = float(params.get("min_notional", DEFAULT_MIN_NOTIONAL))
        min_trade_size = float(params.get("min_trade_size", DEFAULT_MIN_TRADE_SIZE))
        scaling_factor_min = float(params.get("scaling_factor_min", 0.9))
        exclusions = list(params.get("exclusions", [])) + [stable_coin]
        exclusions = list(set(exclusions))

        # The marked book: positions, quote cash and one snapshot of marks for
        # the targets and the held names (quantbox.trade.book).
        book = read_book(broker, wanted=weights.keys(), exclude=exclusions, cash=(stable_coin, "USD"))
        cash_available = float(book.cash or 0.0)

        # Portfolio value. `broker.get_equity()` is preferred where the broker
        # has one (for derivatives, cash + sum(qty * price) is wrong for
        # shorts); otherwise the book is marked. Either way an unmarkable
        # holding is a REFUSAL in live mode, never a silently smaller number --
        # see quantbox.trade.book.
        valuation = book.valuation(
            broker=broker,
            mode=mode,
            # This path marks a spot/cash book, so an undeclared broker in a
            # SIMULATION keeps being valued the way it always was here. On live
            # an undeclared broker refuses.
            fallback_basis=BASIS_MARK,
            stable_coin=stable_coin,
            exclusions=exclusions,
            tolerance=float(params.get("equity_reconciliation_tolerance", DEFAULT_RECONCILIATION_TOLERANCE)),
            require_reconciliation=bool(params.get("require_equity_reconciliation", True)),
        )
        total_value = valuation.value

        if total_value <= 0:
            logger.error(
                "Portfolio value is zero or negative (cash=%.2f, %d holding(s), state=%s)",
                valuation.cash,
                valuation.n_holdings,
                valuation.state,
            )
            return {
                "orders": pd.DataFrame(),
                "rebalancing": pd.DataFrame(),
                "total_value": 0.0,
                "valuation": valuation,
            }

        # Apply capital-at-risk
        adjusted_weights = {a: w * capital_at_risk for a, w in weights.items()}

        # Target positions
        target_positions: dict[str, float] = {}
        for asset, weight in adjusted_weights.items():
            p = book.mark(asset)
            if p is not None:
                target_positions[asset] = (total_value * weight) / p

        # Rebalancing DataFrame
        rebalancing_df = self._build_rebalancing(
            book=book,
            target_positions=target_positions,
            total_value=total_value,
            strategy_weights=adjusted_weights,
            exclusions=exclusions,
        )

        # Executable orders
        orders_df = self._create_executable_orders(
            rebalancing_df=rebalancing_df,
            broker=broker,
            stable_coin=stable_coin,
            min_notional_default=min_notional_cfg,
            min_trade_size=min_trade_size,
            cash_available=cash_available,
            scaling_factor_min=scaling_factor_min,
        )

        return {
            "orders": orders_df,
            "rebalancing": rebalancing_df,
            "total_value": total_value,
            "valuation": valuation,
        }

    # ------------------------------------------------------------------

    def _build_rebalancing(
        self,
        book: MarkedBook,
        target_positions: dict[str, float],
        total_value: float,
        strategy_weights: dict[str, float],
        exclusions: list[str],
    ) -> pd.DataFrame:
        """Build the rebalancing analysis DataFrame.

        Ported from quantlab ``orders.py:generate_rebalancing_dataframe()``.
        Values come from the marked book. A held name with no mark has an
        UNKNOWN current value and weight (NaN), never 0 (TOM-1455).
        """
        current_holdings = book.held
        assets = sorted(set(current_holdings.keys()) | set(target_positions.keys()))
        assets = [a for a in assets if a not in exclusions]

        rows: list[dict[str, Any]] = []
        for asset in assets:
            current_qty = current_holdings.get(asset, 0.0)
            price = book.mark(asset)
            current_value = nan_if_unknown(book.value_of(asset))
            target_qty = target_positions.get(asset, 0.0)
            target_value = nan_if_unknown(book.value_of(asset, target_qty))
            current_weight = current_value / total_value if total_value > 0 else 0
            target_weight = target_value / total_value if total_value > 0 else 0
            weight_delta = strategy_weights.get(asset, 0) - current_weight
            delta_qty = target_qty - current_qty

            if delta_qty > 0:
                trade_action = "Buy"
            elif delta_qty < 0:
                trade_action = "Sell"
            else:
                trade_action = "Hold"

            rows.append(
                {
                    "Asset": asset,
                    "Current Quantity": current_qty,
                    "Current Value": current_value,
                    "Current Weight": current_weight,
                    "Target Quantity": target_qty,
                    "Target Value": target_value,
                    "Target Weight": target_weight,
                    "Weight Delta": weight_delta,
                    "Delta Quantity": delta_qty,
                    "Price": price,
                    "Trade Action": trade_action,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------

    def _create_executable_orders(
        self,
        rebalancing_df: pd.DataFrame,
        broker: BrokerPlugin,
        stable_coin: str,
        min_notional_default: float,
        min_trade_size: float,
        cash_available: float,
        scaling_factor_min: float,
    ) -> pd.DataFrame:
        """Convert rebalancing to executable orders with lot/step/notional validation.

        Ported from quantlab ``orders.py:generate_orders_from_rebalancing()``.
        Uses ``broker.get_market_snapshot()`` for symbol info where available.
        """
        if rebalancing_df.empty:
            return pd.DataFrame(columns=list(ORDER_COLUMNS))

        order_records: list[dict[str, Any]] = []

        for _, row in rebalancing_df.iterrows():
            asset = row["Asset"]
            symbol = f"{asset}{stable_coin}"
            action = row["Trade Action"].lower()
            delta_qty = row["Delta Quantity"]
            price = row["Price"]
            status = None
            reason = None
            adjusted_qty = 0.0
            notional_value = 0.0
            min_qty = 0.0
            step_size = 0.0
            min_notional = min_notional_default
            scaling_factor = None

            if action == "hold" or delta_qty == 0:
                status = "Zero delta"
                reason = "No trade needed"
            elif price is None or pd.isna(price) or price == 0:
                # No mark: the held name's value and weight are UNKNOWN (NaN, the
                # marked book), so no threshold, notional or scaling can judge
                # it. Before TOM-1455 its weight read as 0 -- a short slipped into
                # buy scaling with a NaN price, and a long close was "To be
                # placed" at a NaN price.
                status = "Zero price"
                reason = "No price available"
            else:
                zero_target_sell = (
                    action == "sell" and row.get("Target Weight", 0) == 0 and row.get("Current Quantity", 0) > 0
                )
                if abs(row["Weight Delta"]) < min_trade_size and not zero_target_sell:
                    status = "Below threshold"
                    reason = f"abs(weight delta) < {min_trade_size}"
                else:
                    # Try to get symbol info from broker snapshot
                    try:
                        snap = broker.get_market_snapshot([asset])
                        if snap is not None and not snap.empty:
                            info_row = snap.iloc[0]
                            min_qty = float(info_row.get("min_qty", 0) or 0)
                            step_size = float(info_row.get("step_size", 0) or 0)
                            mn = info_row.get("min_notional")
                            if mn is not None and float(mn) > 0:
                                min_notional = float(mn)
                    except Exception:
                        pass

                    if action == "sell":
                        adjusted_qty = _adjust_quantity(abs(delta_qty), step_size) if step_size > 0 else abs(delta_qty)
                        notional_value = adjusted_qty * price if price else 0.0

                        if price is None or price == 0:
                            status = "Zero price"
                            reason = "No price available"
                            adjusted_qty = 0.0
                        elif notional_value < min_notional:
                            if zero_target_sell and min_qty > 0 and (min_qty * price) >= min_notional:
                                # Full close-out: round up to min lot so position actually closes
                                adjusted_qty = min_qty
                                notional_value = min_qty * price
                                status = "To be placed"
                                reason = f"Rounded up to min_qty={min_qty} (full close-out)"
                            else:
                                status = "Below min notional"
                                reason = f"Notional {notional_value:.4f} < min_notional {min_notional:.4f}"
                                adjusted_qty = 0.0
                        elif min_qty > 0 and adjusted_qty < min_qty:
                            if zero_target_sell:
                                adjusted_qty = min_qty
                                notional_value = min_qty * price
                                status = "To be placed"
                                reason = f"Rounded up to min_qty={min_qty} (full close-out)"
                            else:
                                status = "Below min qty"
                                reason = f"Qty {adjusted_qty:.8f} < min_qty {min_qty:.8f}"
                                adjusted_qty = 0.0
                        else:
                            status = "To be placed"
                            reason = ""
                    elif action == "buy":
                        status = "Pending scaling"
                        reason = ""

            order_records.append(
                {
                    "Asset": asset,
                    "Symbol": symbol,
                    "Action": action.capitalize(),
                    "Raw Quantity": abs(delta_qty),
                    "Adjusted Quantity": adjusted_qty,
                    "Notional Value": notional_value,
                    "Price": price,
                    "Min Notional": min_notional,
                    "Min Qty": min_qty,
                    "Step Size": step_size,
                    "Order Status": status,
                    "Reason": reason,
                    "Scaling Factor": scaling_factor,
                }
            )

        # --- Buy scaling (ported from quantlab) ---
        buy_indices = [
            i
            for i, rec in enumerate(order_records)
            if rec["Action"] == "Buy" and rec["Order Status"] == "Pending scaling"
        ]
        total_buy_value = sum(order_records[i]["Raw Quantity"] * (order_records[i]["Price"] or 0) for i in buy_indices)
        cash_from_sells = sum(
            rec["Notional Value"]
            for rec in order_records
            if rec["Action"] == "Sell" and rec["Order Status"] == "To be placed"
        )
        available_cash = cash_available + cash_from_sells
        scaling_factor = min(1.0, available_cash / total_buy_value) if total_buy_value > 0 else 0.0

        if total_buy_value > 0 and scaling_factor >= scaling_factor_min:
            for i in buy_indices:
                rec = order_records[i]
                price = rec["Price"]
                step_size = rec["Step Size"]
                min_qty_val = rec["Min Qty"]
                mn = rec["Min Notional"]
                raw_qty = rec["Raw Quantity"]

                scaled_qty = (
                    _adjust_quantity(raw_qty * scaling_factor, step_size)
                    if step_size and price
                    else raw_qty * scaling_factor
                )
                scaled_notional = scaled_qty * price if price else 0.0
                rec["Scaling Factor"] = scaling_factor
                rec["Adjusted Quantity"] = scaled_qty
                rec["Notional Value"] = scaled_notional

                if price is None or price == 0:
                    rec["Order Status"] = "Zero price"
                    rec["Reason"] = "No price available"
                    rec["Adjusted Quantity"] = 0.0
                elif scaled_notional < mn:
                    rec["Order Status"] = "Below min notional"
                    rec["Reason"] = f"Notional {scaled_notional:.4f} < min_notional {mn:.4f}"
                    rec["Adjusted Quantity"] = 0.0
                elif min_qty_val > 0 and scaled_qty < min_qty_val:
                    rec["Order Status"] = "Below min qty"
                    rec["Reason"] = f"Qty {scaled_qty:.8f} < min_qty {min_qty_val:.8f}"
                    rec["Adjusted Quantity"] = 0.0
                elif scaled_qty == 0:
                    rec["Order Status"] = "Zero quantity"
                    rec["Reason"] = "Scaled quantity is zero"
                else:
                    rec["Order Status"] = "To be placed"
                    rec["Reason"] = ""
        elif total_buy_value > 0:
            logger.warning(
                "Buy scaling factor %.4f < minimum %.2f, skipping buys",
                scaling_factor,
                scaling_factor_min,
            )

        columns = [
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
        ]
        order_df = pd.DataFrame(order_records, columns=columns)
        order_df["Executable"] = (order_df["Adjusted Quantity"] > 0) & (order_df["Order Status"] == "To be placed")
        return order_df

    # ==================================================================
    # Stage 7: Execution
    # ==================================================================
    def _report_order_failures(self, report: dict[str, Any]) -> None:
        """Record + alert this cycle's order failures. Idempotent; no-op at zero.

        The two freeze guards are both CONJUNCTIONS requiring a total standstill:
        one needs ``total_executed == 0`` AND a skipped close-out SELL, the other
        needs the broker to return an EMPTY frame. Neither fires on the commonest
        real failure — "some orders filled, some were rejected" — which is how
        carver-HL emitted the same rejected ETH/SOL close-outs for 33 consecutive
        days while every run was recorded as healthy (quantbox#87).

        Called from BOTH exits of :meth:`_execute_orders`: the normal path and the
        ``place_orders`` exception path. The exception path is the worst case —
        the whole batch failed at submission and nothing traded — so it must alert
        too, rather than returning early past the report.
        """
        n_failed = int(report["summary"]["total_failed"])
        if not n_failed:
            return
        failed_list = ", ".join(
            f"{str(d.get('action', '')).upper()} {d.get('symbol', '')} ({d.get('error', 'unknown')})"
            for d in report["orders_details"]
            if d.get("status") == "FAILED"
        )
        report["order_failures"] = n_failed
        report["order_failure_detail"] = failed_list
        logger.error(
            "ORDER FAILURES: %d of %d submitted order(s) failed — %s",
            n_failed,
            n_failed + int(report["summary"]["total_executed"]),
            failed_list,
        )
        self._alert(
            f"❌ <b>{n_failed} ORDER(S) FAILED</b>\n{escape_html(failed_list)}\n"
            "Targets NOT reached for these symbols; exposure is unchanged.",
            site="order_failures",
        )

    def _execute_orders(
        self,
        broker: BrokerPlugin,
        orders_df: pd.DataFrame,
        stable_coin: str,
        trading_enabled: bool,
        mode: str,
        ledger: Any = None,
        cycle_id: str | None = None,
        gate_orders_allowed: bool = True,
        gate_reduce_only: bool = False,
        capture_fail_closed: bool = False,
        working_store: Any = None,
    ) -> dict[str, Any]:
        """Execute orders via broker with sell-before-buy ordering.

        Ported from quantlab ``orders.py:execute_orders()``.

        Reconciliation (issue #90):

        * If ``ledger`` is supplied, the order INTENT is captured at the broker
          submission call — inside this method, immediately before
          ``broker.place_orders`` — and each order's RESULT is recorded as fills
          are processed. This makes the ledger a crash-durable audit trail: if the
          run dies between submission and Stage 7b, the intent is already on disk.
        * ``gate_orders_allowed`` / ``gate_reduce_only`` are the PRE-EXECUTION gate
          the caller derived from the persisted prior-cycle recon state (enforce
          mode only). ``gate_orders_allowed=False`` halts all orders this cycle;
          ``gate_reduce_only=True`` keeps only exposure-reducing orders. In observe
          mode the caller passes the permissive defaults, so behavior is unchanged.
        * ``working_store`` (optional): queue for orders the venue ACCEPTED that
          were still resting on the book when the confirmation window closed. They
          are recorded here so the next cycle can resolve them against the venue
          and book the real fill; without a store they are still reported, but
          their eventual fill is not recovered.
        * ``capture_fail_closed`` (enforce mode): if intent capture fails for an
          order, that order is DROPPED (not submitted) — under enforcement we must
          never trade live without the crash-durable intent record. In observe mode
          (default) a capture failure is only an observability loss and the order
          still sends.
        """
        # Orders left working at the venue this cycle. Defined before any early
        # return so the persist step at the end is always safe to run.
        working_to_queue: list[dict[str, Any]] = []
        report: dict[str, Any] = {
            "executed_orders": [],
            "failed_orders": [],
            "summary": {
                "total_executed": 0,
                "total_partial": 0,
                "total_failed": 0,
                # Accepted by the venue, still resting on the book when the run
                # ended. Neither executed nor failed — resolved next cycle.
                "total_working": 0,
                "total_value": 0.0,
                "total_cost": 0.0,
            },
            "orders_details": [],
            "api_errors": [],  # broker API errors caught during execution (#62)
        }

        if orders_df is None or orders_df.empty:
            logger.info("No orders generated")
            return report

        if not trading_enabled:
            logger.info("trading_enabled=False, skipping execution")
            return report

        # --- Pre-execution HALT gate (enforce mode, issue #90) -------------
        # Derived by the caller from the persisted prior-cycle recon state. A HALT
        # means the order path / books are not trustworthy: send nothing this cycle.
        if not gate_orders_allowed:
            report["recon_gated"] = "halt"
            logger.error(
                "RECON ENFORCE: %d order(s) HALTED pre-execution by the prior-cycle gate — nothing sent this cycle.",
                len(orders_df),
            )
            self._alert(
                f"🛑 <b>RECON ENFORCE — HALT</b>\n{len(orders_df)} order(s) NOT sent; "
                "book is in HALT from the prior cycle. Manual clear required.",
                site="recon_halt",
            )
            return report

        executable = (
            orders_df[orders_df.get("Executable", pd.Series(dtype=bool))].copy()
            if "Executable" in orders_df.columns
            else orders_df
        )

        # Partial reduces below the venue minimum (TOM-1778): never sent, because
        # the venue rejects them. A known residual — listed, never a freeze.
        residuals = (
            orders_df[orders_df["Order Status"] == RESIDUAL_STATUS]
            if "Order Status" in orders_df.columns
            else pd.DataFrame()
        )
        if not residuals.empty:
            report["residuals"] = _suppressed_order_records(residuals)
            logger.info(
                "RESIDUAL: %d partial reduce(s) below the venue minimum not sent, residual retained: %s",
                len(residuals),
                "; ".join(_format_suppressed_line(r) for r in report["residuals"]),
            )

        # --- Pre-execution FLATTEN gate (enforce mode, issue #90) ----------
        # Keep only exposure-reducing orders (clamped so they never cross zero).
        if gate_reduce_only and isinstance(executable, pd.DataFrame) and not executable.empty:
            before_n = len(executable)
            executable = filter_reduce_only(executable, broker)
            report["recon_gated"] = "flatten"
            logger.warning(
                "RECON ENFORCE: reduce-only gate — %d of %d order(s) retained (exposure-reducing only).",
                len(executable),
                before_n,
            )
            if executable.empty:
                # All orders were opening/increasing — nothing to reduce. This is a
                # clean gated no-op, NOT a freeze.
                report["recon_reduce_only_noop"] = True
                logger.warning("RECON ENFORCE reduce-only: no exposure-reducing orders to send.")
                return report

        if executable.empty:
            reasons: dict[str, Any] = {}
            # Statuses that mean "the strategy WANTED to trade but the order was
            # suppressed by a guard" (vs. a legitimately quiet day of zero-deltas).
            suppressed_statuses = {
                "Below min notional",
                "Below threshold",
                "Below min qty",
                "Zero quantity",
                "Zero price",
                "Invalid (NaN)",
            }
            suppressed = pd.DataFrame()
            if "Order Status" in orders_df.columns:
                reasons = orders_df["Order Status"].value_counts().to_dict()
                suppressed = orders_df[orders_df["Order Status"].isin(suppressed_statuses)]

            # RESIDUAL DAY (TOM-1778; Tom, 2026-10-12: "Residual is not a freeze").
            # When the book's only untraded EXITS are partial reduces below the
            # venue minimum, and everything else suppressed is a sub-min ENTRY (an
            # open from flat or a same-side add, flagged `opens_exposure` by the
            # rebalancer), nothing is trapped: the venue cannot take those trims at
            # this book size. Drift stays recon's job. Anything else suppressed —
            # a close, a flip through zero, a reduce, a data fault (zero price,
            # NaN), or a row without the flag — still goes to the dead-man below.
            if not residuals.empty:
                opens = (
                    suppressed["opens_exposure"].eq(True)
                    if "opens_exposure" in suppressed.columns
                    else pd.Series(False, index=suppressed.index)
                )
                only_entries = bool(
                    (
                        suppressed["Order Status"].isin({"Below min notional", "Below threshold", "Below min qty"})
                        & opens
                    ).all()
                )
                if only_entries:
                    report["residual_day"] = True
                    report["residual_reasons"] = reasons
                    logger.info(
                        "RESIDUAL DAY: no order to send — %d residual partial reduce(s) below the venue "
                        "minimum and %d sub-min entry leg(s). Not a freeze.",
                        len(residuals),
                        len(suppressed),
                    )
                    return report

            if len(suppressed) > 0:
                # Distinguish a TRAPPED book (real freeze) from a QUIET DAY
                # (all-cash, nothing to do). The dead-man freeze exists to catch
                # a book that SHOULD be rebalanced/exited but CANNOT — i.e. a
                # held position whose exit/reduce order is sub-min and therefore
                # suppressed. That manifests as a suppressed SELL leg, and/or the
                # broker still reporting liquidatable positions (post stablecoin-
                # dust exclusion, see kraken.get_positions).
                #
                # By contrast, on a weak-signal day an all-cash book produces a
                # ~few-percent gross target where every ENTRY (buy) leg is below
                # min_notional. Nothing is stuck — there are no positions to exit
                # and no suppressed sells. Staying in cash is the correct, clean
                # outcome, not a frozen rebalancer. Classify it as a QUIET DAY at
                # INFO level so flat-trend days don't fire the loud freeze alert.
                actions = suppressed.get("Action", pd.Series(dtype=object)).astype(str).str.lower()
                has_suppressed_sell = bool((actions == "sell").any())

                has_liquidatable_positions = False
                cannot_confirm_flat = False
                get_positions = getattr(broker, "get_positions", None)
                if callable(get_positions):
                    try:
                        pos = get_positions()
                        has_liquidatable_positions = pos is not None and len(pos) > 0
                    except Exception as exc:  # never let a position probe crash the run
                        # Fail SAFE: if we cannot confirm the book is flat, treat
                        # it as potentially trapped (do not downgrade to quiet).
                        logger.warning("Position probe for quiet/freeze classification failed: %s", exc)
                        cannot_confirm_flat = True
                else:
                    # Fail SAFE (issue #82): a broker with no position probe cannot
                    # confirm the book is flat. On a long-only spot book a suppressed
                    # BUY is always a (harmless) entry, but on a futures book a BUY can
                    # be a short-close EXIT that the has_suppressed_sell heuristic does
                    # NOT catch. Without get_positions we cannot tell quiet from
                    # trapped, so we must NOT silently downgrade to quiet — classify as
                    # potentially trapped and alert. (Both live books — Kraken spot and
                    # Hyperliquid perps — always expose get_positions, so this never
                    # false-fires on a healthy live run.)
                    logger.warning(
                        "Broker %s exposes no get_positions; cannot confirm a flat book "
                        "for quiet/freeze classification — failing safe to trapped.",
                        type(broker).__name__,
                    )
                    cannot_confirm_flat = True

                trapped = has_suppressed_sell or has_liquidatable_positions or cannot_confirm_flat

                if not trapped:
                    # QUIET DAY: all-cash, every target leg sub-min. Stay in cash.
                    report["quiet_day"] = True
                    report["quiet_reasons"] = reasons
                    order_list = ", ".join(
                        f"{str(r.get('Action', '')).upper()} {r.get('Asset', '')} "
                        f"(${float(r.get('Notional Value', 0) or 0):.2f}, {r.get('Order Status', '')})"
                        for _, r in suppressed.iterrows()
                    )
                    logger.info(
                        "QUIET DAY: all-cash, %d sub-min target leg(s) — staying in cash. "
                        "No liquidatable positions and no suppressed exits, so the book is "
                        "NOT frozen. Suppressed targets: %s",
                        len(suppressed),
                        order_list,
                    )
                    return report

                # DEAD-MAN: a real rebalance was intended but EVERY order was
                # filtered. The book is NOT tracking its targets — it is frozen
                # on stale positions. Do not exit quietly: flag the run and alert.
                report["frozen"] = True
                report["freeze_reasons"] = reasons
                # Structured per-order detail so the notifier can explain WHICH
                # orders died and by how much, without re-reading the run log.
                report["frozen_orders"] = _suppressed_order_records(suppressed)
                order_lines = [_format_suppressed_line(r) for r in report["frozen_orders"]]
                order_list = "; ".join(order_lines)
                logger.error(
                    "REBALANCER FROZEN: all %d intended orders suppressed %s. "
                    "Portfolio NOT rebalanced — holding stale positions. Orders: %s",
                    len(suppressed),
                    reasons,
                    order_list,
                )
                self._alert(
                    f"🧊 <b>REBALANCER FROZEN — 0 of {len(orders_df)} orders executable</b>\n"
                    f"Reasons: {escape_html(reasons)}\n"
                    "Suppressed:\n" + "\n".join(f"• {escape_html(line)}" for line in order_lines) + "\n"
                    "Portfolio NOT rebalanced; holding stale positions. "
                    "Likely min_notional too high for account size (or stale/NaN data).",
                    site="frozen",
                )
            else:
                logger.warning("All %d orders filtered out: %s", len(orders_df), reasons)
            return report

        # Sell before buy: sort by Action descending (Sell > Buy)
        executable = executable.sort_values(by="Action", ascending=False)

        # Convert to broker-compatible format
        broker_orders_data: list[dict[str, Any]] = []
        # Submission INTENT, keyed by (symbol, side). Brokers hand back skipped and
        # failed rows carrying only symbol/side/qty/price/status/error — the
        # notional and the venue minimum are dropped at that boundary. Without this
        # map the broker-side freeze alert reports "SELL ADA $0.00" with no
        # shortfall, in precisely the trapped-residual case the alert exists to
        # explain (caught in review of #143).
        #
        # Deliberately a SIDE MAP rather than extra columns on broker_orders:
        # that frame is handed to third-party adapters, and widening it risks a
        # broker treating an unexpected column as an order field.
        #
        # A per-key FIFO, not a single dict entry: two orders can share
        # (symbol, side) — the intent/result matching below already carries a FIFO
        # for exactly that shape. With last-wins, both frozen rows would report the
        # LAST order's notional, so a $12 trapped residual reads as $1.20 (caught in
        # review of #144).
        order_intent: dict[tuple[str, str], deque[dict[str, float]]] = defaultdict(deque)
        for _, row in executable.iterrows():
            action = str(row.get("Action", "")).lower()
            side = "sell" if action == "sell" else "buy"
            symbol = str(row.get("Asset", ""))
            # NaN-safe: a mixed-column DataFrame fills a missing reduce_only with
            # NaN, and bool(NaN) is True — which would send OPENS reduce-only.
            _ro = row.get("reduce_only", False)
            reduce_only = bool(_ro) if pd.notna(_ro) else False
            broker_orders_data.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "qty": float(row.get("Adjusted Quantity", 0)),
                    "price": float(row.get("Price", 0)),
                    # Thread the rebalancer's close flag through so the broker sends
                    # a flat-target exit reduce-only (venue exempts it from the $10
                    # minimum; can't flip through zero). Defaults False for opens.
                    "reduce_only": reduce_only,
                }
            )
            order_intent[(symbol, side)].append(
                {
                    "notional": float(row.get("Notional Value", 0) or 0),
                    "min_notional": float(row.get("Min Notional", 0) or 0),
                }
            )

        broker_orders = pd.DataFrame(broker_orders_data)
        if broker_orders.empty:
            return report

        # --- Submission-time INTENT capture (issue #90) --------------------
        # Record every order's intent to the append-only ledger at the broker
        # submission call, BEFORE place_orders. This is the crash-durable audit
        # trail: an intent on disk before submission proves what the system meant
        # to do even if the run dies mid-submission. `intent_refs` is a per-(symbol,
        # side) FIFO of order_refs so results bind one-to-one to intents below.
        #
        # Per-(symbol, side) FIFO of order_refs. A None entry is a SENTINEL for an
        # order that executed but whose intent write failed (observe) — it reserves
        # the positional slot so results stay 1:1 without recording against a
        # never-written intent.
        intent_refs: dict[tuple[str, str], list[str | None]] = defaultdict(list)
        if ledger is not None and cycle_id is not None:
            dropped_idx: list[int] = []
            # Intents of the orders that SURVIVE to submission, rebuilt as we go.
            #
            # An order dropped under enforce still gets an `orders_details` row
            # (status FAILED), so the freeze builder processes it — but it never
            # consumed its own intent, which left the FIFO one ahead and handed
            # every later order the WRONG notional. Reproduced in review of #145:
            # a submitted-then-skipped ADA SELL of $12.00 was reported as $1.20,
            # the notional of a different, dropped order. A residual figure that
            # is confidently wrong is worse than none — it is the number a human
            # would act on.
            #
            # popleft-per-visit is positionally exact: this loop walks
            # broker_orders in submission order, which is the order the intents
            # were appended in, so the k-th visit to a (symbol, side) pops that
            # key's k-th intent.
            surviving_intent: dict[tuple[str, str], deque[dict[str, float]]] = defaultdict(deque)
            for i, (_, brow) in enumerate(broker_orders.iterrows()):
                sym = str(brow["symbol"])
                side = str(brow["side"]).strip().lower()
                this_intent: dict[str, float] = order_intent[(sym, side)].popleft() if order_intent[(sym, side)] else {}
                order_ref = f"{cycle_id}:{i}:{sym}:{side}"
                try:
                    ledger.record_intent(
                        cycle_id=cycle_id,
                        symbol=sym,
                        side=side,
                        order_ref=order_ref,
                        target_qty=_safe_float(brow.get("qty")),
                        limit_px=_safe_float(brow.get("price")),
                    )
                except Exception as exc:
                    if capture_fail_closed:
                        # ENFORCE: never trade live without the durable intent
                        # record — DROP the order (do not submit it).
                        logger.error(
                            "RECON ENFORCE: intent capture failed for %s %s — DROPPING order "
                            "(fail closed; not sent): %s",
                            side,
                            sym,
                            exc,
                        )
                        report["orders_details"].append(
                            {
                                "symbol": sym,
                                "action": side,
                                "status": "FAILED",
                                "error": f"intent capture failed under enforce (order dropped): {exc}",
                                # Its OWN intent, popped above — so the freeze
                                # alert reports this order's notional and not a
                                # neighbour's.
                                "notional": this_intent.get("notional", 0.0),
                                "min_notional": this_intent.get("min_notional", 0.0),
                            }
                        )
                        report["summary"]["total_failed"] += 1
                        report["recon_capture_dropped"] = report.get("recon_capture_dropped", 0) + 1
                        dropped_idx.append(i)
                        continue
                    # OBSERVE: ledger must never block execution — the order still
                    # executes. But we MUST still reserve a positional slot so result
                    # matching stays 1:1: append a None SENTINEL. Without it, a later
                    # fill for another same-(symbol, side) order would pop THIS
                    # order's would-be slot and bind to the WRONG intent. The sentinel
                    # absorbs this order's own result (recording nothing, since its
                    # intent was never durably written) and keeps every other order's
                    # result bound to its own ref.
                    logger.warning("Intent capture failed for %s %s (order still sent): %s", side, sym, exc)
                    intent_refs[(sym, side)].append(None)
                    surviving_intent[(sym, side)].append(this_intent)
                    continue
                intent_refs[(sym, side)].append(order_ref)
                surviving_intent[(sym, side)].append(this_intent)

            # The freeze builder must consume only the intents of orders that were
            # actually submitted; the dropped ones now travel on their own
            # orders_details rows instead.
            order_intent = surviving_intent

            # Under enforce, actually remove the dropped orders before submission.
            if dropped_idx:
                broker_orders = broker_orders.drop(broker_orders.index[dropped_idx]).reset_index(drop=True)
                if broker_orders.empty:
                    logger.error("RECON ENFORCE: all orders dropped (intent capture failed) — nothing sent.")
                    # Same rule as the batch-exception path: every exit from this
                    # method that leaves failures on the report must report them.
                    # Dropping the entire batch under enforce IS a total trading
                    # standstill — precisely when a human must be told.
                    self._report_order_failures(report)
                    return report

        def _record_result(symbol: str, side: str, status: str, filled_qty: Any = None, avg_px: Any = None) -> None:
            """Bind a broker result back to a captured intent (one-to-one, FIFO)."""
            if ledger is None or cycle_id is None:
                return
            pool = intent_refs.get((str(symbol), str(side).strip().lower()))
            if not pool:
                return
            ref = pool.pop(0)
            if ref is None:
                # Sentinel slot (intent write had failed): consume it to keep the
                # FIFO aligned, but record nothing — there is no durable intent to
                # bind this result to.
                return
            try:
                ledger.record_result(
                    order_ref=ref,
                    cycle_id=cycle_id,
                    status=_EXEC_STATUS_TO_LEDGER.get(str(status).strip().upper(), str(status).strip().lower()),
                    filled_qty=_safe_float(filled_qty),
                    avg_px=_safe_float(avg_px),
                )
            except Exception as exc:  # never let the ledger crash the run
                logger.warning("Result capture failed for %s %s: %s", side, symbol, exc)

        try:
            fills = broker.place_orders(broker_orders)
        except Exception as exc:
            logger.error("Broker place_orders failed: %s", exc)
            report["api_errors"].append({"stage": "place_orders", "error": str(exc)})
            for _, row in broker_orders.iterrows():
                report["orders_details"].append(
                    {
                        "symbol": row["symbol"],
                        "action": row["side"],
                        "status": "FAILED",
                        "error": str(exc),
                    }
                )
                report["summary"]["total_failed"] += 1
                # The whole batch failed at submission — record a failed result so
                # the ledger closes each intent (no dangling missed-fill timeouts).
                _record_result(row["symbol"], row["side"], "FAILED")
            # A batch-level submission failure is the WORST case (nothing traded
            # at all), so it must alert on the same path as a per-order rejection.
            # This early return previously skipped the failure report entirely.
            self._report_order_failures(report)
            return report

        # Process fills and failures
        if fills is not None and not fills.empty:
            n_skipped = 0
            n_skipped_sell = 0  # close-out SELLs the broker could not place (#81)
            for _, fill_row in fills.iterrows():
                # Strip whitespace before normalising: a broker that returns a
                # padded side/status (e.g. "SELL " / "SKIPPED ") must not slip
                # past the SKIPPED / close-out-SELL freeze counter (#81).
                #
                # A missing, empty or unrecognised status is UNKNOWN, never
                # FILLED (TOM-1336): defaulting it to FILLED is how an order the
                # venue never confirmed booked as a full fill.
                _raw_status = fill_row.get("status")
                status = "" if _raw_status is None or pd.isna(_raw_status) else str(_raw_status).strip().upper()
                if status not in _EXEC_STATUSES:
                    status = "UNKNOWN"
                side = str(fill_row.get("side", "")).strip().lower()
                if status == "SKIPPED":
                    # Broker intentionally did not place this order (sub-minimum /
                    # sub-precision dust). A clean no-op: neither executed nor
                    # failed — record for visibility but do not count it.
                    report["orders_details"].append(
                        {
                            "symbol": str(fill_row.get("symbol", "")),
                            "action": str(fill_row.get("side", "")),
                            "quantity": float(fill_row.get("qty", 0)),
                            "status": "SKIPPED",
                            "error": str(fill_row.get("error", "below exchange minimum (skipped)")),
                        }
                    )
                    n_skipped += 1
                    _record_result(fill_row.get("symbol", ""), side, "SKIPPED")
                    if side == "sell":
                        # A close-out/reduce the book WANTED to make but the broker
                        # could not place (sub-exchange-min). Tracked so an all-
                        # skipped batch with a trapped residual is not mistaken for
                        # a healthy run (#81).
                        n_skipped_sell += 1
                    continue
                if status == "WORKING":
                    # The venue accepted this order and it was still on the book
                    # when the confirmation window closed. NOT a failure and NOT a
                    # fill: record it for visibility, queue it for resolution by
                    # the next cycle, and leave the ledger intent deliberately
                    # unmatched — the order has no terminal result yet, and
                    # writing one would forge an outcome we have not observed.
                    order_id = str(fill_row.get("order_id") or "")
                    report["orders_details"].append(
                        {
                            "symbol": str(fill_row.get("symbol", "")),
                            "action": str(fill_row.get("side", "")),
                            "quantity": float(fill_row.get("qty", 0)),
                            "status": "WORKING",
                            "order_id": order_id,
                            "error": str(fill_row.get("error", "still working at the venue")),
                        }
                    )
                    report["summary"]["total_working"] += 1
                    # Consume this order's intent slot to keep the per-(symbol,
                    # side) FIFO aligned with the results that follow, and carry
                    # the ref forward so the late fill binds to THIS intent.
                    _pool = intent_refs.get((str(fill_row.get("symbol", "")), side))
                    intent_ref = _pool.pop(0) if _pool else None
                    # Record an explicit NON-TERMINAL `working` result. Leaving the
                    # intent unmatched would be read by Stage 7b as "submitted, no
                    # result observed" -- a MISSED FILL -- and with
                    # degraded_failed_streak=1 a single resting limit order would
                    # open a reconciliation break every cycle. That is the same
                    # cry-wolf this change exists to remove, moved from Telegram
                    # into recon. The record claims no outcome; it states the
                    # observed fact that the order is alive.
                    if ledger is not None and cycle_id is not None and intent_ref:
                        try:
                            ledger.record_result(
                                order_ref=str(intent_ref),
                                cycle_id=cycle_id,
                                status="working",
                                order_id=order_id,
                            )
                        except Exception:  # noqa: BLE001 - ledger must not break the run
                            logger.exception(
                                "Failed to record the working result for %s",
                                fill_row.get("symbol", ""),
                            )
                    if order_id:
                        working_to_queue.append(
                            {
                                "symbol": str(fill_row.get("symbol", "")),
                                "side": side,
                                "order_id": order_id,
                                "requested_qty": float(fill_row.get("requested_qty", 0) or 0.0),
                                "reason": str(fill_row.get("error", "")),
                                "order_ref": intent_ref,
                            }
                        )
                    else:
                        # Working but untrackable: no id means no later resolution,
                        # so this one really can go missing. Say so loudly rather
                        # than let it look like the handled case.
                        logger.error(
                            "Order for %s is working at the venue but carries NO order id — "
                            "it cannot be resolved next cycle and its fill may go unrecorded",
                            fill_row.get("symbol", ""),
                        )
                    continue
                if status == "UNKNOWN":
                    # The broker could not confirm what happened. Book NO fill.
                    # It is reported as a failure (so the operator is alerted and
                    # every FAILED consumer downstream sees a non-fill), and when
                    # it carries an order id it joins the working-order queue so
                    # the next cycle books the venue's real outcome.
                    _oid = fill_row.get("order_id")
                    order_id = "" if _oid is None or pd.isna(_oid) else str(_oid)
                    _err = fill_row.get("error")
                    _err = "" if _err is None or pd.isna(_err) else str(_err)
                    # A QUEUED unconfirmed order is non-terminal `unknown` in the
                    # ledger, the same rule as WORKING: the resolver books its real
                    # outcome against this order_ref next cycle. Booking it terminal
                    # `failed` let Stage 7b count it into the failure streak — one
                    # unconfirmed order opened a consecutive_failed break, three
                    # escalated to HALT. With no id or no queue nothing will ever
                    # resolve it, so it stays terminal `failed` and loud.
                    queued = bool(order_id) and working_store is not None
                    report["orders_details"].append(
                        {
                            "symbol": str(fill_row.get("symbol", "")),
                            "action": str(fill_row.get("side", "")),
                            "quantity": 0.0,
                            "status": "FAILED",
                            "fill_status": "UNKNOWN",
                            "queued": queued,
                            "order_id": order_id,
                            "error": (
                                f"fill UNCONFIRMED (broker status={_raw_status!r}); no fill booked"
                                + (f" — {_err}" if _err else "")
                            ),
                        }
                    )
                    report["summary"]["total_failed"] += 1
                    _pool = intent_refs.get((str(fill_row.get("symbol", "")), side))
                    intent_ref = _pool.pop(0) if _pool else None
                    if ledger is not None and cycle_id is not None and intent_ref:
                        try:
                            if queued:
                                ledger.record_result(
                                    order_ref=str(intent_ref), cycle_id=cycle_id, status="unknown", order_id=order_id
                                )
                            else:
                                ledger.record_result(order_ref=str(intent_ref), cycle_id=cycle_id, status="failed")
                        except Exception:  # noqa: BLE001 - ledger must not break the run
                            logger.exception(
                                "Failed to record the unconfirmed result for %s", fill_row.get("symbol", "")
                            )
                    if order_id:
                        working_to_queue.append(
                            {
                                "symbol": str(fill_row.get("symbol", "")),
                                "side": side,
                                "order_id": order_id,
                                "requested_qty": float(fill_row.get("requested_qty", 0) or 0.0),
                                "reason": f"fill unconfirmed: {_err}",
                                "order_ref": intent_ref,
                            }
                        )
                    else:
                        logger.error(
                            "Order for %s has an UNCONFIRMED fill and NO order id — it cannot be "
                            "resolved next cycle; reconcile it against the venue by hand",
                            fill_row.get("symbol", ""),
                        )
                    continue
                if status == "FAILED":
                    report["orders_details"].append(
                        {
                            "symbol": str(fill_row.get("symbol", "")),
                            "action": str(fill_row.get("side", "")),
                            "quantity": float(fill_row.get("qty", 0)),
                            "status": "FAILED",
                            "error": str(fill_row.get("error", "placement failed")),
                        }
                    )
                    report["summary"]["total_failed"] += 1
                    _record_result(fill_row.get("symbol", ""), side, "FAILED")
                    continue

                # FILLED or PARTIAL: a real (possibly partial) fill. Record the
                # ACTUAL filled qty the broker confirmed (issue #68) so the book's
                # recorded position matches reality; a partial is flagged so
                # monitors can see the unfilled remainder.
                is_partial = status == "PARTIAL"
                if is_partial:
                    logger.warning(
                        "Partial fill: %s %s — %s",
                        side,
                        str(fill_row.get("symbol", "")),
                        str(fill_row.get("error", "")),
                    )
                detail = {
                    "symbol": str(fill_row.get("symbol", "")),
                    "action": str(fill_row.get("side", "")),
                    "side": str(fill_row.get("side", "")),
                    "executed_quantity": float(fill_row.get("qty", 0)),
                    "executed_price": float(fill_row.get("price", 0)),
                    # #92: absent means the broker did not report one — UNKNOWN,
                    # not free. No LIVE broker's place_orders emits a fee key;
                    # only futures_paper does. The old `, 0` default is what made
                    # `fees_this_run` read $0.00 on every live run.
                    "fee": _to_fee(fill_row.get("fee")),
                    "status": "PARTIAL" if is_partial else "FILLED",
                }
                # Find reference price
                ref_price = 0.0
                matching = broker_orders[broker_orders["symbol"] == detail["symbol"]]
                if not matching.empty:
                    ref_price = float(matching.iloc[0]["price"])
                detail["reference_price"] = ref_price
                exec_price = detail["executed_price"]
                if ref_price > 0 and exec_price > 0:
                    detail["spread_pct"] = abs(exec_price - ref_price) / ref_price
                else:
                    detail["spread_pct"] = 0.0

                report["orders_details"].append(detail)
                report["summary"]["total_executed"] += 1
                _record_result(
                    detail["symbol"],
                    side,
                    "PARTIAL" if is_partial else "FILLED",
                    filled_qty=detail["executed_quantity"],
                    avg_px=detail["executed_price"],
                )
                if is_partial:
                    # A partial fill DID execute (so it counts as executed and keeps
                    # the freeze logic honest), but the unfilled remainder is real
                    # residual exposure — surface it as a first-class incomplete
                    # outcome so the notifier can raise an exception on it instead of
                    # reading the run as a clean success.
                    report["summary"]["total_partial"] += 1

            # Freeze guard (issue #81): the broker returned rows, but if EVERY
            # submitted order was SKIPPED (zero fills, zero hard failures) and at
            # least one of those skips was a close-out SELL, the book has a
            # trapped residual it wanted to exit but could not place at the venue.
            # Post the dust-fix this surfaces as SKIPPED instead of the pre-PR
            # FAILED count, so without this it would read as a healthy run. Treat
            # it as a freeze and alert — the trapped-residual signal must not be
            # lost just because the orders were *executable* upstream.
            # Fires whenever zero orders filled AND at least one close-out SELL was
            # skipped — regardless of whether other orders ALSO hard-failed. The
            # trapped-residual signal must NOT be masked just because total_failed>0
            # (a concurrent placement failure would otherwise hide the retained
            # residual behind the generic FAILED alert).
            if report["summary"]["total_executed"] == 0 and n_skipped_sell >= 1:
                report["frozen"] = True
                reasons = {
                    "skipped_sell": n_skipped_sell,
                    "skipped_total": n_skipped,
                    "failed": int(report["summary"]["total_failed"]),
                }
                report["freeze_reasons"] = reasons

                # Same explainability contract as the upstream freeze: name each
                # order and why it could not go through, not just a count. These
                # rows come from the BROKER (post-submission), so the fields are
                # the order_details shape rather than the rebalancer's columns.
                # Recover notional + venue minimum from the submission intent: the
                # broker's own rows do not carry them (see order_intent above), so
                # reading only `d` yields $0.00 and no shortfall — useless in exactly
                # the trapped-residual case this alert exists for.
                def _frozen_record(d: dict[str, Any]) -> dict[str, Any]:
                    # Strip before matching: `orders_details` carries the broker's
                    # RAW side, and a padded " SELL " (#81's shape) misses the
                    # intent key — reinstating the "$0.00" bug in the very case
                    # the recovery exists for (caught in review of #144).
                    symbol = str(d.get("symbol", "")).strip()
                    action = str(d.get("action", "")).strip()
                    # Consume the FIFO so duplicate (symbol, side) orders each get
                    # their OWN intent rather than all reading the last one.
                    #
                    # ONLY when this row does not already carry its own numbers.
                    # An enforce-DROPPED order brings its intent along on its own
                    # details row and was never added to the surviving FIFO, so
                    # popping for it would consume a SUBMITTED order's intent and
                    # leave that order reading $0.00 — the same misalignment one
                    # step removed.
                    own_notional = d.get("notional") or d.get("target_value")
                    own_min = d.get("min_notional")
                    if own_notional is None or own_min is None:
                        pending = order_intent.get((symbol, action.lower()))
                        intent = pending.popleft() if pending else {}
                    else:
                        intent = {}
                    notional = own_notional or intent.get("notional") or 0
                    min_notional = own_min or intent.get("min_notional") or 0
                    return {
                        "symbol": symbol,
                        "action": action.upper(),
                        "notional_usd": float(notional),
                        "min_notional_usd": float(min_notional),
                        "status": str(d.get("status", "")),
                        "reason": str(d.get("reason") or d.get("error") or ""),
                    }

                report["frozen_orders"] = [
                    _frozen_record(d) for d in report["orders_details"] if d.get("status") in {"SKIPPED", "FAILED"}
                ]
                for rec in report["frozen_orders"]:
                    mn, n = rec["min_notional_usd"], rec["notional_usd"]
                    if mn > 0 and 0 < n < mn:
                        rec["shortfall_usd"] = mn - n
                skipped_lines = [
                    _format_suppressed_line(r) for r in report["frozen_orders"] if r["status"] == "SKIPPED"
                ]
                skipped_list = "; ".join(skipped_lines)
                failed_n = int(report["summary"]["total_failed"])
                failed_note = f"; {failed_n} order(s) also hard-failed" if failed_n else ""
                logger.error(
                    "REBALANCER FROZEN: 0 fills; %d close-out SELL(s) skipped "
                    "sub-minimum (trapped residual)%s. Residual exposure RETAINED — "
                    "book NOT rebalanced. Skipped: %s",
                    n_skipped_sell,
                    failed_note,
                    skipped_list,
                )
                self._alert(
                    "🧊 <b>REBALANCER FROZEN — trapped residual</b>\n"
                    f"0 fills; {n_skipped_sell} close-out SELL(s) below exchange "
                    f"minimum{failed_note}; residual exposure RETAINED.\n"
                    f"Skipped: {escape_html(skipped_list)}\n"
                    "Book NOT rebalanced — monitor for a trapped residual.",
                    site="trapped_residual",
                )
        else:
            # Submitted orders but received zero fills — broker-level failure.
            order_list = ", ".join(
                f"{r['side'].upper()} {r['qty']:.4f} {r['symbol']}" for _, r in broker_orders.iterrows()
            )
            logger.error(
                "Broker returned 0 fills for %d submitted orders: %s",
                len(broker_orders),
                order_list,
            )
            # Was the one unwrapped site: a raising notify here failed the run.
            self._alert(
                f"⚠️ <b>0 FILLS for {len(broker_orders)} orders</b>\n"
                f"Submitted: {escape_html(order_list)}\n"
                "Check broker connectivity — portfolio NOT rebalanced.",
                site="zero_fills",
            )

        # Close any INTENT with no observed result as a TIMEOUT (a missed fill:
        # submitted but the broker returned nothing for it). Authority rule: the
        # internal ledger is the truth for intent — this is the class the exchange
        # alone cannot prove. Done after all fills are matched so only genuinely
        # unresolved intents remain.
        if ledger is not None and cycle_id is not None:
            for (sym, side), refs in intent_refs.items():
                for ref in refs:
                    if ref is None:
                        continue  # sentinel: no durable intent to close
                    try:
                        ledger.record_result(order_ref=ref, cycle_id=cycle_id, status="timeout")
                    except Exception as exc:  # never let the ledger crash the run
                        logger.warning("Timeout capture failed for %s %s: %s", side, sym, exc)
            intent_refs.clear()

        report["summary"]["total_value"] = sum(
            d.get("executed_quantity", 0) * d.get("executed_price", 0)
            for d in report["orders_details"]
            if d.get("status") in ("FILLED", "PARTIAL")
        )

        # Persist BEFORE the alerting call below: queuing a working order is the
        # durable step that lets its fill be recovered next cycle, and it must not
        # be skipped because a notification path raised.
        if working_to_queue and working_store is not None:
            for rec in working_to_queue:
                try:
                    working_store.record(cycle_id=str(cycle_id or ""), **rec)
                except Exception:
                    # A lost queue entry is a lost fill, so this is loud and it
                    # does NOT swallow: the run continues (the order is live and
                    # the position reconciles next cycle regardless), but the
                    # operator must see that automatic resolution will not happen.
                    logger.exception(
                        "FAILED to queue working order %s %s (id=%s) for next-cycle "
                        "resolution — its fill will not be booked automatically",
                        rec.get("side"),
                        rec.get("symbol"),
                        rec.get("order_id"),
                    )
        elif working_to_queue:
            logger.warning(
                "%d order(s) working at the venue but no working-order store is "
                "configured — their fills will not be booked automatically",
                len(working_to_queue),
            )

        self._report_order_failures(report)
        return report

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
                order_fee = _to_fee(detail.get("fee"))
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
