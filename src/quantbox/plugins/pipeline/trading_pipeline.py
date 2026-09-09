"""Full trading pipeline plugin.

Orchestrates: strategy execution -> aggregation -> risk transforms ->
order generation -> execution -> artifact storage.

Ported from the quantlab ``trading.py`` workflow, ``orders.py``, and
``portfolio.py`` into a single PipelinePlugin that the quantbox runner
can invoke via ``pipeline.run()``.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal, getcontext
from typing import Any

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
from quantbox.reconciliation.ledger import EXEC_STATUS_TO_LEDGER
from quantbox.reconciliation.working_orders import DEFAULT_MAX_AGE_DAYS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (from quantlab orders.py)
# ---------------------------------------------------------------------------
DEFAULT_CAPITAL_AT_RISK = 1.0
DEFAULT_MIN_NOTIONAL = 1.0
DEFAULT_MIN_TRADE_SIZE = 0.01
DEFAULT_STABLE_COIN = "USDC"

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


class ReconEnforcementError(RuntimeError):
    """A CRITICAL enforce-mode reconciliation failure that must NOT be swallowed.

    Raised when the enforcement authority itself is untrustworthy under acknowledged
    enforce (e.g. the persisted state file cannot be written). Unlike ordinary recon
    faults — which are observability losses and never fatal — this must escape the
    guard and fail the run, so the book stops rather than silently trading the next
    cycle from stale/absent enforcement state.
    """


def _atomic_write_text(path: Any, text: str) -> None:
    """Durably replace ``path`` with ``text``: write a temp file in the same dir,
    fsync it, then ``os.replace`` (atomic on POSIX). A crash can never leave the
    target truncated/half-written — the reader sees either the old file or the new
    one, never a corrupt one. Used for the recon state file, which IS the
    enforcement authority (a corrupt state must never silently fail open).
    """
    import os
    import tempfile
    from pathlib import Path

    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        # fsync the parent directory so the rename itself is crash-durable on POSIX
        # (a bare os.replace is not persisted until the dir entry is synced).
        with contextlib.suppress(OSError, AttributeError):
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# Ledger result status mapped from an execution ``orders_details`` row status.
# Backwards-compatible alias. The map itself now lives in
# ``reconciliation.ledger`` beside the vocabulary it targets, because the
# out-of-cycle working-order resolver must translate into the same map and a
# private second copy is how two callers start disagreeing.
_EXEC_STATUS_TO_LEDGER = EXEC_STATUS_TO_LEDGER


@dataclass
class _ReconCtx:
    """Per-cycle reconciliation context, built once and shared by the pre-execution
    gate (preflight) and the post-execution evaluator so config is parsed once and
    the SAME ledger + persisted state back both stages.

    Types are intentionally ``Any`` so the (optional) ``quantbox.reconciliation``
    subpackage is imported lazily, never at module load.
    """

    book_key: str
    root: Any
    tol: Any
    cycle_id: str
    mode: str
    enforce_refused: bool
    ledger: Any
    machine: Any
    state_path: Any
    persisted: dict[str, Any]


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


# Funding reporting window. The pipeline runs daily, so the window that ends at
# `asof` starts 24h earlier; Hyperliquid settles funding hourly, giving ~24
# payments per open position per run.
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
        window_start = _funding_window_start(asof)
        charge = broker.fetch_funding_payments(window_start)
        if charge is None:
            logger.error(
                "Funding UNMEASURED for the window starting %s — reported as UNKNOWN, not $0.00",
                window_start,
            )
        else:
            logger.info("Realised funding since %s: %.4f (negative = paid)", window_start, charge)
        return charge

    logger.warning(
        "Broker %s exposes no funding source (neither apply_funding nor "
        "fetch_funding_payments) — funding reported UNKNOWN, not $0.00",
        type(broker).__name__,
    )
    return None


def _funding_window_start(asof: str) -> str:
    """ISO start of the funding reporting window ending at ``asof``."""
    try:
        ts = pd.Timestamp(asof)
    except Exception:  # noqa: BLE001 - unparseable asof: fall back to a live window
        ts = None
    if ts is None or pd.isna(ts):
        logger.warning("Unparseable asof %r for the funding window — using the last %s", asof, FUNDING_LOOKBACK)
        ts = pd.Timestamp.utcnow()
    return (ts - FUNDING_LOOKBACK).isoformat()


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
                "strategy_weights": {
                    "type": "object",
                    "description": "Override strategy-level weights {name: weight}.",
                },
                "capital_at_risk": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 1.0,
                },
                "stable_coin_symbol": {"type": "string", "default": "USDC"},
                "risk": {
                    "type": "object",
                    "properties": {
                        "tranches": {"type": "integer", "minimum": 1, "default": 1},
                        "max_leverage": {"type": "number", "minimum": 0, "default": 1},
                        "allow_short": {"type": "boolean", "default": False},
                    },
                },
                "min_trade_size": {
                    "type": "number",
                    "minimum": 0,
                    "default": 0.01,
                    "description": "Min abs(weight delta) to consider a trade.",
                },
                "min_notional": {"type": "number", "minimum": 0, "default": 1.0},
                "scaling_factor_min": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.9,
                },
                "trading_enabled": {"type": "boolean", "default": True},
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

        # Run-state exception inputs the Tier-2 notifier consumes (issue #62):
        # API errors caught (but survived) during the run. Data-staleness and
        # pipeline-failure are derived later from the feed + the run completing.
        api_errors: list[dict[str, Any]] = []

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
        # Store prices as wide-format parquet
        prices_wide = market_data_dict.get("prices", pd.DataFrame())
        if not prices_wide.empty:
            store.put_parquet(
                "prices", prices_wide.reset_index() if isinstance(prices_wide.index, pd.DatetimeIndex) else prices_wide
            )
        else:
            store.put_parquet("prices", pd.DataFrame())

        # Build data dict for strategies (quantlab convention)
        market_data = self._build_market_data(market_data_dict, universe)

        # --- Stage 2: Strategy Execution ---
        if strategies:
            strategy_results = self._run_strategy_plugins(
                strategies,
                strategies_cfg,
                market_data,
            )
        else:
            strategy_results = self._run_strategies(market_data, strategies_cfg, params)

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
        # Use injected aggregator if provided via kwargs, else fallback
        _aggregator = kwargs.get("aggregator")
        if _aggregator is not None:
            agg_data = {**market_data, "strategy_results": strategy_results}
            agg_result = _aggregator.run(data=agg_data, params=aggregator_cfg.get("params", {}))
            final_weights = agg_result.get("simple_weights", {})
            if not final_weights:
                # Try extracting from weights DataFrame
                w_df = agg_result.get("weights", pd.DataFrame())
                if isinstance(w_df, pd.DataFrame) and not w_df.empty:
                    last = w_df.iloc[-1]
                    final_weights = {str(k): float(v) for k, v in last.items()}
        else:
            final_weights = self._aggregate_strategies(strategy_results, params)
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

        # --- Stage 4: Risk Transforms + Stage 5: Order Generation ---
        if rebalancer is not None and mode not in ("backtest",) and broker is not None:
            # Use injected rebalancer for risk transforms + order generation
            rebal_params = dict(rebalancer_cfg.get("params", {}))
            rebal_params["strategy_results"] = strategy_results
            rebal_params.setdefault("capital_at_risk", params.get("capital_at_risk", DEFAULT_CAPITAL_AT_RISK))
            rebal_params.setdefault("stable_coin_symbol", params.get("stable_coin_symbol", DEFAULT_STABLE_COIN))
            rebal_params.setdefault("exclusions", params.get("exclusions", []))
            rebal_params.setdefault("strategy_weights", params.get("strategy_weights", {}))
            order_result = rebalancer.generate_orders(
                weights=final_weights,
                broker=broker,
                params=rebal_params,
            )
            final_weights = order_result.get("weights", final_weights)
        else:
            # Fallback: use internal risk transforms
            final_weights = self._apply_risk_transforms(final_weights, strategy_results, params)

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
                notes={"kind": "trading", **token_policy_notes},
            )

        # Live / paper execution path
        stable_coin = str(params.get("stable_coin_symbol", DEFAULT_STABLE_COIN))
        capital_at_risk = float(params.get("capital_at_risk", DEFAULT_CAPITAL_AT_RISK))

        if rebalancer is not None:
            # Already ran rebalancer above; reuse result
            pass
        else:
            order_result = self._generate_orders(
                broker=broker,
                weights=final_weights,
                capital_at_risk=capital_at_risk,
                stable_coin=stable_coin,
                params=params,
            )
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
        # applied. Guarded so a recon fault fails safe to no-gate (never fatal).
        recon_ctx = None
        recon_preflight: dict[str, Any] = {}
        try:
            recon_ctx = self._recon_load(params, store.run_id, asof)
            if recon_ctx is not None:
                recon_preflight = self._recon_preflight(recon_ctx, broker)
        except Exception as exc:  # noqa: BLE001 - recon must never be fatal
            recon_ctx = None
            if self._enforce_requested(params):
                # FAIL CLOSED: under acknowledged enforce, an inability to load or
                # compute the persisted gate must NOT let orders through — block
                # this cycle. (Observe mode fails open to no-gate: no gating claimed.)
                logger.error(
                    "RECON ENFORCE: preflight failed under acknowledged enforce — failing CLOSED (HALT) "
                    "this cycle; no orders will be sent. Error: %s",
                    exc,
                )
                recon_preflight = {
                    "applied": True,
                    "orders_allowed": False,
                    "reduce_only": False,
                    "enforced": True,
                    "gate_from_state": "unknown",
                    "fail_closed": True,
                }
            else:
                logger.error("Recon preflight failed (observe, non-fatal, fail-safe to no-gate): %s", exc)
                recon_preflight = {}

        gate_applied = bool(recon_preflight.get("applied"))

        # --- Stage 6c: resolve orders left WORKING by a previous cycle ---
        # A priced order goes to the venue as a LIMIT order and may rest on the
        # book long after the run that placed it exited. Those are queued rather
        # than alarmed; this is where the book finds out what became of them, so a
        # fill that landed minutes (or hours) later still reaches the accounts.
        # Guarded: resolution is bookkeeping and must never block trading.
        working_store = None
        resolved_working: list[dict[str, Any]] = []
        try:
            working_store = self._working_store(params)
            if working_store is not None:
                resolved_working = self._resolve_working_orders(
                    working_store,
                    broker,
                    ledger=recon_ctx.ledger if recon_ctx is not None else None,
                )
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
            ledger=recon_ctx.ledger if recon_ctx is not None else None,
            cycle_id=recon_ctx.cycle_id if recon_ctx is not None else None,
            # Enforce-mode pre-execution gate (permissive defaults when not applied).
            gate_orders_allowed=recon_preflight.get("orders_allowed", True) if gate_applied else True,
            gate_reduce_only=recon_preflight.get("reduce_only", False) if gate_applied else False,
            # Under enforce, a failed intent write drops the order (fail closed) —
            # never trade live without the durable intent record.
            capture_fail_closed=bool(recon_ctx is not None and recon_ctx.tol.is_enforce),
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
            if hasattr(broker, "get_equity"):
                portfolio_value_post = float(broker.get_equity())
                cash2 = broker.get_cash() or {}
                cash_usd_post = sum(float(v) for v in cash2.values())
            else:
                cash2 = broker.get_cash() or {}
                cash_usd_post = sum(float(v) for v in cash2.values())
                pos2 = broker.get_positions()
                if pos2 is not None and len(pos2) > 0:
                    snap = broker.get_market_snapshot(pos2["symbol"].tolist())
                    if snap is not None and "mid" in snap.columns:
                        merged = pos2.merge(snap[["symbol", "mid"]], on="symbol", how="left")
                        merged["mid"] = merged["mid"].fillna(0).astype(float)
                        merged["qty"] = merged["qty"].astype(float)
                        portfolio_value_post = cash_usd_post + (merged["qty"] * merged["mid"]).sum()
                    else:
                        portfolio_value_post = cash_usd_post
                else:
                    portfolio_value_post = cash_usd_post
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
        recon_notes = self._run_reconciliation(
            params=params,
            broker=broker,
            final_weights=final_weights,
            orders_df=orders_df,
            execution_report=execution_report,
            portfolio_value=portfolio_value_post,
            stable_coin=stable_coin,
            asof=asof,
            run_id=store.run_id,
            # Intent was captured at submission (Stage 7) whenever recon is active,
            # so Stage 7b reads the ledger back instead of reconstructing it.
            intent_captured=recon_ctx is not None,
        )
        # Attach this cycle's pre-execution gate outcome for observability.
        if recon_preflight and isinstance(recon_notes, dict):
            recon_notes["preflight_gate"] = recon_preflight
            recon_notes["gate_applied"] = gate_applied
            if execution_report.get("recon_gated"):
                recon_notes["recon_gated"] = execution_report["recon_gated"]

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
            "pipeline_ok": True,
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
                "rebalance_frozen": bool(execution_report.get("frozen", False)),
                "freeze_reasons": execution_report.get("freeze_reasons", {}),
                # Per-order freeze detail (symbol/side/notional/min/shortfall).
                # freeze_reasons alone is a histogram — it says three orders died,
                # never which or by how much, so every page started with a log dig.
                "frozen_orders": execution_report.get("frozen_orders", []),
                "quiet_day": bool(execution_report.get("quiet_day", False)),
                "quiet_reasons": execution_report.get("quiet_reasons", {}),
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
                **token_policy_notes,
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
    # Stage 2: Strategy execution
    # ==================================================================
    def _run_strategies(
        self,
        market_data: dict[str, Any],
        strategies_cfg: list[dict[str, Any]],
        pipeline_params: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Import and run each strategy, collecting results."""
        results: dict[str, dict[str, Any]] = {}

        for strat_cfg in strategies_cfg:
            name = strat_cfg["name"]
            weight = float(strat_cfg.get("weight", 1.0))
            strat_params = strat_cfg.get("params", {})

            try:
                module = importlib.import_module(f"quantbox.plugins.strategies.{name}")
            except ImportError:
                logger.error("Could not import strategy '%s'", name)
                raise

            result = module.run(data=market_data, params=strat_params)

            # Normalize multi-level weight columns
            weights_df = result.get("weights", pd.DataFrame())
            if isinstance(weights_df, pd.DataFrame) and weights_df.columns.nlevels > 1:
                if weights_df.droplevel("ticker", axis=1).columns.unique().shape[0] > 1:
                    logger.warning(
                        "Strategy %s has multiple weights columns: %s",
                        name,
                        weights_df.droplevel("ticker", axis=1).columns.unique().tolist(),
                    )
                weights_df = weights_df.T.groupby("ticker").sum().T
                result["weights"] = weights_df

            results[name] = {"result": result, "weight": weight}
            logger.info("Strategy '%s' completed (weight=%.2f)", name, weight)

        return results

    def _run_strategy_plugins(
        self,
        strategy_plugins: list[StrategyPlugin],
        strategies_cfg: list[dict[str, Any]],
        market_data: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Run injected StrategyPlugin instances, collecting results."""
        results: dict[str, dict[str, Any]] = {}

        for i, strat in enumerate(strategy_plugins):
            strat_cfg = strategies_cfg[i] if i < len(strategies_cfg) else {}
            weight = float(strat_cfg.get("weight", 1.0))
            strat_params = strat_cfg.get("params", {})

            result = strat.run(data=market_data, params=strat_params)

            # Normalize multi-level weight columns (same as _run_strategies)
            weights_df = result.get("weights", pd.DataFrame())
            if isinstance(weights_df, pd.DataFrame) and weights_df.columns.nlevels > 1:
                if weights_df.droplevel("ticker", axis=1).columns.unique().shape[0] > 1:
                    logger.warning(
                        "Strategy %s has multiple weights columns: %s",
                        strat.meta.name,
                        weights_df.droplevel("ticker", axis=1).columns.unique().tolist(),
                    )
                weights_df = weights_df.T.groupby("ticker").sum().T
                result["weights"] = weights_df

            results[strat.meta.name] = {"result": result, "weight": weight}
            logger.info("Strategy plugin '%s' completed (weight=%.2f)", strat.meta.name, weight)

        return results

    # ==================================================================
    # Stage 3: Strategy aggregation
    # ==================================================================
    def _aggregate_strategies(
        self,
        strategy_results: dict[str, dict[str, Any]],
        params: dict[str, Any],
    ) -> dict[str, float]:
        """Aggregate multi-strategy weights into a single weight per asset.

        Ported from quantlab trading.py aggregation logic:
        1. Concat weights with MultiIndex columns (strategy, ticker)
        2. Multiply by account-level strategy weights
        3. Sum to single weight per ticker
        """
        names = list(strategy_results.keys())
        if not names:
            return {}

        weight_overrides = params.get("strategy_weights", {})

        weight_dfs = []
        account_weights = []
        for sname in names:
            sinfo = strategy_results[sname]
            w_df = sinfo["result"].get("weights", pd.DataFrame())
            if w_df is None or (isinstance(w_df, pd.DataFrame) and w_df.empty):
                continue

            weight_dfs.append(w_df)
            w = float(weight_overrides.get(sname, sinfo["weight"]))
            account_weights.append(w)

        if not weight_dfs:
            return {}

        if len(weight_dfs) == 1:
            # Single strategy: just use last row scaled by weight
            df = weight_dfs[0]
            last = df.iloc[-1] if isinstance(df, pd.DataFrame) else df
            return {str(k): float(v) * account_weights[0] for k, v in last.items()}

        # Multi-strategy aggregation
        try:
            combined = pd.concat(weight_dfs, axis=1, keys=names[: len(weight_dfs)], names=["strategy"])
            acct_w = pd.Series(
                account_weights,
                index=pd.Index(names[: len(weight_dfs)], name="strategy"),
            )
            weighted = combined.mul(acct_w, level="strategy")
            # Drop strategy level and sum per ticker
            flat = weighted.droplevel(0, axis=1)
            # If multiple strategies have the same ticker, sum their weights
            if isinstance(flat.columns, pd.MultiIndex) or flat.columns.duplicated().any():
                flat = flat.T.groupby(level=0).sum().T
            aggregated = flat.iloc[-1]
        except Exception:
            # Fallback: manual aggregation
            agg: dict[str, float] = {}
            for i, df in enumerate(weight_dfs):
                last = df.iloc[-1] if isinstance(df, pd.DataFrame) else df
                w = account_weights[i]
                for ticker, val in last.items():
                    ticker_str = str(ticker)
                    agg[ticker_str] = agg.get(ticker_str, 0.0) + float(val) * w
            return agg

        return {str(k): float(v) for k, v in aggregated.items()}

    # ==================================================================
    # Stage 4: Risk transforms
    # ==================================================================
    def _apply_risk_transforms(
        self,
        weights: dict[str, float],
        strategy_results: dict[str, dict[str, Any]],
        params: dict[str, Any],
    ) -> dict[str, float]:
        """Apply tranching, leverage cap, and negative-weight clamping.

        Ported from quantlab trading.py risk management section.
        """
        risk_cfg = params.get("risk", {})
        tranches = int(risk_cfg.get("tranches", 1))
        max_leverage = float(risk_cfg.get("max_leverage", 1))
        allow_short = bool(risk_cfg.get("allow_short", False))

        s = pd.Series(weights, dtype=float)

        # Tranching: rolling mean over N days (requires historical weights)
        if tranches > 1:
            # We need the full time series of aggregated weights
            # Build from strategy results
            try:
                names = list(strategy_results.keys())
                weight_overrides = params.get("strategy_weights", {})
                weight_dfs = []
                account_weights = []
                for sname in names:
                    sinfo = strategy_results[sname]
                    w_df = sinfo["result"].get("weights", pd.DataFrame())
                    if w_df is not None and not w_df.empty:
                        weight_dfs.append(w_df)
                        account_weights.append(float(weight_overrides.get(sname, sinfo["weight"])))

                if len(weight_dfs) == 1:
                    full_ts = weight_dfs[0] * account_weights[0]
                else:
                    combined = pd.concat(
                        weight_dfs,
                        axis=1,
                        keys=names[: len(weight_dfs)],
                        names=["strategy"],
                    )
                    acct_w = pd.Series(
                        account_weights,
                        index=pd.Index(names[: len(weight_dfs)], name="strategy"),
                    )
                    weighted = combined.mul(acct_w, level="strategy")
                    full_ts = weighted.droplevel(0, axis=1)
                    if isinstance(full_ts.columns, pd.MultiIndex) or full_ts.columns.duplicated().any():
                        full_ts = full_ts.T.groupby(level=0).sum().T

                smoothed = full_ts.rolling(window=tranches).mean().iloc[-1]
                s = smoothed
            except Exception:
                logger.warning("Tranching failed, using un-smoothed weights")

        # Max leverage
        gross = s.abs().sum()
        if gross > max_leverage:
            logger.warning("Leverage %.4f exceeds max_leverage %.1f, scaling down", gross, max_leverage)
            s = s / gross * max_leverage

        # Clamp negatives
        if not allow_short:
            s = s.clip(lower=0)

        # Drop zeros and sort
        s = s[s != 0].sort_values(ascending=False)

        return {str(k): float(v) for k, v in s.items()}

    # ==================================================================
    # Stage 5: Order generation
    # ==================================================================
    def _generate_orders(
        self,
        broker: BrokerPlugin,
        weights: dict[str, float],
        capital_at_risk: float,
        stable_coin: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate rebalancing DataFrame and executable orders.

        Ported from quantlab ``orders.py:generate_portfolio_orders()``.
        """
        min_notional_cfg = float(params.get("min_notional", DEFAULT_MIN_NOTIONAL))
        min_trade_size = float(params.get("min_trade_size", DEFAULT_MIN_TRADE_SIZE))
        scaling_factor_min = float(params.get("scaling_factor_min", 0.9))
        exclusions = list(params.get("exclusions", [])) + [stable_coin]
        exclusions = list(set(exclusions))

        # Fetch current state from broker
        positions_df = broker.get_positions()
        cash = broker.get_cash() or {}
        cash_available = float(cash.get(stable_coin, cash.get("USD", 0.0)))

        # Build holdings dict
        current_holdings: dict[str, float] = {}
        if positions_df is not None and not positions_df.empty:
            for _, row in positions_df.iterrows():
                sym = str(row.get("symbol", ""))
                qty = float(row.get("qty", 0))
                if qty != 0:
                    current_holdings[sym] = qty

        # Get prices via broker market snapshot
        all_symbols = sorted(set(list(weights.keys()) + list(current_holdings.keys())))
        all_symbols = [s for s in all_symbols if s not in exclusions]

        price_map: dict[str, float | None] = {}
        if all_symbols:
            try:
                snap = broker.get_market_snapshot(all_symbols)
                if snap is not None and not snap.empty:
                    for _, row in snap.iterrows():
                        sym = str(row.get("symbol", ""))
                        mid = row.get("mid")
                        if mid is not None and not (isinstance(mid, float) and np.isnan(mid)):
                            price_map[sym] = float(mid)
            except Exception:
                pass

        def get_price(asset: str) -> float | None:
            return price_map.get(asset)

        # Portfolio value -- prefer broker.get_equity() for derivatives
        # brokers where cash + sum(qty * price) is wrong for shorts.
        if hasattr(broker, "get_equity"):
            total_value = float(broker.get_equity())
        else:
            total_value = max(0, cash_available)
            for asset, qty in current_holdings.items():
                if asset == stable_coin:
                    total_value += max(0, qty)
                elif asset not in exclusions:
                    p = get_price(asset)
                    if p is not None and qty > 0:
                        total_value += qty * p

        if total_value <= 0:
            logger.error("Portfolio value is zero or negative")
            return {
                "orders": pd.DataFrame(),
                "rebalancing": pd.DataFrame(),
                "total_value": 0.0,
            }

        # Apply capital-at-risk
        adjusted_weights = {a: w * capital_at_risk for a, w in weights.items()}

        # Target positions
        target_positions: dict[str, float] = {}
        for asset, weight in adjusted_weights.items():
            p = get_price(asset)
            if p is not None and p > 0:
                target_positions[asset] = (total_value * weight) / p

        # Rebalancing DataFrame
        rebalancing_df = self._build_rebalancing(
            current_holdings=current_holdings,
            target_positions=target_positions,
            get_price=get_price,
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
        }

    # ------------------------------------------------------------------

    def _build_rebalancing(
        self,
        current_holdings: dict[str, float],
        target_positions: dict[str, float],
        get_price: Callable[[str], float | None],
        total_value: float,
        strategy_weights: dict[str, float],
        exclusions: list[str],
    ) -> pd.DataFrame:
        """Build the rebalancing analysis DataFrame.

        Ported from quantlab ``orders.py:generate_rebalancing_dataframe()``.
        """
        assets = sorted(set(current_holdings.keys()) | set(target_positions.keys()))
        assets = [a for a in assets if a not in exclusions]

        rows: list[dict[str, Any]] = []
        for asset in assets:
            current_qty = current_holdings.get(asset, 0.0)
            price = get_price(asset)
            current_value = current_qty * price if price else 0
            target_qty = target_positions.get(asset, 0.0)
            target_value = target_qty * price if price else 0
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
            return pd.DataFrame(
                columns=[
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
                ]
            )

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
    def _report_order_failures(self, report: dict[str, Any], broker: BrokerPlugin | None) -> None:
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
        notify = getattr(broker, "notify", None)
        if callable(notify):
            try:
                notify(
                    f"❌ <b>{n_failed} ORDER(S) FAILED</b>\n{failed_list}\n"
                    "Targets NOT reached for these symbols; exposure is unchanged."
                )
            except Exception as exc:  # never let alerting crash the run
                logger.warning("Order-failure notify failed: %s", exc)

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
            notify = getattr(broker, "notify", None)
            if callable(notify):
                try:
                    notify(
                        f"🛑 <b>RECON ENFORCE — HALT</b>\n{len(orders_df)} order(s) NOT sent; "
                        "book is in HALT from the prior cycle. Manual clear required."
                    )
                except Exception as exc:  # never let alerting crash the run
                    logger.warning("Recon HALT notify failed: %s", exc)
            return report

        executable = (
            orders_df[orders_df.get("Executable", pd.Series(dtype=bool))].copy()
            if "Executable" in orders_df.columns
            else orders_df
        )

        # --- Pre-execution FLATTEN gate (enforce mode, issue #90) ----------
        # Keep only exposure-reducing orders (clamped so they never cross zero).
        if gate_reduce_only and isinstance(executable, pd.DataFrame) and not executable.empty:
            before_n = len(executable)
            executable = self._filter_reduce_only(executable, broker)
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
                notify = getattr(broker, "notify", None)
                if notify:
                    try:
                        notify(
                            f"🧊 <b>REBALANCER FROZEN — 0 of {len(orders_df)} orders executable</b>\n"
                            f"Reasons: {reasons}\n"
                            "Suppressed:\n" + "\n".join(f"• {line}" for line in order_lines) + "\n"
                            "Portfolio NOT rebalanced; holding stale positions. "
                            "Likely min_notional too high for account size (or stale/NaN data)."
                        )
                    except Exception as exc:  # never let alerting crash the run
                        logger.warning("Freeze alert notify failed: %s", exc)
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
                    self._report_order_failures(report, broker)
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
            self._report_order_failures(report, broker)
            return report

        # Process fills and failures
        if fills is not None and not fills.empty:
            n_skipped = 0
            n_skipped_sell = 0  # close-out SELLs the broker could not place (#81)
            for _, fill_row in fills.iterrows():
                # Strip whitespace before normalising: a broker that returns a
                # padded side/status (e.g. "SELL " / "SKIPPED ") must not slip
                # past the SKIPPED / close-out-SELL freeze counter (#81).
                status = str(fill_row.get("status", "FILLED")).strip().upper()
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
                notify = getattr(broker, "notify", None)
                if notify:
                    try:
                        notify(
                            "🧊 <b>REBALANCER FROZEN — trapped residual</b>\n"
                            f"0 fills; {n_skipped_sell} close-out SELL(s) below exchange "
                            f"minimum{failed_note}; residual exposure RETAINED.\n"
                            f"Skipped: {skipped_list}\n"
                            "Book NOT rebalanced — monitor for a trapped residual."
                        )
                    except Exception as exc:  # never let alerting crash the run
                        logger.warning("Freeze alert notify failed: %s", exc)
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
            notify = getattr(broker, "notify", None)
            if notify:
                notify(
                    f"⚠️ <b>0 FILLS for {len(broker_orders)} orders</b>\n"
                    f"Submitted: {order_list}\n"
                    "Check broker connectivity — portfolio NOT rebalanced."
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

        self._report_order_failures(report, broker)
        return report

    # ==================================================================
    # Reconciliation context + PRE-EXECUTION gate (issue #90)
    # ==================================================================
    # ==================================================================
    # Working orders left resting at the venue (cross-cycle resolution)
    # ==================================================================
    # Retention cap for the queue; rationale lives with the constant in
    # ``reconciliation.working_orders``.
    _WORKING_MAX_AGE_DAYS = DEFAULT_MAX_AGE_DAYS

    def _working_store(self, params: dict[str, Any]) -> Any:
        """Build the per-book working-order queue, or None when unidentifiable.

        Needs a ``book_key`` to namespace the queue — the same key the token
        policy store and the reconciliation ledger use. Without one there is no
        safe per-book path, so we return None and let the caller say so loudly
        rather than silently share one queue between books.
        """
        from quantbox.reconciliation.working_orders import WorkingOrderStore

        recon = params.get("reconciliation") or {}
        book_key = str(params.get("book_key") or recon.get("book_key") or "")
        if not book_key:
            logger.warning(
                "No book_key configured — orders left working at the venue cannot be "
                "queued for resolution, and a late fill will not be booked automatically"
            )
            return None
        root = str(recon.get("data_dir") or params.get("data_dir") or "data")
        return WorkingOrderStore(book_key=book_key, root=root)

    def _resolve_working_orders(self, store: Any, broker: BrokerPlugin, ledger: Any = None) -> list[dict[str, Any]]:
        """Stage 6c: ask the venue what became of orders a previous cycle left working.

        Thin delegate. The semantics — unreadable venue keeps the order queued,
        still-working keeps it, terminal books it against the carried order_ref,
        past the cap drops it loudly — live in
        :func:`quantbox.reconciliation.working_orders.resolve_working_orders`,
        which the out-of-cycle follow-up job calls too. Returns the orders that
        reached a terminal state this cycle.
        """
        from quantbox.reconciliation.working_orders import resolve_working_orders

        return resolve_working_orders(
            store,
            broker,
            ledger=ledger,
            max_age_days=self._WORKING_MAX_AGE_DAYS,
        ).resolved

    def _recon_load(self, params: dict[str, Any], run_id: str, asof: str) -> _ReconCtx | None:
        """Parse the `reconciliation` config block ONCE and build the per-cycle
        context (tolerances, ledger, persisted state, state machine).

        Returns ``None`` when no `reconciliation` block is present (no-op default).
        Raises on a bad config (bad tolerance key, unsafe book_key) — callers guard
        it so a recon fault is never fatal to the trading run.

        ENFORCE IS TOM-GATED. ``mode="enforce"`` is honored ONLY when the book
        config also carries ``enforce_acknowledged: true`` — an explicit, auditable
        opt-in that no casual config flip satisfies. Absent the acknowledgment,
        enforce is refused and forced back to observe (``enforce_refused=True``), so
        a book can never silently start gating live capital. Enabling enforce on a
        live book remains a deliberate action (this ack + a deploy), never automatic.
        """
        import json
        from pathlib import Path

        from quantbox.reconciliation import (
            BookTolerances,
            OrderFillLedger,
            ReconciliationStateMachine,
            ReconState,
        )

        cfg = params.get("reconciliation")
        if not cfg:
            return None

        book_key = str(cfg.get("book_key") or params.get("book_key") or "default")
        root = Path(str(cfg.get("data_dir", "data")))
        tol_cfg = dict(cfg.get("tolerances", {}))
        # `mode` may sit at the block top-level or inside tolerances; block wins.
        mode = str(cfg.get("mode", tol_cfg.pop("mode", "observe")))

        enforce_refused = False
        if mode == "enforce" and not bool(cfg.get("enforce_acknowledged", False)):
            enforce_refused = True
            logger.error(
                "RECON [%s]: mode='enforce' requested WITHOUT enforce_acknowledged=true — "
                "refusing to gate live capital on an unacknowledged config. Forcing OBSERVE. "
                "Enabling enforce is a deliberate, Tom-gated action.",
                book_key,
            )
            mode = "observe"

        tol = BookTolerances(mode=mode, **tol_cfg)
        # cycle_id namespaces order_refs in the ledger. A STATIC/reused cycle_id
        # conflates old and new intents/results (ledger matching is by order_ref),
        # which corrupts the failure/missed-fill signals the enforcement gate acts
        # on. Under enforce we therefore IGNORE any config-supplied cycle_id and
        # force the run-unique run_id (issue #90 review). Observe keeps the flexible
        # config override for testing/backfills.
        cfg_cycle = cfg.get("cycle_id")
        if tol.is_enforce:
            if cfg_cycle:
                logger.warning(
                    "RECON ENFORCE [%s]: ignoring config cycle_id %r — forcing run-unique run_id for live gating.",
                    book_key,
                    cfg_cycle,
                )
            if not run_id:
                raise ReconEnforcementError("enforce mode requires a run-unique run_id for cycle_id; none was provided")
            cycle_id = str(run_id)
        else:
            cycle_id = str(cfg_cycle or run_id or asof)
        ledger = OrderFillLedger(book_key=book_key, root=root)
        state_path = ledger.path.parent / "recon_state.json"

        persisted: dict[str, Any] = {}
        state_corrupt = False
        if state_path.exists():
            try:
                persisted = json.loads(state_path.read_text())
                if not isinstance(persisted, dict):
                    raise ValueError("recon state is not a JSON object")
            except Exception as exc:
                state_corrupt = True
                persisted = {}
                logger.error("Reconciliation state at %s is unreadable/corrupt: %s", state_path, exc)

        # FAIL CLOSED under enforce: the persisted state IS the enforcement
        # authority, so a corrupt/unreadable state must NOT silently reset to
        # NORMAL and let the next cycle trade freely. Force HALT so the book stops
        # until Tom clears it. In observe mode a corrupt state is only an
        # observability loss, so we fall back to a fresh NORMAL.
        if state_corrupt and tol.is_enforce:
            logger.error(
                "RECON ENFORCE [%s]: corrupt persisted state — failing CLOSED to HALT until manually cleared.",
                book_key,
            )
            persisted = {"state": "halt", "degraded_cycles": 0, "streaks": {}}

        machine = ReconciliationStateMachine(book_key=book_key, tol=tol)
        # Validate the persisted STATE VALUE (not just JSON well-formedness): a
        # readable file with a bogus state like {"state": "bogus"} is just as
        # dangerous as corrupt JSON — under enforce it must fail CLOSED to HALT,
        # never silently fall back to the default NORMAL and permit trading.
        raw_state = persisted.get("state", "normal")
        try:
            machine.state = ReconState(raw_state)
        except (ValueError, KeyError):
            if tol.is_enforce:
                logger.error(
                    "RECON ENFORCE [%s]: invalid persisted state value %r — failing CLOSED to HALT.",
                    book_key,
                    raw_state,
                )
                machine.state = ReconState.HALT
            else:
                logger.warning("Reconciliation state value %r invalid — resetting to NORMAL (observe).", raw_state)
                machine.state = ReconState.NORMAL
        try:
            machine.degraded_cycles = int(persisted.get("degraded_cycles", 0))
        except (TypeError, ValueError):
            machine.degraded_cycles = 0

        return _ReconCtx(
            book_key=book_key,
            root=root,
            tol=tol,
            cycle_id=cycle_id,
            mode=mode,
            enforce_refused=enforce_refused,
            ledger=ledger,
            machine=machine,
            state_path=state_path,
            persisted=persisted,
        )

    def _enforce_requested(self, params: dict[str, Any]) -> bool:
        """True iff the config asks for acknowledged enforce, WITHOUT loading the
        ledger. Used to decide fail-open vs fail-closed when `_recon_load` itself
        raised (so we cannot trust its parsed mode)."""
        cfg = params.get("reconciliation")
        if not isinstance(cfg, dict):
            return False
        tol_cfg = cfg.get("tolerances")
        tol_mode = tol_cfg.get("mode") if isinstance(tol_cfg, dict) else None
        mode = str(cfg.get("mode", tol_mode or "observe"))
        return mode == "enforce" and bool(cfg.get("enforce_acknowledged", False))

    def _recon_preflight(self, ctx: _ReconCtx, broker: BrokerPlugin | None) -> dict[str, Any]:
        """Compute the PRE-EXECUTION gate from the PERSISTED (prior-cycle) state.

        This is the real enforcement touchpoint (issue #90): the state machine
        persisted the outcome of the PREVIOUS cycle, and we read it back HERE,
        before this cycle's orders are constructed/executed. A HALT persisted last
        cycle blocks all orders now; a FLATTEN persisted last cycle makes this
        cycle reduce-only. Because the decision is durable state consumed strictly
        before execution, enforcing it is real safety, not the post-execution false
        safety #87 rejected.

        The gate is only APPLIED in enforce mode; observe mode computes it for the
        record (``applied=False``) and changes no orders.
        """
        from quantbox.reconciliation import preflight_gate

        orders_allowed, reduce_only = preflight_gate(ctx.machine.state)
        enforced = ctx.tol.is_enforce
        applied = bool(enforced and (not orders_allowed or reduce_only))
        note = {
            "gate_from_state": ctx.machine.state.value,
            "orders_allowed": orders_allowed,
            "reduce_only": reduce_only,
            "enforced": enforced,
            "applied": applied,
        }
        if applied:
            action = "HALT (no new orders)" if not orders_allowed else "FLATTEN (reduce-only)"
            msg = (
                f"🛑 RECON ENFORCE [{ctx.book_key}] PRE-EXECUTION gate from prior state "
                f"'{ctx.machine.state.value}': {action}. This cycle's orders are gated BEFORE execution."
            )
            logger.error(msg)
            notify = getattr(broker, "notify", None)
            if callable(notify):
                try:
                    notify(msg)
                except Exception as exc:  # never let alerting crash the run
                    logger.warning("Recon preflight notify failed: %s", exc)
        elif enforced:
            logger.info(
                "RECON ENFORCE [%s] preflight: prior state '%s' permits normal trading.",
                ctx.book_key,
                ctx.machine.state.value,
            )
        return note

    def _filter_reduce_only(self, executable: pd.DataFrame, broker: BrokerPlugin | None) -> pd.DataFrame:
        """Keep only orders that REDUCE existing exposure, clamped so they never
        flip a position past zero. Used by the FLATTEN gate.

        An order reduces |position| iff its signed delta opposes the current
        signed position. Opening/increasing legs are dropped; a reducing leg is
        clamped to at most the current position magnitude. Fails SAFE: if we cannot
        read positions, we drop everything (send nothing) rather than risk opening.
        """
        positions: dict[str, float] = {}
        get_positions = getattr(broker, "get_positions", None)
        if callable(get_positions):
            try:
                pos = get_positions()
                if pos is not None and len(pos) > 0:
                    for _, r in pos.iterrows():
                        positions[str(r.get("symbol", ""))] = float(r.get("qty", 0) or 0)
            except Exception as exc:  # fail safe: no positions → reduce nothing
                logger.warning("Reduce-only gate could not read positions (dropping all orders): %s", exc)
                return executable.iloc[0:0]
        else:
            logger.warning("Reduce-only gate: broker exposes no get_positions — dropping all orders (fail safe).")
            return executable.iloc[0:0]

        kept: list[Any] = []
        for _, row in executable.iterrows():
            sym = str(row.get("Asset", ""))
            action = str(row.get("Action", "")).strip().lower()
            qty = float(row.get("Adjusted Quantity", 0) or 0)
            cur = positions.get(sym, 0.0)
            if qty <= 0 or cur == 0:
                continue
            # FAIL CLOSED: only a recognised buy/sell can be judged reduce-or-not.
            # An unknown action under a FLATTEN gate must be DROPPED, never assumed
            # to be exposure-reducing (assuming sell could send an opening order).
            if action not in ("buy", "sell"):
                logger.warning(
                    "Reduce-only gate: dropping order with unrecognised action %r for %s (fail closed).",
                    action,
                    sym,
                )
                continue
            delta = qty if action == "buy" else -qty
            # Reduces only if the order's signed delta opposes the current position.
            if (delta > 0) == (cur > 0):
                continue
            new_qty = min(qty, abs(cur))  # clamp: never cross zero
            if new_qty <= 0:
                continue
            r = row.copy()
            r["Adjusted Quantity"] = new_qty
            # Exposure-reducing under a FLATTEN gate, i.e. an emergency exit. Mark
            # reduce_only so the broker sends reduceOnly: the venue exempts a
            # sub-min close from its min-notional floor and it can never flip
            # through zero. Without this the gate's own exits could be rejected as
            # ordinary sub-min orders (quantbox#87 / #138 review).
            r["reduce_only"] = True
            kept.append(r)
        return pd.DataFrame(kept) if kept else executable.iloc[0:0]

    # ==================================================================
    # Stage 7b helper: reconciliation ledger + break enforcement (#87, #90)
    # ==================================================================
    def _run_reconciliation(self, **kwargs: Any) -> dict[str, Any]:
        """Guarded wrapper: reconciliation must NEVER crash the trading run.

        This runs AFTER orders have already executed. A config typo, a bad
        tolerance key, a permission/disk-full error on the ledger write, or any
        other fault inside the reconciliation layer must not abort the run and
        lose the run's artifacts. On any failure we log, surface an `error` note,
        and return — observe mode changes no orders, so a recon crash is strictly
        an observability loss, never a trading fault.
        """
        try:
            return self._run_reconciliation_impl(**kwargs)
        except ReconEnforcementError:
            # CRITICAL under enforce (e.g. unwritable enforcement authority): must
            # NOT be swallowed. Let it escape and fail the run so the book stops,
            # rather than completing "successfully" while the next cycle trades from
            # stale/absent enforcement state. The impl already alerted hard.
            raise
        except Exception as exc:  # noqa: BLE001 - ordinary recon faults are never fatal
            logger.error("Reconciliation stage failed (non-fatal to the run): %s", exc)
            return {"error": str(exc)}

    def _run_reconciliation_impl(
        self,
        *,
        params: dict[str, Any],
        broker: BrokerPlugin | None,
        final_weights: dict[str, float],
        orders_df: pd.DataFrame,
        execution_report: dict[str, Any],
        portfolio_value: float,
        stable_coin: str,
        asof: str,
        run_id: str,
        intent_captured: bool = False,
    ) -> dict[str, Any]:
        """Evaluate the break state machine against the order/fill ledger.

        Observe by default (default mode); ``enforce`` is Tom-gated (see
        :meth:`_recon_load`). This stage classifies breaks and computes the
        NORMAL→DEGRADED→HALT/FLATTEN transition, logs/alerts, and PERSISTS the
        resulting state. The persisted state is what the NEXT cycle's
        pre-execution gate (:meth:`_recon_preflight`) reads back — that
        persist-then-read is the real enforcement path (issue #90). No-op unless a
        `reconciliation` config block is present.

        Authority rule: exchange = truth for holdings (drift/phantom are computed
        against the broker's real positions); the ledger = the intent record
        (surfaces a submitted order that produced no fill).

        LEDGER SOURCE (issue #90): when the execution stage captured intent at the
        broker submission call (the crash-durable path — `ledger` was threaded into
        ``_execute_orders``), this stage READS those records back rather than
        reconstructing them, so the audit trail reflects the exact submission path.
        When intent was NOT pre-captured (e.g. a direct unit-test call, or a run
        where execution recorded nothing), it falls back to reconstructing intent +
        result from ``orders_df`` / ``execution_report`` so observe-mode shadow
        detection still works.
        """
        from collections import defaultdict

        from quantbox.reconciliation import KIND_INTENT, classify_breaks

        ctx = self._recon_load(params, run_id, asof)
        if ctx is None:
            return {}

        book_key = ctx.book_key
        tol = ctx.tol
        cycle_id = ctx.cycle_id
        ledger = ctx.ledger
        machine = ctx.machine
        state_path = ctx.state_path
        persisted = ctx.persisted
        enforce_refused = ctx.enforce_refused

        # Failed/zero-fill counting for the consecutive-failure break class.
        # `attempted_syms` = every symbol we submitted an intent for THIS cycle;
        # only these keep/accrue a streak, so a symbol that is simply no longer
        # traded cannot leave a stale streak that falsely escalates to HALT.
        this_cycle_failed: dict[str, int] = {}
        filled_syms: set[str] = set()
        attempted_syms: set[str] = set()
        missed_fills: list[str] = []

        # Did the execution stage already capture intent at submission for THIS
        # cycle? The caller (run()) says so explicitly via `intent_captured` — we do
        # NOT infer it from the presence of same-cycle_id intents, because a bare
        # cycle_id can be reused across cycles. When capture was active we ALWAYS
        # read the ledger back (never reconstruct), so a HALTED cycle that submitted
        # nothing correctly reads as zero attempts rather than re-recording orders
        # that were never sent.
        already_captured = bool(intent_captured)

        if already_captured:
            # --- Read the submission-time-captured ledger (authoritative) ---
            matched = ledger.match_intents_to_results()
            for _ref, slot in matched.items():
                intent = slot.get("intent")
                if intent is None or str(intent.get("cycle_id")) != cycle_id:
                    continue
                sym = str(intent.get("symbol", ""))
                attempted_syms.add(sym)
                result = slot.get("result")
                status = str(result.get("status")).lower() if result else None
                if status in ("filled", "partial"):
                    filled_syms.add(sym)
                elif status == "working":
                    # Accepted and still live on the book. Not a fill (nothing
                    # executed yet) and NOT a missed fill (nothing went wrong) --
                    # so it must touch neither `filled_syms` nor the failure
                    # streak. Stage 6c resolves it against the venue next cycle
                    # and records the terminal result against this same ref.
                    continue
                elif status in ("timeout", None):
                    # Submitted but no (real) result observed = missed fill.
                    this_cycle_failed[sym] = this_cycle_failed.get(sym, 0) + 1
                    missed_fills.append(sym)
                else:  # failed / skipped / rejected / anything non-fill
                    this_cycle_failed[sym] = this_cycle_failed.get(sym, 0) + 1
        elif execution_report.get("recon_gated"):
            # --- Cycle was GATED pre-execution: submit-set is NOT orders_df -----
            # The pre-execution gate blocked orders (HALT) or filtered them
            # (FLATTEN), so the broker never received orders_df. Reconstructing
            # intents/results from orders_df here would fabricate ledger records for
            # orders that were NEVER submitted — corrupting the audit trail and
            # escalating failure/missed-fill streaks off a SAFETY GATE rather than
            # real broker behavior. Record NOTHING; leave attempted_syms empty so no
            # streak accrues. (This is the enforce fail-closed path where run() sets
            # recon_ctx=None → intent_captured=False, so `already_captured` is False
            # but nothing was actually sent.)
            logger.info(
                "RECON [%s]: cycle gated pre-execution (%s) — no orders submitted; "
                "skipping ledger reconstruction (no synthetic intents/results).",
                book_key,
                execution_report.get("recon_gated"),
            )
        else:
            # --- Reconstruction fallback: record intent + result here -------
            details = execution_report.get("orders_details", [])
            executable = (
                orders_df[orders_df.get("Executable", pd.Series(dtype=bool))]
                if isinstance(orders_df, pd.DataFrame) and "Executable" in orders_df.columns
                else orders_df
            )
            # Match results to intents by (symbol, side) CONSUMING each result once,
            # so N same-(symbol, side) orders in a cycle can't all bind to the first
            # result (duplicating a fill or masking a missed one).
            details_pool: dict[tuple[str, str], list[dict]] = defaultdict(list)
            for d in details:
                k = (str(d.get("symbol", "")), str(d.get("side", d.get("action", ""))).lower())
                details_pool[k].append(d)

            if isinstance(executable, pd.DataFrame) and not executable.empty:
                for i, (_, row) in enumerate(executable.iterrows()):
                    sym = str(row.get("Asset", ""))
                    side = str(row.get("Action", "")).lower()
                    attempted_syms.add(sym)
                    order_ref = f"{cycle_id}:{i}:{sym}:{side}"
                    ledger.record_intent(
                        cycle_id=cycle_id,
                        symbol=sym,
                        side=side,
                        order_ref=order_ref,
                        target_qty=_safe_float(row.get("Adjusted Quantity")),
                        target_wt=final_weights.get(sym),
                        limit_px=_safe_float(row.get("Price")),
                    )
                    pool = details_pool.get((sym, side))
                    match = pool.pop(0) if pool else None
                    if match is not None:
                        status = str(match.get("status", "failed")).upper()
                        ledger.record_result(
                            order_ref=order_ref,
                            cycle_id=cycle_id,
                            status=status.lower(),
                            filled_qty=_safe_float(match.get("executed_quantity")),
                            avg_px=_safe_float(match.get("executed_price")),
                        )
                        if status in ("FILLED", "PARTIAL"):
                            filled_syms.add(sym)
                        elif status == "WORKING":
                            # Accepted, still live on the book. Neither a fill nor
                            # a failure — the same rule as the captured-intent path
                            # above. Without this the reconstruction path would
                            # count every resting limit order as a failed order.
                            pass
                        else:  # FAILED / SKIPPED / anything non-fill
                            this_cycle_failed[sym] = this_cycle_failed.get(sym, 0) + 1
                    else:
                        # Intent written but NO result observed = missed fill.
                        ledger.record_result(order_ref=order_ref, cycle_id=cycle_id, status="timeout")
                        this_cycle_failed[sym] = this_cycle_failed.get(sym, 0) + 1
                        missed_fills.append(sym)

        # --- Reconcile against external truth (holdings) -------------------
        actual_wt: dict[str, float] = {}
        phantom: list[str] = []
        get_positions = getattr(broker, "get_positions", None)
        pv = float(portfolio_value) if portfolio_value else 0.0
        if callable(get_positions) and pv > 0:
            try:
                pos = get_positions()
                if pos is not None and len(pos) > 0 and hasattr(broker, "get_market_snapshot"):
                    snap = broker.get_market_snapshot(pos["symbol"].tolist())
                    if snap is not None and "mid" in snap.columns:
                        merged = pos.merge(snap[["symbol", "mid"]], on="symbol", how="left")
                        for _, r in merged.iterrows():
                            sym = str(r.get("symbol", ""))
                            val = float(r.get("qty", 0) or 0) * float(r.get("mid", 0) or 0)
                            actual_wt[sym] = val / pv
            except Exception as exc:  # never let recon crash the run
                logger.warning("Reconciliation position read failed: %s", exc)

        # Drift must be FRACTIONAL (|actual - target| / |target|), because that is
        # what BookTolerances documents and what `max_drift=0.10` / `drift_halt=0.25`
        # mean. The previous absolute weight delta made both tolerances wrong in
        # opposite directions: on a book whose largest target is ~5% of equity an
        # absolute delta can never reach 0.25 (the halt was dead), while a 30%
        # holding against a zero target tripped FLATTEN instantly.
        #
        # `drift_notional_floor` keeps dust out of the signal: a $2 residual against
        # a zero target is fractionally infinite but economically irrelevant, and a
        # book that halts on dust is a book whose halt gets switched off.
        drift_floor = float(getattr(tol, "drift_notional_floor", 10.0))
        drifts: dict[str, float] = {}
        for sym in set(final_weights) | set(actual_wt):
            if sym == stable_coin:
                continue
            actual = actual_wt.get(sym, 0.0)
            target = float(final_weights.get(sym, 0.0))
            # Ignore positions too small to matter (both legs sub-floor in notional).
            if pv > 0 and max(abs(actual), abs(target)) * pv < drift_floor:
                continue
            denom = abs(target)
            if denom < 1e-9:
                # No target at all: any real holding is a 100% drift (fully
                # un-intended exposure), which is the honest fractional reading.
                drifts[sym] = 1.0 if abs(actual) > 1e-9 else 0.0
            else:
                drifts[sym] = (actual - target) / denom

        # Phantom = an exchange holding with no matching intent in the ledger over
        # a RECENT window (authority rule: internal ledger is truth for intent).
        # Scoping matters both ways: using final_weights (round-1 logic) reflags
        # every legitimate carryover/residual as phantom; using ALL history
        # (round-2 logic) means a name traded once can NEVER be phantom again,
        # masking a genuine re-appearing phantom on a rotating book. So we count a
        # holding as explained only if it was intended within the last
        # `phantom_lookback` cycles (default 2) — recent carryover is explained, a
        # stale reappearance is not. The current cycle's intents were just written.
        recs = ledger.read_all()
        cycle_order = sorted(
            {r.get("cycle_id") for r in recs if r.get("cycle_id")},
            key=lambda c: max((r.get("ts", "") for r in recs if r.get("cycle_id") == c), default=""),
        )
        recent_cycles = set(cycle_order[-max(1, tol.phantom_lookback) :])
        intended_symbols = {
            r.get("symbol") for r in recs if r.get("kind") == KIND_INTENT and r.get("cycle_id") in recent_cycles
        }
        for sym in actual_wt:
            if sym == stable_coin:
                continue
            # Same dust guard as drift: a sub-floor residual is not a phantom worth
            # flattening the book over. Every phantom is a HARD break, so without
            # this one stray $0.30 dust position sends the book to FLATTEN.
            if pv > 0 and abs(actual_wt.get(sym, 0.0)) * pv < drift_floor:
                continue
            if sym not in intended_symbols and abs(actual_wt.get(sym, 0.0)) > 1e-6:
                phantom.append(sym)

        # --- Persistent per-book state (streaks + machine state) -----------
        # `persisted` / `state_path` / `machine` (with prior state loaded) all come
        # from the shared _ReconCtx so the preflight gate and this evaluator agree
        # on the same prior state. Only carry a streak forward for a symbol we
        # ATTEMPTED again this cycle; a symbol we stopped trading has, by
        # definition, no *consecutive* failure to escalate on (prevents a stale
        # streak from falsely reaching HALT).
        prior_streaks: dict[str, int] = dict(persisted.get("streaks", {}))
        streaks: dict[str, int] = {s: n for s, n in prior_streaks.items() if s in attempted_syms}
        for sym in filled_syms:
            streaks.pop(sym, None)  # a fill breaks the failure streak
        for sym, n in this_cycle_failed.items():
            streaks[sym] = streaks.get(sym, 0) + n

        breaks = classify_breaks(
            tol=tol,
            drifts=drifts,
            failed_streaks=streaks,
            phantom_symbols=phantom,
        )

        decision = machine.evaluate(breaks)

        import json

        state_payload = json.dumps(
            {
                "state": machine.state.value,
                "degraded_cycles": machine.degraded_cycles,
                "streaks": streaks,
            }
        )
        # Atomic write (temp + fsync + os.replace): the state file is the
        # enforcement authority, so a crash mid-persist must never leave it
        # truncated/corrupt. Retry once on failure.
        persist_err: Exception | None = None
        for _attempt in range(2):
            try:
                _atomic_write_text(state_path, state_payload)
                persist_err = None
                break
            except Exception as exc:  # noqa: BLE001
                persist_err = exc
        if persist_err is not None:
            if tol.is_enforce:
                # ENFORCE: the next cycle's gate reads this file. If we cannot
                # persist a (possibly more severe) state, the enforcement authority
                # is unwritable — do NOT continue silently. Alert hard and RAISE so
                # the run surfaces as failed and the operator halts the book; a
                # silent continue would let the next cycle trade from stale state.
                logger.error(
                    "RECON ENFORCE [%s]: could not persist recon state (%s) — enforcement "
                    "authority unwritable. Raising to fail the run.",
                    book_key,
                    persist_err,
                )
                notify = getattr(broker, "notify", None)
                if callable(notify):
                    with contextlib.suppress(Exception):
                        notify(
                            f"🛑 <b>RECON ENFORCE [{book_key}] — state persist FAILED</b>\n"
                            f"Could not write recon_state.json ({persist_err}). The next cycle's "
                            "enforcement gate cannot be trusted — HALT the book and investigate disk."
                        )
                raise ReconEnforcementError(f"recon state persist failed under enforce: {persist_err}") from persist_err
            logger.warning("Reconciliation state persist failed (observe): %s", persist_err)

        # --- Alert (observe = log/alert only, no gating) -------------------
        if decision.alert:
            log_fn = logger.error if decision.to_state.value in ("halt", "flatten") else logger.warning
            log_fn("RECON [%s] %s", book_key, decision.alert.replace("\n", " | "))
            notify = getattr(broker, "notify", None)
            if callable(notify) and decision.transitioned:
                try:
                    notify(decision.alert)
                except Exception as exc:  # never let alerting crash the run
                    logger.warning("Reconciliation alert notify failed: %s", exc)

        return {
            "book_key": book_key,
            "mode": decision.mode,
            "enforced": decision.enforced,
            # True iff a config asked for enforce WITHOUT enforce_acknowledged=true
            # and we forced it back to observe. Surfaced so a monitor can flag a
            # misconfigured book that believes it is gating when it is not.
            "enforce_refused": enforce_refused,
            # Whether the ledger for this cycle was captured at submission time
            # (crash-durable) vs reconstructed post-execution (fallback).
            "intent_captured_at_submission": already_captured,
            "from_state": decision.from_state.value,
            "to_state": decision.to_state.value,
            "would_be_action": decision.would_be_action.value,
            "transitioned": decision.transitioned,
            "orders_allowed": decision.orders_allowed,
            "reduce_only": decision.reduce_only,
            "n_breaks": len(breaks),
            "breaks": [
                {
                    "class": b.klass.value,
                    "symbol": b.symbol,
                    "severity": b.severity,
                    "detail": b.detail,
                    "recommended_action": b.recommended_action.value,
                }
                for b in breaks
            ],
            "missed_fills": missed_fills,
            "ledger_path": str(ledger.path),
        }

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
