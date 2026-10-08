"""Backtesting pipeline plugin.

Uses the same config format as :class:`TradingPipeline` but replaces
broker execution with a historical simulation through the engine seam
(:mod:`quantbox.engine`, docs/adr/0008): vectorbt or rsims, chosen by
``engine:`` alone.

Workflow
-------
1. Load universe & market data (same as TradingPipeline)
2. Run strategies → full weights time series
3. Aggregate across strategies (same logic), then the overlay chain
   (``plugins.overlays``, ADR-0004) modifies the decided book in config order
4. The decision (:func:`quantbox.decision.final_targets`, the transform live
   trading calls too, TOM-1520): the short clip (``venue.allow_shorts``), the
   gross cap (``risk.max_leverage``), the group limits, then ``venue.leverage``
   normalisation — the FINAL target weights (``risk.tranches`` is the seam's
   tranche cadence, TOM-1513)
5-6. Hand the target weights to the engine seam
   (:func:`quantbox.engine.simulate`): calendars, the rebalancing schedule,
   the execution lag (``execution.lag_bars``, default 1 = next-bar), the cash
   cap, financing legs, then the engine adapter — the ONE place target weights
   become traded weights
7. Compute performance + traded-book metrics
8. Save artifacts (weights_history = decided targets, traded_weights = what
   the engine received, returns, metrics, portfolio_daily)

Usage
-----
Swap ``pipeline.name`` from ``trade.full_pipeline.v1`` to
``backtest.pipeline.v1`` in your YAML config — everything else stays
the same::

    run:
      mode: backtest
      asof: "2026-02-01"
      pipeline: "backtest.pipeline.v1"

    plugins:
      pipeline:
        name: "backtest.pipeline.v1"
        params:
          engine: vectorbt          # or "rsims"
          fees: 0.001
          rebalancing_freq: 1       # daily
          # ... same strategy / universe / prices params
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from quantbox.decision import (
    DecisionRules,
    decision_metrics,
    final_targets,
    gross_cap,
    log_normalisation,
    with_decision,
)
from quantbox.engine import (
    DEFAULT_ENGINE,
    NAN_POLICY,
    Costs,
    TradedBook,
    engine_names,
    get_engine,
    materialise_nan,
    simulate,
)
from quantbox.engine.groups import GROUP_LIMITS_SCHEMA, GroupLimits, resolve_group_limits
from quantbox.engine.policy import (
    POLICY_SCHEMA,
    RebalancePolicy,
    legacy_policy,
    resolve_policy,
    tranches_alias,
)
from quantbox.exceptions import DataLoadError
from quantbox.execution import (
    EXECUTION_SCHEMA,
    VENUE_SCHEMA,
    check_schedule_venue,
    exposure_metrics,
    resolve_allow_shorts,
    resolve_execution,
    timing_record,
    warn_on_shorts,
)
from quantbox.financing import ASSUMED_FREE, resolve_financing, resolve_leverage
from quantbox.frequency import Frequency, resolve_pipeline_frequency
from quantbox.funding_guard import FUNDING_SCHEMA, check_funding, ignored_record, planned_market, resolve_funding
from quantbox.instrument_calendar import calendar_summary
from quantbox.overlays import OverlayLink, apply_overlays
from quantbox.plugins.datasources._utils import interval_step, normalize_data_frequency
from quantbox.strategy_runner import build_strategy_context, run_strategies

logger = logging.getLogger(__name__)


def _variant_risk_cfg(base_risk_cfg: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    """A variant's risk config: the run's ``risk`` with ``overrides.risk`` on top."""
    return {**base_risk_cfg, **((variant.get("overrides") or {}).get("risk") or {})}


_LEGACY_SCHEDULE_KEYS = ("rebalancing_freq", "threshold")


def _run_policy(params: dict[str, Any]) -> RebalancePolicy:
    """The run's rebalancing policy: ``rebalancing_policy``, or the legacy ``rebalancing_freq`` / ``threshold``.

    Declaring both is refused: one schedule, stated once (docs/adr/0008).
    """
    declared = params.get("rebalancing_policy")
    legacy = [k for k in _LEGACY_SCHEDULE_KEYS if k in params]
    if declared is not None and legacy:
        raise ValueError(
            f"declare the schedule ONCE: rebalancing_policy or {legacy}, not both "
            "(rebalancing_policy.frequency and band replace rebalancing_freq and threshold)"
        )
    if declared is not None:
        return resolve_policy(declared)
    return legacy_policy(params.get("rebalancing_freq", 1), params.get("threshold"))


def _variant_policy(
    vname: str, overrides: dict[str, Any], run_params: dict[str, Any], run: RebalancePolicy
) -> RebalancePolicy:
    """A variant's policy: ``overrides.rebalancing_policy``, the legacy override keys, or the run's."""
    declared = overrides.get("rebalancing_policy")
    legacy = [k for k in _LEGACY_SCHEDULE_KEYS if k in overrides]
    if declared is not None and legacy:
        raise ValueError(f"Variant {vname!r}: overrides declare rebalancing_policy and {legacy}; declare one")
    if declared is not None:
        try:
            return resolve_policy(declared)
        except ValueError as exc:
            raise ValueError(f"Variant {vname!r}: overrides.{exc}") from exc
    if legacy:
        if run.declared:
            raise ValueError(
                f"Variant {vname!r}: the run declares rebalancing_policy; override it with "
                f"overrides.rebalancing_policy, not {legacy}"
            )
        return legacy_policy(
            overrides.get("rebalancing_freq", run_params.get("rebalancing_freq", 1)),
            overrides.get("threshold", run_params.get("threshold")),
        )
    return run


def _risk_tranches(policy: RebalancePolicy, risk_cfg: dict[str, Any], where: str = "") -> RebalancePolicy:
    """``risk.tranches: N`` (N > 1): the DEPRECATED spelling of the seam's tranche cadence (TOM-1513).

    One tranche concept: the alias sets the policy's cadence to ``tranche``
    with N tranches and keeps its frequency and trigger, so it books exactly
    what ``rebalancing_policy: {cadence: tranche, tranches: N}`` books. It is
    no longer a rolling mean of N bars before the seam; the two agree on a
    daily schedule after the first N-1 decisions (tests/test_rebalancing_policies.py
    proves it). Declaring tranches twice is refused. The one implementation is
    :func:`quantbox.engine.policy.tranches_alias`, which live trading calls too
    (TOM-1518).
    """
    return tranches_alias(policy, risk_cfg.get("tranches", 1), key=f"{where}risk.tranches")


def _number(key: str, value: Any, *, cast: type = float, where: str = "") -> Any:
    """``cast(value)``, refused with the param's *key* when it is not a number.

    Callers pass ``params.get(key, default)`` literally, so the params-schema
    check (tests/test_params_schema.py) still sees which keys are read.
    """
    try:
        return cast(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}'{key}' must be a number, got {value!r}") from exc


def _plan_variants(
    variants: list[dict[str, Any]], venue_declared: bool, costs: dict[str, float]
) -> dict[str, dict[str, float]]:
    """Refuse a variants block the run could not honour; the per-variant costs it will use.

    Every check here once ran only inside the variants flow, after the data
    was loaded, so ``quantbox config explain`` said ok for configs the run
    then refused (TOM-1362).
    """
    out: dict[str, dict[str, float]] = {}
    for v in variants:
        vname = str(v.get("name"))
        ov = dict(v.get("overrides", {}) or {})
        # Execution timing and venue are properties of the RUN, not of a
        # variant: variants that differ in timing are not comparable, and
        # a silently ignored override is worse than a refusal.
        for run_level in ("execution", "venue"):
            if run_level in ov or run_level in v:
                raise ValueError(
                    f"Variant {vname!r}: '{run_level}' is run-level — declare it once in the pipeline params"
                )
        if "allow_short" in (ov.get("risk") or {}) and venue_declared:
            raise ValueError(
                f"Variant {vname!r}: overrides.risk.allow_short conflicts with the run-level `venue` block"
            )
        out[vname] = {
            key: _number(key, ov.get(key, costs[key]), where=f"Variant {vname!r}: overrides.")
            for key in ("fees", "fixed_fees", "slippage")
        }
    return out


def _vbt_portfolio(book: TradedBook) -> Any:
    """The native object the HTML report plots (``pf.plot()``), when the engine's native is a vbt.Portfolio."""
    return book.native if book.native_key == "vbt_portfolio" else None


@dataclass
class BacktestPipeline:
    meta = PluginMeta(
        name="backtest.pipeline.v1",
        kind="pipeline",
        version="0.2.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Backtesting pipeline: same config as TradingPipeline but "
            "routes weights through the engine seam (vectorbt or rsims) "
            "instead of broker execution."
        ),
        tags=("backtesting", "research", "crypto"),
        capabilities=("backtest",),
        schema_version="v1",
        params_schema={
            "type": "object",
            "properties": {
                "engine": {
                    "type": "string",
                    "enum": engine_names(),
                    "default": DEFAULT_ENGINE,
                    "description": (
                        "Engine adapter behind the seam (quantbox.engine, docs/adr/0008). Changing it alone "
                        "changes nothing else in the run: single run and variants both run on either. "
                        "Which engine: docs/adr/0008 decision 14 (perps, funding or margin -> rsims). "
                        "Funding data on an engine that does not charge funding is refused (see funding)."
                    ),
                },
                "fees": {
                    "type": "number",
                    "minimum": 0,
                    "default": 0.001,
                    "description": "Proportional fee rate (e.g. 0.001 = 10 bps).",
                },
                "fixed_fees": {
                    "type": "number",
                    "minimum": 0,
                    "default": 0.0,
                    "description": "Fixed fee per order, in quote currency, on every engine (TOM-1500).",
                },
                "slippage": {
                    "type": "number",
                    "minimum": 0,
                    "default": 0.0,
                    "description": "Proportional slippage applied to fills (e.g. 0.0005 = 5 bps).",
                },
                "rebalancing_freq": {
                    "type": ["integer", "string", "array", "null"],
                    "description": (
                        "The legacy spelling of rebalancing_policy {cadence: periodic, frequency: ...} "
                        "(docs/adr/0008 decision 11); new configs declare rebalancing_policy, and "
                        "declaring both is refused. "
                        "How often portfolio is rebalanced to target weights. Accepts: "
                        "int (every N bars; e.g. 5 = weekly on daily data, every 5 hours on hourly), "
                        "string pandas-offset (1D/1W/1M/1Y), explicit list of dates, "
                        "or null for buy-and-hold (rebalance only on the first bar). "
                        "BETWEEN rebalances the engine holds positions and lets them DRIFT "
                        "with returns (buy-and-hold semantics, matching `bt.vectorized.backtest`); "
                        "it does NOT re-equalise to target every bar. Daily-target reset "
                        "behaviour is not currently supported — use rebalancing_freq=1 (daily) "
                        "if you need it explicitly. WARNING: a daily-forward-fill-of-target "
                        "approximation OVERSTATES returns by ~3-4 pp/decade vs buy-and-hold "
                        "for sticky-weight strategies. The engine here is correct; this note "
                        "is for anyone writing custom validation simulators outside the pipeline."
                    ),
                    "default": 1,
                },
                "threshold": {
                    "type": ["number", "null"],
                    "default": None,
                    "description": (
                        "Rebalancing band (absolute weight), on every engine: a scheduled rebalance is placed "
                        "only when an instrument's held weight has drifted more than this from its target. The "
                        "seam computes the drift cost-free (docs/adr/0008); a run with costs can trigger on "
                        "slightly different bars than an in-engine band would. The legacy spelling of "
                        "rebalancing_policy {policy: band}; refused together with rebalancing_policy."
                    ),
                },
                "rebalancing_policy": POLICY_SCHEMA,
                "group_limits": GROUP_LIMITS_SCHEMA,
                "initial_cash": {
                    "type": "number",
                    "minimum": 0,
                    "default": 10000,
                    "description": (
                        "Starting cash, the same on every engine (TOM-1500): a fixed fee is a share of it."
                    ),
                },
                "margin": {
                    "type": "number",
                    "minimum": 0,
                    "default": 0.0,
                    "description": (
                        "Maintenance-margin rate for rsims exchange-liquidation modelling. "
                        "DEFAULT 0.0 (off): the legacy 0.05 maintenance/deleverage path is "
                        "unreliable for leveraged long/short books — it spuriously liquidates as "
                        "gross grows and breaks vol-invariance (target_vol 0.50 collapsed +0.51 "
                        "Sharpe to -0.64). Opt in only to study exchange liquidation explicitly."
                    ),
                },
                "trade_buffer": {
                    "type": "number",
                    "minimum": 0,
                    "default": 0.0,
                    "description": "No-trade buffer half-width (rsims only).",
                },
                "capitalise_profits": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "Compound profits into sizing (rsims only). DEFAULT true, as vectorbt always "
                        "compounds (TOM-1500); false sizes every bar off min(initial_cash, equity)."
                    ),
                },
                "equity_basis": {
                    "type": "string",
                    "enum": ["rsims", "mtm"],
                    "default": "rsims",
                    "description": (
                        "Equity for sizing. 'rsims' = cash + maint_margin; with margin=0 this "
                        "reduces to cash (= true equity for perps, where cash accumulates MTM "
                        "PnL) and is stable. 'mtm' = cash + sum(signed position_value) — double "
                        "-counts notional for perps and destabilises sizing on leveraged books; "
                        "use only for cash-instrument backtests."
                    ),
                },
                "trading_days": {
                    "type": "integer",
                    "default": 365,
                    "description": "Annualization factor for metrics.",
                },
                "strategies": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Strategy configs (same as TradingPipeline).",
                },
                "universe": {
                    "type": "object",
                    "default": {},
                    "description": "Universe selection params, passed to the data plugin's load_universe().",
                },
                "prices": {
                    "type": "object",
                    "default": {"lookback_days": 365},
                    "description": (
                        "Market-data request passed to the data plugin's load_market_data() "
                        "(lookback_days, frequency, symbols, ...). `mode` is set from the run."
                    ),
                },
                "frequency": {
                    "type": ["string", "object"],
                    "description": (
                        "Bar frequency: '1h' or {bar_size, calendar}. Wins over prices.frequency + "
                        "market_calendar; its bars_per_year is the default trading_days and strategy annualize."
                    ),
                },
                "market_calendar": {
                    "type": "string",
                    "default": "24/7",
                    "description": "Calendar used with prices.frequency when `frequency` is absent (e.g. NYSE).",
                },
                "risk": {
                    "type": "object",
                    "default": {},
                    "description": (
                        "Risk transforms applied to the weights time series and handed to risk plugins "
                        "(allow_short, max_leverage, ...). max_leverage is the decision's gross cap, default 1 "
                        "as in trading (TOM-1525): a levered book declares it. tranches: N is DEPRECATED "
                        "(TOM-1513): it is the seam's tranche cadence, rebalancing_policy {cadence: tranche, "
                        "tranches: N}, and warns."
                    ),
                    "properties": {
                        "max_leverage": {
                            "type": "number",
                            "minimum": 0,
                            "default": 1,
                            "description": "The decision's gross cap (sum |w| per row); default 1, as in trading.",
                        },
                    },
                },
                "strategy_weights": {
                    "type": "object",
                    "default": {},
                    "description": "Per-strategy weight overrides by strategy name, used when aggregating.",
                },
                "variants": {
                    "type": "array",
                    "items": {"type": "object"},
                    "default": [],
                    "description": (
                        "Independent variants overlaid in one report; each has name, strategy {name, params, "
                        "params_init} and optional overrides (fees, fixed_fees, slippage, rebalancing_freq, "
                        "threshold, rebalancing_policy, risk)."
                    ),
                },
                "narrative": {
                    "type": "object",
                    "description": (
                        "Report narrative: title, methodology, findings inline, or title_file / "
                        "methodology_file / findings_file paths."
                    ),
                },
                "execution": {
                    **EXECUTION_SCHEMA,
                    "description": "Run-level execution timing (lag_bars; same_bar = the explicit lag 0 override).",
                },
                "venue": {
                    **VENUE_SCHEMA,
                    "description": "Run-level venue constraints (allow_shorts).",
                },
                "funding": FUNDING_SCHEMA,
                "full_report": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Also write the heavy HTML research report (report.html + report_data.json, "
                        "tens of MB on a typical line). Off by default: every backtest run writes the "
                        "slim finding_report.json (qute-research/finding-report@1) instead."
                    ),
                },
            },
        },
        inputs=(),
        outputs=(
            "strategy_weights",
            "aggregated_weights",
            "weights_history",
            "traded_weights",
            "portfolio_daily",
            "returns",
            "metrics",
        ),
        examples=(
            "run:\n  mode: backtest\n  asof: '2026-02-01'\n  pipeline: backtest.pipeline.v1\n"
            "plugins:\n  pipeline:\n    name: backtest.pipeline.v1\n    params:\n"
            "      engine: vectorbt\n      fees: 0.001\n"
            "      rebalancing_policy: {cadence: periodic, frequency: weekly}\n"
            "      strategies:\n        - name: crypto_trend\n          weight: 1.0\n",
        ),
    )
    kind = "research"
    # The runner refuses ``plugins.overlays`` for a pipeline that does not say it applies them.
    accepts_overlays = True

    # ==================================================================
    # Plan: what run() will do with these params, decided before any data
    # ==================================================================
    def plan(self, params: dict[str, Any]) -> dict[str, Any]:
        """Everything :meth:`run` decides from *params* alone, refused here if it would refuse — no data read.

        :meth:`run` takes engine, timing, venue, frequency and costs from here and
        records ``execution`` / ``venue`` in its notes verbatim; ``quantbox config
        explain`` calls this same method, so a config the run would refuse on its
        params is refused BEFORE any data is loaded, by one check shared by both
        (TOM-1362). Raises ``ValueError`` on an unknown engine, a malformed
        ``execution`` / ``venue`` / ``frequency``, a non-numeric cost or engine
        parameter, or a ``variants`` block the run cannot honour;
        ``MissingExtraError`` when the engine's extra is not installed.
        """
        adapter = get_engine(params.get("engine", DEFAULT_ENGINE))
        engine = adapter.name
        # A declared funding: {ignore, reason} (TOM-1609); check_funding decides, on the data, whether it is honoured.
        funding_ignore = resolve_funding(params.get("funding"))
        timing = resolve_execution(params.get("execution"))
        allow_shorts, venue_declared = resolve_allow_shorts(params.get("venue"), params.get("risk"))
        check_schedule_venue(timing, params.get("venue"))
        financing = resolve_financing((params.get("venue") or {}).get("financing"))
        # schedule: bars applies no venue.leverage (recorded as "none"); the calendar's default is one value.
        leverage = (
            "none" if timing.schedule == "bars" else resolve_leverage((params.get("venue") or {}).get("leverage"))
        )
        if financing is None and leverage == "borrow":
            # Borrowing without a declared price: free, ASSUMED, and recorded as such.
            financing = ASSUMED_FREE
        load_params = dict(params.get("prices", {"lookback_days": 365}))
        freq = self._resolve_frequency(params, load_params)
        bars_per_year = freq.bars_per_year()
        costs = {
            "fees": _number("fees", params.get("fees", 0.001)),
            "fixed_fees": _number("fixed_fees", params.get("fixed_fees", 0.0)),
            "slippage": _number("slippage", params.get("slippage", 0.0)),
        }
        trading_days = _number("trading_days", params.get("trading_days", round(bars_per_year)), cast=int)
        # The adapter reads its own params (rsims: trade_buffer, initial_cash, margin, ...).
        engine_params = adapter.plan_params(params)
        full_report = params.get("full_report", False)
        if not isinstance(full_report, bool):  # a truthy "no" must not write tens of MB
            raise ValueError(f"'full_report' must be true or false, got {full_report!r}")
        variants = params.get("variants") or []
        variant_costs = _plan_variants(variants, venue_declared, costs)
        # The rebalancing policy (the seam's schedule) and the group limits, refused here when malformed.
        # risk.tranches (deprecated) is the tranche cadence: the run's, and each variant's own (TOM-1513).
        declared_policy = _run_policy(params)
        policy = _risk_tranches(declared_policy, params.get("risk") or {})
        variant_policies = {
            str(v.get("name")): _risk_tranches(
                _variant_policy(str(v.get("name")), dict(v.get("overrides") or {}), params, declared_policy),
                _variant_risk_cfg(params.get("risk") or {}, v),
                where=f"Variant {str(v.get('name'))!r}: ",
            )
            for v in variants
        }
        group_spec = params.get("group_limits")
        groups = resolve_group_limits(group_spec) if group_spec is not None else None
        # The run's files are the PRIMARY (first) variant's book, so its cap is the one recorded.
        risk_cfg = _variant_risk_cfg(params.get("risk") or {}, variants[0]) if variants else params.get("risk", {})
        return {
            "engine": engine,
            "execution": {**timing_record(timing), "calendar": timing.calendar},
            # The resolved timing itself: a same-bar run carries the granted override to the lag.
            "timing": timing,
            "venue": {
                "declared": venue_declared,
                "allow_shorts": allow_shorts,
                "max_leverage": gross_cap(risk_cfg),
                # A decision with net exposure above 1: scaled to 1, or borrowed (docs/adr/0007).
                "leverage": leverage,
                # What borrowed / idle cash costs; null = not declared (and not borrowing).
                "financing": financing.record() if financing is not None else None,
            },
            "financing": financing,
            # Whether the engine charges the funding series it is handed (rsims does, vectorbt does not).
            "charges_funding": adapter.charges_funding,
            # funding: {ignore, reason} — the one escape from the funding guard (quantbox.funding_guard).
            "funding_ignore": funding_ignore,
            "frequency": freq,
            "bars_per_year": bars_per_year,
            "trading_days": trading_days,
            "costs": costs,
            "engine_params": engine_params,
            "variant_costs": variant_costs,
            # What data.load_market_data receives (before the warmup lookback and mode are added).
            "load_params": load_params,
            # Also write the heavy report.html + report_data.json (TOM-1365); off by default.
            "full_report": full_report,
            # The rebalancing policy every engine follows (docs/adr/0008), and per variant.
            "policy": policy,
            "variant_policies": variant_policies,
            "rebalancing": policy.record(),
            # Group limits, unbound: run() binds them to the loaded universe.
            "groups": groups,
            "group_limits": groups.record() if groups is not None else None,
        }

    def check_planned_data(self, data: Any, paths: dict[str, str | None] | None, plan: dict[str, Any]) -> None:
        """Refuse a data plan this pipeline cannot run on — *paths* from ``data.planned_paths``.

        A backtest needs prices: with no prices file the data plugin hands back an
        empty frame and the run died later on an IndexError, after ``quantbox config
        explain`` had said ok. :meth:`run` and explain both call this on the same
        planned paths and :meth:`plan` (TOM-1362). A by-name dataset was already
        verified against its lock when it was resolved, so only an inline path is
        checked on disk. *paths* is None for a data plugin that plans no files.

        The funding guard (:func:`quantbox.funding_guard.check_funding`) runs here on
        the planned funding file and the data plugin's market
        (:func:`~quantbox.funding_guard.planned_market`): a funding file or a perp
        market on an engine that does not charge funding is refused, and a perp
        market with no funding file on one that does.
        """
        if paths is not None:
            name = getattr(getattr(data, "meta", None), "name", type(data).__name__)
            ppath = paths.get("prices")
            if not ppath:
                raise DataLoadError(name, "no prices source: set prices_path or dataset")
            if getattr(data, "dataset_resolution", None) is None and not Path(ppath).is_file():
                raise DataLoadError(name, f"prices file not found: {ppath}", path=str(ppath))
        check_funding(
            plan,
            (paths or {}).get("funding_rates"),
            market=planned_market(data),
            funding_known=paths is not None,
        )

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
        # Engine, execution timing + venue: resolved (and refused if malformed)
        # BEFORE any data is loaded, so a typo costs nothing and never runs
        # same-bar by accident. `quantbox config explain` reports this same plan.
        plan = self.plan(params)
        engine = plan["engine"]
        costs = plan["costs"]

        lag_bars, same_bar = plan["timing"].lag_bars, plan["timing"].same_bar
        allow_shorts, venue_declared = plan["venue"]["allow_shorts"], plan["venue"]["declared"]
        logger.info("Execution timing: %s", plan["execution"]["description"])
        if same_bar is not None:
            logger.warning(
                "SAME-BAR override (docs/adr/0006): this is a RESEARCH run, not a backtest — %s", same_bar.reason
            )

        # --- Stage 1: Universe & Market Data ---
        universe_params = params.get("universe", {})
        prices_params = plan["load_params"]
        planned_paths = getattr(data, "planned_paths", None)
        self.check_planned_data(data, planned_paths(prices_params) if callable(planned_paths) else None, plan)

        # ------------------------------------------------------------------
        # Frequency resolution (PR B / issue #20) — done in plan()
        #
        # A single Frequency value object from either the new top-level
        # `frequency:` block, or the legacy `prices.frequency` + optional
        # `market_calendar:` shorthand. `bars_per_year` derived there is used
        # as the DEFAULT for both `trading_days` (metrics annualization) and
        # the StrategyContext's bars_per_year (strategy-level vol annualization), so the
        # two cannot silently drift apart. Explicit `trading_days` /
        # strategy `annualize` values still win, with a drift warning.
        # ------------------------------------------------------------------
        freq, bars_per_year, trading_days = plan["frequency"], plan["bars_per_year"], plan["trading_days"]
        if "trading_days" in params and abs(trading_days - bars_per_year) > 1:
            logger.warning(
                "trading_days=%s overrides derived frequency=%s (bars_per_year=%.1f). "
                "If this is intentional, ignore; otherwise consider removing trading_days "
                "and letting the pipeline derive it from frequency.",
                params["trading_days"],
                freq,
                bars_per_year,
            )

        # Auto-derive minimum lookback from strategy warmup requirements.
        # Strategies may declare min_lookback_periods (in bars); convert to
        # days based on the requested frequency and take the max with whatever
        # the config already specifies.
        if strategies:
            min_bars = max(
                (getattr(s, "min_lookback_periods", 0) for s in strategies),
                default=0,
            )
            if min_bars > 0:
                step = interval_step(normalize_data_frequency(prices_params.get("frequency", "1d")))
                import math

                min_days = math.ceil(min_bars * step.total_seconds() / 86400)
                prices_params["lookback_days"] = max(int(prices_params.get("lookback_days", 0)), min_days)
                logger.info(
                    "Auto-derived lookback: %d bars → %d days (frequency=%s)",
                    min_bars,
                    prices_params["lookback_days"],
                    prices_params.get("frequency", "1d"),
                )

        universe = data.load_universe(universe_params)
        # Group limits read each symbol's group from the universe metadata (refused when it is missing).
        groups: GroupLimits | None = plan["groups"].bind(universe) if plan["groups"] is not None else None
        # Wire the run mode to the data plugin so mode-aware sources (e.g. the
        # universe-screen market_cap / screen_volume) pick the point-in-time
        # backtest path vs the live snapshot. Run mode is authoritative.
        prices_params["mode"] = mode
        market_data_dict = data.load_market_data(universe, asof, prices_params)

        store.put_parquet("universe", universe)

        market_data: dict[str, Any] = {"universe": universe}
        market_data.update(market_data_dict)
        for key in ("prices", "volume", "high", "low", "market_cap", "funding_rates", "eligibility_mask"):
            market_data.setdefault(key, pd.DataFrame())
        # The funding guard on what the data plugin handed back, before any strategy runs: a data plugin
        # that plans no file (an API plugin, dataset.curated.v1) is seen only here.
        data_name = getattr(getattr(data, "meta", None), "name", type(data).__name__)
        check_funding(
            plan,
            None if market_data["funding_rates"].empty else f"funding_rates returned by data plugin '{data_name}'",
            market=planned_market(data),
        )

        prices_wide = market_data["prices"]
        logger.info(
            "Data loaded: %d dates x %d tickers, range %s to %s",
            prices_wide.shape[0],
            prices_wide.shape[1],
            prices_wide.index[0] if len(prices_wide) else "?",
            prices_wide.index[-1] if len(prices_wide) else "?",
        )

        # --- Variants branch ---
        # When the config declares `variants:`, each variant runs an independent
        # backtest with its own strategy + optional overrides; results are
        # overlaid in a single combined report. Single-strategy flow is unchanged
        # when `variants:` is absent.
        variants_cfg = params.get("variants") or []
        variant_plugins = kwargs.get("variant_plugins") or {}
        overlay_chain: list[OverlayLink] = list(kwargs.get("overlays") or [])
        if variants_cfg:
            return self._run_variants_flow(
                mode=mode,
                asof=asof,
                params=params,
                store=store,
                market_data=market_data,
                variants_cfg=variants_cfg,
                variant_plugins=variant_plugins,
                risk=risk,
                engine=engine,
                groups=groups,
                trading_days=trading_days,
                bars_per_year=bars_per_year,
                lag_bars=lag_bars,
                allow_shorts=allow_shorts,
                venue_declared=venue_declared,
                plan=plan,
                overlay_chain=overlay_chain,
            )

        # --- Stage 2: Strategy Execution ---
        # The ONE strategy runner, shared with the trading pipeline (TOM-1448):
        # the strategy reads bars_per_year from the StrategyContext, the same
        # value `trading_days` defaults to.
        strategies_cfg = params.get("_strategies_cfg", params.get("strategies", []))
        context = build_strategy_context(mode, asof, params, prices_params)
        strategy_results = run_strategies(market_data, strategies_cfg, context, plugins=strategies)

        # Save per-strategy weights snapshot (last row, same as TradingPipeline)
        strat_weights_records: list[dict[str, Any]] = []
        for sname, sinfo in strategy_results.items():
            w = sinfo["result"].get("weights")
            if w is not None and not w.empty:
                last_row = w.iloc[-1] if isinstance(w, pd.DataFrame) else w
                for ticker, wt in last_row.items():
                    strat_weights_records.append({"strategy": sname, "symbol": str(ticker), "weight": float(wt)})
        a_strat_w = store.put_parquet("strategy_weights", pd.DataFrame(strat_weights_records))

        # --- Stage 3: Aggregate weights → full time series ---
        weights_history = self._aggregate_weights_history(strategy_results, params)
        logger.info(
            "Aggregated weights: %d dates x %d assets",
            weights_history.shape[0],
            weights_history.shape[1],
        )

        # --- Stage 3b: the overlay chain modifies the DECIDED book ---
        base_weights = weights_history
        weights_history, overlays_applied = self._apply_overlay_stage(weights_history, market_data, overlay_chain)
        overlay_artifacts: dict[str, str] = {}
        if overlays_applied:
            base_save = base_weights.copy()
            base_save.index.name = "date"
            overlay_artifacts["base_weights_history"] = store.put_parquet(
                "base_weights_history", base_save.reset_index()
            )

        # Save latest aggregated weights (same as TradingPipeline)
        latest_weights = weights_history.iloc[-1]
        agg_records = [{"symbol": str(k), "weight": float(v)} for k, v in latest_weights.items() if v != 0]
        a_agg_w = store.put_parquet("aggregated_weights", pd.DataFrame(agg_records))

        # Save full weights history
        wh_save = weights_history.copy()
        wh_save.index.name = "date"
        a_wh = store.put_parquet("weights_history", wh_save.reset_index())

        # --- Stage 4: the decision — the final target weights (short clip, gross cap, group limits,
        # normalisation; quantbox.decision, the same transform live trading calls, TOM-1520) ---
        risk_cfg = params.get("risk", {})
        target_stats = exposure_metrics(weights_history, "target")
        weights_history, decision = self._decide(weights_history, risk_cfg, allow_shorts, groups, plan)

        # --- Stage 5-6: the engine seam — calendars, the lag, the policy, the cash cap, financing legs, the engine ---
        book = self._simulate(prices_wide, weights_history, market_data, plan, costs, plan["policy"])
        bt_prices, bt_weights = book.prices, book.weights
        common_cols = [c for c in weights_history.columns if c in prices_wide.columns]
        a_traded = store.put_parquet("traded_weights", bt_weights.rename_axis("date").reset_index())
        a_validation = store.put_json("data_validation", with_decision(book.data_validation, decision))
        a_schedule = store.put_parquet("rebalance_schedule", book.schedule)

        logger.info(
            "Backtest window: %d dates x %d assets, engine=%s",
            bt_prices.shape[0],
            bt_prices.shape[1],
            engine,
        )

        # --- Stage 6: Save artifacts ---
        returns_series = book.returns
        metrics = {
            **book.metrics,
            **self._book_metrics(target_stats, bt_weights, lag_bars, allow_shorts, venue_declared, "single run"),
            **book.book_metrics,
            **decision_metrics(decision),
        }
        portfolio_daily = book.portfolio_daily

        # run@1 returns: (date, returns). Named here: a data plugin's index may carry no name (TOM-1529).
        a_returns = store.put_parquet("returns", returns_series.to_frame("returns").rename_axis("date").reset_index())
        a_port = store.put_parquet("portfolio_daily", portfolio_daily.reset_index())
        a_metrics = store.put_json("metrics", metrics)

        # --- Stage 7: Reports ---
        from quantbox.plugins.pipeline._report import (
            build_reproducibility,
            generate_html_report,
            generate_report_data,
            generate_summary_md,
            report_data_to_json,
            resolve_narrative,
        )

        period_start = str(returns_series.index[0])[:10] if len(returns_series) else asof
        period_end = str(returns_series.index[-1])[:10] if len(returns_series) else asof
        report_strategy_names = [sc["name"] for sc in strategies_cfg]
        report_metrics = {**metrics, "n_assets": float(len(common_cols)), "n_dates": float(len(bt_prices))}
        strategy_details = {
            sname: sinfo["result"].get("details", {}) or {} for sname, sinfo in strategy_results.items()
        }
        narrative = resolve_narrative(params.get("narrative"))
        reproducibility = build_reproducibility(
            run_id=store.run_id,
            asof=asof,
            pipeline_name=self.meta.name,
            pipeline_version=self.meta.version,
            params=params,
            period_start=period_start,
            period_end=period_end,
            execution=plan["execution"],
        )

        store.put_text(
            "summary.md",
            generate_summary_md(
                run_id=store.run_id,
                asof=asof,
                metrics=report_metrics,
                strategy_names=report_strategy_names,
                period_start=period_start,
                period_end=period_end,
                execution=plan["execution"]["description"],
            ),
        )
        # The heavy HTML report is opt-in; the slim finding_report.json is written by the runner.
        if plan["full_report"]:
            try:
                rd = generate_report_data(
                    run_id=store.run_id,
                    asof=asof,
                    metrics=report_metrics,
                    portfolio_daily=portfolio_daily,
                    returns=returns_series,
                    # The report describes the TRADED book (post venue/risk/lag):
                    # its attribution is `w.shift(1) * ret` — "held at the close
                    # of t, earns t+1" — which is only true of what the engine
                    # actually filled.
                    weights_history=bt_weights,
                    bt_prices=bt_prices,
                    strategy_names=report_strategy_names,
                    period_start=period_start,
                    period_end=period_end,
                    vbt_portfolio=_vbt_portfolio(book),
                    strategy_details=strategy_details,
                    narrative=narrative,
                    reproducibility=reproducibility,
                    execution=plan["execution"]["description"],
                )
                store.put_text("report_data.json", report_data_to_json(rd))
                store.put_text("report.html", generate_html_report(rd))
            except Exception as _report_exc:
                logger.warning("HTML report generation failed: %s", _report_exc)

        # --- Risk checks on latest targets ---
        targets = pd.DataFrame(agg_records)
        risk_findings: list[dict[str, Any]] = []
        for rp in risk:
            try:
                risk_findings.extend(rp.check_targets(targets, risk_cfg))
            except Exception as exc:
                logger.warning("Risk check failed: %s", exc)

        logger.info(
            "Backtest complete: total_return=%.4f, sharpe=%.4f, max_dd=%.4f",
            metrics.get("total_return", 0),
            metrics.get("sharpe", 0),
            metrics.get("max_drawdown", 0),
        )

        return RunResult(
            run_id=store.run_id,
            pipeline_name=self.meta.name,
            mode=mode,
            asof=asof,
            artifacts={
                "strategy_weights": a_strat_w,
                "aggregated_weights": a_agg_w,
                "weights_history": a_wh,
                "traded_weights": a_traded,
                "portfolio_daily": a_port,
                "returns": a_returns,
                "metrics": a_metrics,
                "data_validation": a_validation,
                "rebalance_schedule": a_schedule,
                **overlay_artifacts,
            },
            metrics={
                "n_strategies": float(len(strategy_results)),
                "n_assets": float(len(common_cols)),
                "n_dates": float(len(bt_prices)),
                **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            },
            notes={
                "kind": "backtest",
                "engine": engine,
                "execution": plan["execution"],
                "venue": plan["venue"],
                # The rebalancing policy and the group limits the seam applied (docs/adr/0008).
                "rebalancing": plan["rebalancing"],
                "group_limits": plan["group_limits"],
                "funding": {"modelled": book.funding_modelled, **ignored_record(plan)},
                "financing": book.financing,
                "data_validation": self._validation_note(book.data_validation),
                "overlays": overlays_applied,
                "risk_findings": risk_findings,
            },
        )

    # ==================================================================
    # Stage 3: Aggregate weights → full time series
    # ==================================================================
    def _aggregate_weights_history(
        self,
        strategy_results: dict[str, dict[str, Any]],
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Aggregate multi-strategy weights into a single DataFrame over
        the full historical period (not just the last row)."""
        weight_overrides = params.get("strategy_weights", {})
        names = list(strategy_results.keys())

        weight_dfs: list[pd.DataFrame] = []
        account_weights: list[float] = []

        for sname in names:
            sinfo = strategy_results[sname]
            w_df = sinfo["result"].get("weights", pd.DataFrame())
            if w_df is None or (isinstance(w_df, pd.DataFrame) and w_df.empty):
                continue
            weight_dfs.append(w_df)
            account_weights.append(float(weight_overrides.get(sname, sinfo["weight"])))

        if not weight_dfs:
            return pd.DataFrame()

        if len(weight_dfs) == 1:
            return weight_dfs[0] * account_weights[0]

        # Multi-strategy: concat, scale, sum
        try:
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
            flat = weighted.droplevel(0, axis=1)
            if isinstance(flat.columns, pd.MultiIndex) or flat.columns.duplicated().any():
                flat = flat.T.groupby(level=0).sum().T
            return flat
        except Exception:
            logger.warning("Multi-strategy concat failed, using manual fallback")
            # Manual fallback
            all_idx = weight_dfs[0].index
            all_cols: set = set()
            for df in weight_dfs:
                all_idx = all_idx.union(df.index)
                all_cols.update(df.columns.tolist())
            result = pd.DataFrame(0.0, index=all_idx, columns=sorted(all_cols))
            for df, w in zip(weight_dfs, account_weights, strict=False):
                aligned = df.reindex(index=all_idx, columns=sorted(all_cols)).fillna(0)
                result += aligned * w
            return result

    # ==================================================================
    # Multi-variant orchestration
    # ==================================================================
    def _run_variants_flow(
        self,
        *,
        mode: Mode,
        asof: str,
        params: dict[str, Any],
        store: ArtifactStore,
        market_data: dict[str, Any],
        variants_cfg: list[dict[str, Any]],
        variant_plugins: dict[str, StrategyPlugin],
        risk: list[RiskPlugin],
        engine: str,
        groups: GroupLimits | None,
        trading_days: int,
        bars_per_year: float,
        lag_bars: int,
        allow_shorts: bool,
        venue_declared: bool,
        plan: dict[str, Any],
        overlay_chain: list[OverlayLink],
    ) -> RunResult:
        """Run N independent variants and emit a combined report.

        Each variant has: name, strategy (registry name), optional strategy.params,
        optional overrides (fees, threshold, rebalancing_freq, rebalancing_policy, risk: {...}).
        Reuses the one strategy runner, _aggregate_weights_history,
        the decision (:meth:`_decide`) and the engine seam (:meth:`_simulate`) for
        parity with the single-variant path — on either engine.
        """
        prices_wide = market_data["prices"]
        base_risk_cfg = dict(params.get("risk", {}) or {})
        # Every variant runs in the same run: one StrategyContext for all of them.
        context = build_strategy_context(mode, asof, params, plan["load_params"])

        variant_results: dict[str, dict[str, Any]] = {}
        overlays_applied: list[dict[str, Any]] = []

        for v in variants_cfg:
            vname = str(v["name"])
            strat_cfg = v.get("strategy") or {}
            if isinstance(strat_cfg, dict):
                sname = strat_cfg.get("name") or strat_cfg.get("source")  # source: file.py:Class
            else:
                sname = str(strat_cfg)
            if not sname:
                raise ValueError(f"Variant {vname!r}: missing strategy.name or strategy.source")
            splugin = variant_plugins.get(vname) or variant_plugins.get(sname)
            if splugin is None:
                raise ValueError(f"Variant {vname!r}: no resolved plugin for strategy {sname!r}")
            strat_params = (strat_cfg.get("params") or {}) if isinstance(strat_cfg, dict) else {}

            # Per-variant overrides; plan() already refused run-level keys, non-numeric
            # costs and a malformed policy, before any data was loaded (TOM-1362).
            v_costs = plan["variant_costs"][vname]
            v_policy = plan["variant_policies"][vname]
            v_risk_cfg = _variant_risk_cfg(base_risk_cfg, v)

            v_strategies_cfg = [{"name": sname, "weight": 1.0, "params": strat_params}]

            # Stage 2: strategy
            s_results = run_strategies(market_data, v_strategies_cfg, context, plugins=[splugin])

            # Stage 3: aggregate (trivial for single strategy)
            wh = self._aggregate_weights_history(s_results, {"_strategies_cfg": v_strategies_cfg})
            # Stage 3b: the overlay chain is run-level — every variant gets the same one.
            wh, overlays_applied = self._apply_overlay_stage(wh, market_data, overlay_chain)

            # Stage 4: the decision (final target weights), exactly as the single run
            v_allow_shorts = allow_shorts if venue_declared else bool(v_risk_cfg.get("allow_short", False))
            v_target_stats = exposure_metrics(wh, "target")
            where = f"Variant {vname!r}: "
            wh, v_decision = self._decide(wh, v_risk_cfg, v_allow_shorts, groups, plan, where=where)

            # Stage 5-6: the engine seam, exactly as the single run
            try:
                res = self._simulate(prices_wide, wh, market_data, plan, v_costs, v_policy, where=where)
            except ValueError as exc:
                msg = str(exc)
                raise ValueError(msg if msg.startswith("Variant") else f"Variant {vname!r}: {msg}") from exc
            bt_p, bt_w = res.prices, res.weights

            details_by_strategy = {
                k: (info.get("result") or {}).get("details", {}) or {} for k, info in s_results.items()
            }

            variant_results[vname] = {
                "name": vname,
                "strategy_name": sname,
                "returns": res.returns,
                "metrics": {
                    **res.metrics,
                    **self._book_metrics(
                        v_target_stats, bt_w, lag_bars, v_allow_shorts, venue_declared, f"variant {vname!r}"
                    ),
                    **res.book_metrics,
                    **decision_metrics(v_decision),
                },
                "data_validation": with_decision(res.data_validation, v_decision),
                "schedule": res.schedule,
                "financing": res.financing,
                "funding_modelled": res.funding_modelled,
                "portfolio_daily": res.portfolio_daily,
                "vbt_portfolio": _vbt_portfolio(res),
                # TRADED weights (post venue/risk/lag) — what the engine filled.
                "weights_history": bt_w,
                "bt_prices": bt_p,
                "strategy_details": details_by_strategy,
                "config": {
                    "strategy_params": strat_params,
                    "fees": v_costs["fees"],
                    "rebalancing_freq": v_policy.frequency,
                    "threshold": v_policy.band if v_policy.trigger == "band" else None,
                    "rebalancing_policy": v_policy.record(),
                    "risk": v_risk_cfg,
                    # Optional explicit flag — when set, this variant becomes
                    # the source of the shared § 03 diagnostics in the report.
                    # Without it, the template picks the highest-Sharpe
                    # non-benchmark variant.
                    "primary": bool(v.get("primary", False)),
                },
            }
            logger.info(
                "Variant %r done — total_return=%.4f sharpe=%.4f maxdd=%.4f",
                vname,
                res.metrics.get("total_return", 0),
                res.metrics.get("sharpe", 0),
                res.metrics.get("max_drawdown", 0),
            )

        if not variant_results:
            raise ValueError("Variants flow: no variants produced results")

        # Primary variant (first) provides top-level artifacts for backwards compat
        primary_name = next(iter(variant_results))
        primary = variant_results[primary_name]

        a_returns = store.put_parquet(
            "returns", primary["returns"].to_frame("returns").rename_axis("date").reset_index()
        )
        a_port = store.put_parquet("portfolio_daily", primary["portfolio_daily"].reset_index())
        a_metrics = store.put_json("metrics", primary["metrics"])
        a_traded = store.put_parquet("traded_weights", primary["weights_history"].rename_axis("date").reset_index())
        # The primary variant's calendar report; every variant's totals beside it (their targets differ).
        validation = {
            **primary["data_validation"],
            "variants": {
                n: {
                    "timing": r["data_validation"]["timing"],
                    "staleness": r["data_validation"]["staleness"],
                    "weight_age": r["data_validation"]["weight_age"],
                    "index_alignment": r["data_validation"]["index_alignment"],
                    "leverage": r["data_validation"]["leverage"],
                    "decision": r["data_validation"]["decision"],
                }
                for n, r in variant_results.items()
            },
        }
        a_validation = store.put_json("data_validation", validation)
        a_schedule = store.put_parquet("rebalance_schedule", primary["schedule"])

        # Per-variant metrics table
        metric_rows = []
        for n, r in variant_results.items():
            row = {"variant": n, "strategy": r["strategy_name"]}
            for k, v_ in r["metrics"].items():
                if isinstance(v_, (int, float)):
                    row[k] = float(v_)
            metric_rows.append(row)
        a_var_metrics = store.put_parquet("variant_metrics", pd.DataFrame(metric_rows))
        # Every arm's return series, LONG (date, variant, returns) so a variant may be
        # named anything ("date" included): the finding-report export reads it.
        a_var_returns = store.put_parquet(
            "variant_returns",
            pd.concat(
                [
                    pd.DataFrame({"date": r["returns"].index, "variant": n, "returns": r["returns"].to_numpy()})
                    for n, r in variant_results.items()
                ],
                ignore_index=True,
            ),
        )

        period_start = str(primary["returns"].index[0])[:10] if len(primary["returns"]) else asof
        period_end = str(primary["returns"].index[-1])[:10] if len(primary["returns"]) else asof

        # Reports
        try:
            from quantbox.plugins.pipeline._report import (
                build_reproducibility,
                generate_html_report,
                generate_report_data,
                generate_summary_md,
                report_data_to_json,
                resolve_narrative,
            )

            report_metrics = {
                **primary["metrics"],
                "n_assets": float(len(primary["bt_prices"].columns)),
                "n_dates": float(len(primary["bt_prices"])),
            }
            narrative = resolve_narrative(params.get("narrative"))
            reproducibility = build_reproducibility(
                run_id=store.run_id,
                asof=asof,
                pipeline_name=self.meta.name,
                pipeline_version=self.meta.version,
                params=params,
                period_start=period_start,
                period_end=period_end,
                variant_results=variant_results,
                execution=plan["execution"],
            )
            store.put_text(
                "summary.md",
                generate_summary_md(
                    run_id=store.run_id,
                    asof=asof,
                    metrics=report_metrics,
                    strategy_names=list(variant_results.keys()),
                    period_start=period_start,
                    period_end=period_end,
                    execution=plan["execution"]["description"],
                ),
            )
            if plan["full_report"]:  # the heavy HTML report is opt-in
                rd = generate_report_data(
                    run_id=store.run_id,
                    asof=asof,
                    metrics=report_metrics,
                    portfolio_daily=primary["portfolio_daily"],
                    returns=primary["returns"],
                    weights_history=primary["weights_history"],
                    bt_prices=primary["bt_prices"],
                    strategy_names=list(variant_results.keys()),
                    period_start=period_start,
                    period_end=period_end,
                    vbt_portfolio=primary["vbt_portfolio"],
                    strategy_details={
                        vname: (next(iter(r["strategy_details"].values()), {}) or {})
                        for vname, r in variant_results.items()
                    },
                    variant_results=variant_results,
                    narrative=narrative,
                    reproducibility=reproducibility,
                    execution=plan["execution"]["description"],
                )
                store.put_text("report_data.json", report_data_to_json(rd))
                store.put_text("report.html", generate_html_report(rd))
        except Exception as _exc:
            logger.warning("Multi-variant report generation failed: %s", _exc)

        # Risk checks on primary variant's latest targets
        latest = primary["weights_history"].iloc[-1] if len(primary["weights_history"]) else pd.Series(dtype=float)
        agg_records = [{"symbol": str(k), "weight": float(v)} for k, v in latest.items() if v != 0]
        risk_findings: list[dict[str, Any]] = []
        for rp in risk:
            try:
                risk_findings.extend(rp.check_targets(pd.DataFrame(agg_records), base_risk_cfg))
            except Exception as exc:
                logger.warning("Risk check failed: %s", exc)

        flat_metrics: dict[str, float] = {
            "execution_lag_bars": float(lag_bars),
            "n_variants": float(len(variant_results)),
            "n_dates": float(len(primary["returns"])),
        }
        for n, r in variant_results.items():
            for k, v_ in r["metrics"].items():
                if isinstance(v_, (int, float)):
                    flat_metrics[f"{n}__{k}"] = float(v_)

        return RunResult(
            run_id=store.run_id,
            pipeline_name=self.meta.name,
            mode=mode,
            asof=asof,
            artifacts={
                "returns": a_returns,
                "portfolio_daily": a_port,
                "metrics": a_metrics,
                "traded_weights": a_traded,
                "variant_metrics": a_var_metrics,
                "variant_returns": a_var_returns,
                "data_validation": a_validation,
                "rebalance_schedule": a_schedule,
            },
            metrics=flat_metrics,
            notes={
                "kind": "backtest-variants",
                "engine": engine,
                "execution": plan["execution"],
                # The run's files are the PRIMARY (first) variant's book; plan() records its cap.
                "venue": plan["venue"],
                # The PRIMARY variant's policy (a variant may override it); the group limits are run-level.
                "rebalancing": primary["config"]["rebalancing_policy"],
                "group_limits": plan["group_limits"],
                # The primary variant's book, as every other file of the run.
                "funding": {"modelled": primary["funding_modelled"], **ignored_record(plan)},
                "financing": primary["financing"],
                "data_validation": self._validation_note(validation),
                "variants": list(variant_results.keys()),
                "overlays": overlays_applied,
                "risk_findings": risk_findings,
            },
        )

    # ==================================================================
    # Stage 3b: overlay chain on the decided book
    # ==================================================================
    @staticmethod
    def _apply_overlay_stage(
        weights: pd.DataFrame,
        market_data: dict[str, Any],
        chain: list[OverlayLink],
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        """Run the overlay chain on the decided weights; a no-op without overlays.

        The seam's NaN policy is materialised FIRST (:func:`quantbox.engine.materialise_nan`,
        HOLD, the same on every engine): an overlay multiplying a NaN would lose
        its effect on exactly the bars it targets. The policy is idempotent, so
        the seam later builds the same book from it.
        No lag is applied here — the engine seam schedules and lags the overlaid book once.

        The filled value must not LEAK past the chain, though: the seam
        reindexes the book onto the price bars BEFORE it resolves its NaNs, so a
        filled cell is not the same book as a NaN one (the decision,
        :func:`quantbox.decision.final_targets`, keeps an untouched NaN for the
        same reason). So a cell that came in NaN and that the chain left at exactly its
        materialised value goes back to NaN — no overlay touched it, and the
        book downstream is the one the run without overlays builds. A cell the
        chain CHANGED keeps the overlay's number.

        Under the HOLD policy a NaN resolves to the PREVIOUS row's value, so
        "untouched" alone is not enough: the first bar after an overlay window
        closes is untouched yet must stay explicit, or the seam holds the last
        REDUCED weight instead of returning to the base one. A NaN goes back
        only where the previous row was untouched too.
        """
        if not chain:
            return weights, []
        assert NAN_POLICY == "hold"  # the mask below is the hold policy's
        materialised = materialise_nan(weights)
        out, record = apply_overlays(materialised, market_data, chain)
        same = out.eq(materialised) | (out.isna() & materialised.isna())
        untouched = weights.isna() & same & same.shift(1, fill_value=True)
        return out.mask(untouched), record

    # ==================================================================
    # Stage 4: the decision — final target weights on the full time series
    # ==================================================================
    @staticmethod
    def _decide(
        weights: pd.DataFrame,
        risk_cfg: dict[str, Any],
        allow_shorts: bool,
        groups: GroupLimits | None,
        plan: dict[str, Any],
        *,
        where: str = "",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """The decided book -> the FINAL target weights (:func:`quantbox.decision.final_targets`, TOM-1520).

        One ordered transform, the one live trading calls: the short clip
        (``venue.allow_shorts`` / ``risk.allow_short``), the gross cap
        (``risk.max_leverage``), the group limits, then ``venue.leverage``
        (``normalize`` scales a row above net 1 to net 1; ``borrow`` keeps it;
        ``none`` on ``execution.schedule: bars`` only measures). The seam and
        its rebalancing policy read these targets; tranching (the tranche
        cadence) therefore averages final rows.
        """
        rules = DecisionRules(
            allow_short=allow_shorts,
            max_leverage=gross_cap(risk_cfg),
            groups=groups,
            leverage=plan["venue"]["leverage"],
        )
        final, report = final_targets(weights, rules)
        log_normalisation(report, where=where)
        return final, report

    @staticmethod
    def _simulate(
        prices_wide: pd.DataFrame,
        weights: pd.DataFrame,
        market_data: dict[str, Any],
        plan: dict[str, Any],
        costs: dict[str, float],
        policy: RebalancePolicy,
        *,
        where: str = "",
    ) -> TradedBook:
        """The final targets through the engine seam (:func:`quantbox.engine.simulate`).

        The ONLY place target weights become traded weights, shared by the
        single-run and variants flows and by every engine: calendars, the
        rebalancing policy, the execution lag, the cash cap, financing legs
        and the adapter (docs/adr/0007, 0008). The group limits and the
        normalisation already ran in :meth:`_decide`. The engine reads the
        funding series only when it charges funding; the seam refuses a series
        it would not charge unless the plan declares the ignore.
        """
        leverage = plan["venue"]["leverage"]
        return simulate(
            prices_wide,
            weights,
            engine=plan["engine"],
            timing=plan["timing"],
            costs=Costs(fees=costs["fees"], fixed_fees=costs["fixed_fees"], slippage=costs["slippage"]),
            policy=policy,
            leverage=None if leverage == "none" else leverage,
            financing=plan.get("financing"),
            funding=market_data.get("funding_rates"),
            funding_ignore=plan["funding_ignore"],
            engine_params=plan["engine_params"],
            trading_days=plan["trading_days"],
            where=where,
        )

    @staticmethod
    def _validation_note(validation: dict[str, Any]) -> dict[str, Any]:
        """``RunResult.notes['data_validation']`` / run@1 ``data_validation``: the file and its totals."""
        exec_cal = validation["execution_calendar"]
        timing = validation["timing"]
        return {
            "schema": validation["schema"],
            "file": "data_validation.json",
            "calendar": calendar_summary(validation["calendar"]),
            "execution_calendar": {k: exec_cal[k] for k in ("calendar", "execution_bars", "total_bars")},
            "timing": {
                "lag_counted_in": timing["lag_counted_in"],
                "decisions": timing["decisions"],
                "executed_decisions": timing["executed_decisions"],
                "deferred_trades": timing["deferred_trades"],
                "targeted_outside_window_bars": timing["targeted_outside_window_bars"],
            },
            "staleness": validation["staleness"],
            "weight_age": {k: v for k, v in validation["weight_age"].items() if k != "rule"},
            "index_alignment": {
                k: validation["index_alignment"][k]
                for k in (
                    "bars_used",
                    "warmup_price_bars_dropped",
                    "price_bars_dropped",
                    "weight_rows_dropped",
                )
            },
            "leverage": validation["leverage"],
        }

    @staticmethod
    def _book_metrics(
        target_stats: dict[str, float],
        traded_weights: pd.DataFrame,
        lag_bars: int,
        allow_shorts: bool,
        venue_declared: bool,
        where: str,
    ) -> dict[str, float]:
        """Traded-book metrics (from what the engine received) + the shorts alarm."""
        traded = exposure_metrics(traded_weights, "traded")
        warn_on_shorts(
            target_short_share=target_stats["target_short_gross_share"],
            traded_short_share=traded["traded_short_gross_share"],
            venue_declared=venue_declared,
            allow_shorts=allow_shorts,
            where=where,
        )
        return {
            "execution_lag_bars": float(lag_bars),
            **traded,
            "target_short_gross_share": target_stats["target_short_gross_share"],
            "target_mean_net_exposure": target_stats["target_mean_net_exposure"],
        }

    # ==================================================================
    # Frequency resolution (issue #20)
    # ==================================================================
    @staticmethod
    def _resolve_frequency(
        params: dict[str, Any],
        prices_params: dict[str, Any],
    ) -> Frequency:
        """Resolve a `Frequency` from pipeline params.

        Delegates to `quantbox.frequency.resolve_pipeline_frequency`, which the
        trading pipeline calls too, so the StrategyContext's bars_per_year is the same value
        in backtest and paper/live (TOM-1338). The resolution order is stated
        there. The derived `bars_per_year` is also the DEFAULT `trading_days`.
        """
        return resolve_pipeline_frequency(params, prices_params)
