"""Backtesting pipeline plugin.

Uses the same config format as :class:`TradingPipeline` but replaces
broker execution with a historical simulation via the vectorbt or rsims
backtesting engines.

Workflow
-------
1. Load universe & market data (same as TradingPipeline)
2. Run strategies → full weights time series
3. Aggregate across strategies (same logic)
4. Apply risk transforms (tranching, leverage cap)
5. Run backtest engine over full history
6. Compute performance metrics
7. Save artifacts (weights, returns, metrics, portfolio_daily)

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

import importlib
import logging
from dataclasses import dataclass
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
from quantbox.frequency import Frequency
from quantbox.plugins.datasources._utils import interval_step, normalize_data_frequency

logger = logging.getLogger(__name__)


@dataclass
class BacktestPipeline:
    meta = PluginMeta(
        name="backtest.pipeline.v1",
        kind="pipeline",
        version="0.2.0",
        core_compat=">=0.1,<0.2",
        description=(
            "Backtesting pipeline: same config as TradingPipeline but "
            "routes weights through vectorbt or rsims backtesting engines "
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
                    "enum": ["vectorbt", "rsims"],
                    "default": "vectorbt",
                    "description": "Backtesting engine to use.",
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
                },
                "slippage": {
                    "type": "number",
                    "minimum": 0,
                    "default": 0.0,
                },
                "rebalancing_freq": {
                    "description": (
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
                    "description": "Deviation threshold for rebalancing bands (vectorbt only).",
                },
                "initial_cash": {
                    "type": "number",
                    "minimum": 0,
                    "default": 10000,
                    "description": "Starting cash (rsims only).",
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
                    "default": False,
                    "description": "Compound profits into sizing (rsims only).",
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
            },
        },
        inputs=(),
        outputs=(
            "strategy_weights",
            "aggregated_weights",
            "weights_history",
            "portfolio_daily",
            "returns",
            "metrics",
        ),
        examples=(
            "run:\n  mode: backtest\n  asof: '2026-02-01'\n  pipeline: backtest.pipeline.v1\n"
            "plugins:\n  pipeline:\n    name: backtest.pipeline.v1\n    params:\n"
            "      engine: vectorbt\n      fees: 0.001\n      rebalancing_freq: 1W\n"
            "      strategies:\n        - name: crypto_trend\n          weight: 1.0\n",
        ),
    )
    kind = "research"

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
        engine = str(params.get("engine", "vectorbt")).lower()
        fees = float(params.get("fees", 0.001))
        fixed_fees = float(params.get("fixed_fees", 0.0))
        slippage_val = float(params.get("slippage", 0.0))
        rebalancing_freq = params.get("rebalancing_freq", 1)
        threshold = params.get("threshold")

        # --- Stage 1: Universe & Market Data ---
        universe_params = params.get("universe", {})
        prices_params = dict(params.get("prices", {"lookback_days": 365}))

        # ------------------------------------------------------------------
        # Frequency resolution (PR B / issue #20)
        #
        # Build a single Frequency value object from either the new top-level
        # `frequency:` block, or the legacy `prices.frequency` + optional
        # `market_calendar:` shorthand. `bars_per_year` derived here is used
        # as the DEFAULT for both `trading_days` (metrics annualization) and
        # `_pipeline_annualize` (strategy-level vol annualization), so the
        # two cannot silently drift apart. Explicit `trading_days` /
        # strategy `annualize` values still win, with a drift warning.
        # ------------------------------------------------------------------
        freq = self._resolve_frequency(params, prices_params)
        bars_per_year = freq.bars_per_year()
        trading_days = int(params.get("trading_days", round(bars_per_year)))
        if "trading_days" in params and abs(int(params["trading_days"]) - bars_per_year) > 1:
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
                fees=fees,
                fixed_fees=fixed_fees,
                slippage_val=slippage_val,
                rebalancing_freq=rebalancing_freq,
                threshold=threshold,
                trading_days=trading_days,
                bars_per_year=bars_per_year,
            )

        # --- Stage 2: Strategy Execution ---
        strategies_cfg = params.get("_strategies_cfg", params.get("strategies", []))
        if strategies:
            strategy_results = self._run_strategy_plugins(
                strategies,
                strategies_cfg,
                market_data,
                injected_annualize=bars_per_year,
            )
        else:
            params = {**params, "_pipeline_annualize": bars_per_year}
            strategy_results = self._run_strategies(market_data, strategies_cfg, params)

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

        # Save latest aggregated weights (same as TradingPipeline)
        latest_weights = weights_history.iloc[-1]
        agg_records = [{"symbol": str(k), "weight": float(v)} for k, v in latest_weights.items() if v != 0]
        a_agg_w = store.put_parquet("aggregated_weights", pd.DataFrame(agg_records))

        # Save full weights history
        wh_save = weights_history.copy()
        wh_save.index.name = "date"
        a_wh = store.put_parquet("weights_history", wh_save.reset_index())

        # --- Stage 4: Risk transforms on the full time series ---
        risk_cfg = params.get("risk", {})
        weights_history = self._apply_risk_transforms_ts(weights_history, risk_cfg)

        # --- Stage 5: Run backtest engine ---
        # Align prices and weights to the same index & columns
        common_idx = prices_wide.index.intersection(weights_history.index)
        common_cols = [c for c in weights_history.columns if c in prices_wide.columns]
        if not common_cols:
            raise ValueError("No overlapping tickers between prices and weights")

        bt_prices = prices_wide.loc[common_idx, common_cols]
        bt_weights = weights_history.loc[common_idx, common_cols]

        # Drop columns with < 50% non-null prices first so a newly-listed coin
        # doesn't truncate the entire simulation window to its listing date.
        min_obs = max(30, int(len(bt_prices) * 0.5))
        sufficient_cols = bt_prices.columns[bt_prices.notna().sum() >= min_obs].tolist()
        dropped = [c for c in bt_prices.columns if c not in sufficient_cols]
        if dropped:
            logger.info("Dropped %d short-history columns: %s", len(dropped), dropped)
        bt_prices = bt_prices[sufficient_cols]
        bt_weights = bt_weights[sufficient_cols]

        # Where a coin has no price yet (not yet listed), force weight to 0 so
        # the backtest doesn't try to hold it, then forward-fill prices for
        # simulation continuity.  This allows a broad dynamic pool where
        # individual coins enter the universe at different dates without
        # truncating the entire simulation window to the latest listing date.
        # Any row that is ALL NaN (before any coin was available) is still dropped.
        all_nan_rows = bt_prices.isna().all(axis=1)
        bt_weights = bt_weights.where(bt_prices.notna(), 0.0)
        bt_prices = bt_prices.ffill().bfill()
        bt_prices = bt_prices.loc[~all_nan_rows]
        bt_weights = bt_weights.loc[~all_nan_rows]

        logger.info(
            "Backtest window: %d dates x %d assets, engine=%s",
            bt_prices.shape[0],
            bt_prices.shape[1],
            engine,
        )

        if engine == "vectorbt":
            result_data = self._run_vectorbt(
                bt_prices,
                bt_weights,
                fees=fees,
                fixed_fees=fixed_fees,
                slippage=slippage_val,
                rebalancing_freq=rebalancing_freq,
                threshold=threshold,
                trading_days=trading_days,
            )
        elif engine == "rsims":
            funding_wide = market_data.get("funding_rates", pd.DataFrame())
            if funding_wide.empty:
                bt_funding = pd.DataFrame(0.0, index=bt_prices.index, columns=bt_prices.columns)
            else:
                bt_funding = funding_wide.reindex(index=bt_prices.index, columns=bt_prices.columns).fillna(0.0)

            result_data = self._run_rsims(
                bt_prices,
                bt_weights,
                bt_funding,
                fees=fees,
                trade_buffer=float(params.get("trade_buffer", 0.0)),
                initial_cash=float(params.get("initial_cash", 10000)),
                margin=float(params.get("margin", 0.0)),
                capitalise_profits=bool(params.get("capitalise_profits", False)),
                equity_basis=str(params.get("equity_basis", "rsims")),
                trading_days=trading_days,
            )
        else:
            raise ValueError(f"Unknown engine: {engine!r}. Use 'vectorbt' or 'rsims'.")

        # --- Stage 6: Save artifacts ---
        returns_series = result_data["returns"]
        metrics = result_data["metrics"]
        portfolio_daily = result_data["portfolio_daily"]

        a_returns = store.put_parquet("returns", returns_series.to_frame("returns").reset_index())
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
            ),
        )
        try:
            rd = generate_report_data(
                run_id=store.run_id,
                asof=asof,
                metrics=report_metrics,
                portfolio_daily=portfolio_daily,
                returns=returns_series,
                weights_history=wh_save,
                bt_prices=bt_prices,
                strategy_names=report_strategy_names,
                period_start=period_start,
                period_end=period_end,
                vbt_portfolio=result_data.get("vbt_portfolio"),
                strategy_details=strategy_details,
                narrative=narrative,
                reproducibility=reproducibility,
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
                "portfolio_daily": a_port,
                "returns": a_returns,
                "metrics": a_metrics,
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
                "risk_findings": risk_findings,
            },
        )

    # ==================================================================
    # Stage 2: Strategy execution (reused from TradingPipeline)
    # ==================================================================
    def _run_strategies(
        self,
        market_data: dict[str, Any],
        strategies_cfg: list[dict[str, Any]],
        pipeline_params: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        injected_annualize = pipeline_params.get("_pipeline_annualize")
        for strat_cfg in strategies_cfg:
            name = strat_cfg["name"]
            weight = float(strat_cfg.get("weight", 1.0))
            strat_params = dict(strat_cfg.get("params", {}))
            if injected_annualize is not None and "_pipeline_annualize" not in strat_params:
                strat_params["_pipeline_annualize"] = injected_annualize

            try:
                module = importlib.import_module(f"quantbox.plugins.strategies.{name}")
            except ImportError:
                logger.error("Could not import strategy '%s'", name)
                raise

            result = module.run(data=market_data, params=strat_params)

            # Normalize multi-level weight columns
            weights_df = result.get("weights", pd.DataFrame())
            if isinstance(weights_df, pd.DataFrame) and weights_df.columns.nlevels > 1:
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
        injected_annualize: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for i, strat in enumerate(strategy_plugins):
            strat_cfg = strategies_cfg[i] if i < len(strategies_cfg) else {}
            weight = float(strat_cfg.get("weight", 1.0))
            strat_params = dict(strat_cfg.get("params", {}))
            if injected_annualize is not None and "_pipeline_annualize" not in strat_params:
                strat_params["_pipeline_annualize"] = injected_annualize

            result = strat.run(data=market_data, params=strat_params)

            weights_df = result.get("weights", pd.DataFrame())
            if isinstance(weights_df, pd.DataFrame) and weights_df.columns.nlevels > 1:
                weights_df = weights_df.T.groupby("ticker").sum().T
                result["weights"] = weights_df

            results[strat.meta.name] = {"result": result, "weight": weight}
            logger.info("Strategy plugin '%s' completed (weight=%.2f)", strat.meta.name, weight)

        return results

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
        fees: float,
        fixed_fees: float,
        slippage_val: float,
        rebalancing_freq: Any,
        threshold: Any,
        trading_days: int,
        bars_per_year: float,
    ) -> RunResult:
        """Run N independent variants and emit a combined report.

        Each variant has: name, strategy (registry name), optional strategy.params,
        optional overrides (fees, threshold, rebalancing_freq, risk: {...}).
        Reuses _run_strategy_plugins, _aggregate_weights_history,
        _apply_risk_transforms_ts, and _run_vectorbt for parity with the
        single-variant path.
        """
        prices_wide = market_data["prices"]
        base_risk_cfg = dict(params.get("risk", {}) or {})

        variant_results: dict[str, dict[str, Any]] = {}

        for v in variants_cfg:
            vname = str(v["name"])
            strat_cfg = v.get("strategy") or {}
            sname = strat_cfg.get("name") if isinstance(strat_cfg, dict) else str(strat_cfg)
            if not sname:
                raise ValueError(f"Variant {vname!r}: missing strategy.name")
            splugin = variant_plugins.get(vname) or variant_plugins.get(sname)
            if splugin is None:
                raise ValueError(f"Variant {vname!r}: no resolved plugin for strategy {sname!r}")
            strat_params = (strat_cfg.get("params") or {}) if isinstance(strat_cfg, dict) else {}

            # Per-variant overrides
            ov = dict(v.get("overrides", {}) or {})
            v_fees = float(ov.get("fees", fees))
            v_fixed = float(ov.get("fixed_fees", fixed_fees))
            v_slip = float(ov.get("slippage", slippage_val))
            v_freq = ov.get("rebalancing_freq", rebalancing_freq)
            v_thresh = ov.get("threshold", threshold)
            v_risk_cfg = {**base_risk_cfg, **(ov.get("risk", {}) or {})}

            v_strategies_cfg = [{"name": sname, "weight": 1.0, "params": strat_params}]

            # Stage 2: strategy
            s_results = self._run_strategy_plugins(
                [splugin],
                v_strategies_cfg,
                market_data,
                injected_annualize=bars_per_year,
            )

            # Stage 3: aggregate (trivial for single strategy)
            wh = self._aggregate_weights_history(s_results, {"_strategies_cfg": v_strategies_cfg})

            # Stage 4: risk transforms
            wh = self._apply_risk_transforms_ts(wh, v_risk_cfg)

            # Stage 5: align + engine
            common_idx = prices_wide.index.intersection(wh.index)
            common_cols = [c for c in wh.columns if c in prices_wide.columns]
            if not common_cols:
                raise ValueError(f"Variant {vname!r}: no overlapping tickers")
            bt_p = prices_wide.loc[common_idx, common_cols]
            bt_w = wh.loc[common_idx, common_cols]
            min_obs = max(30, int(len(bt_p) * 0.5))
            sufficient_cols = bt_p.columns[bt_p.notna().sum() >= min_obs].tolist()
            bt_p = bt_p[sufficient_cols]
            bt_w = bt_w[sufficient_cols]
            all_nan = bt_p.isna().all(axis=1)
            bt_w = bt_w.where(bt_p.notna(), 0.0)
            bt_p = bt_p.ffill().bfill().loc[~all_nan]
            bt_w = bt_w.loc[~all_nan]

            if engine != "vectorbt":
                raise ValueError(f"Variants flow currently supports engine='vectorbt' only (got {engine!r})")
            res = self._run_vectorbt(
                bt_p,
                bt_w,
                fees=v_fees,
                fixed_fees=v_fixed,
                slippage=v_slip,
                rebalancing_freq=v_freq,
                threshold=v_thresh,
                trading_days=trading_days,
            )

            details_by_strategy = {
                k: (info.get("result") or {}).get("details", {}) or {} for k, info in s_results.items()
            }

            variant_results[vname] = {
                "name": vname,
                "strategy_name": sname,
                "returns": res["returns"],
                "metrics": res["metrics"],
                "portfolio_daily": res["portfolio_daily"],
                "vbt_portfolio": res.get("vbt_portfolio"),
                "weights_history": wh,
                "bt_prices": bt_p,
                "strategy_details": details_by_strategy,
                "config": {
                    "strategy_params": strat_params,
                    "fees": v_fees,
                    "rebalancing_freq": v_freq,
                    "threshold": v_thresh,
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
                res["metrics"].get("total_return", 0),
                res["metrics"].get("sharpe", 0),
                res["metrics"].get("max_drawdown", 0),
            )

        if not variant_results:
            raise ValueError("Variants flow: no variants produced results")

        # Primary variant (first) provides top-level artifacts for backwards compat
        primary_name = next(iter(variant_results))
        primary = variant_results[primary_name]

        a_returns = store.put_parquet("returns", primary["returns"].to_frame("returns").reset_index())
        a_port = store.put_parquet("portfolio_daily", primary["portfolio_daily"].reset_index())
        a_metrics = store.put_json("metrics", primary["metrics"])

        # Per-variant metrics table
        metric_rows = []
        for n, r in variant_results.items():
            row = {"variant": n, "strategy": r["strategy_name"]}
            for k, v_ in r["metrics"].items():
                if isinstance(v_, (int, float)):
                    row[k] = float(v_)
            metric_rows.append(row)
        a_var_metrics = store.put_parquet("variant_metrics", pd.DataFrame(metric_rows))

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
                ),
            )
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
                "variant_metrics": a_var_metrics,
            },
            metrics=flat_metrics,
            notes={
                "kind": "backtest-variants",
                "engine": engine,
                "variants": list(variant_results.keys()),
                "risk_findings": risk_findings,
            },
        )

    # ==================================================================
    # Stage 4: Risk transforms on full time series
    # ==================================================================
    def _apply_risk_transforms_ts(
        self,
        weights: pd.DataFrame,
        risk_cfg: dict[str, Any],
    ) -> pd.DataFrame:
        """Apply tranching, leverage cap, and short clamping to the full
        weights time series."""
        tranches = int(risk_cfg.get("tranches", 1))
        max_leverage = float(risk_cfg.get("max_leverage", 99))
        allow_short = bool(risk_cfg.get("allow_short", False))

        w = weights.copy()

        # Tranching (rolling mean)
        if tranches > 1:
            w = w.rolling(window=tranches, min_periods=1).mean()

        # Clamp negatives
        if not allow_short:
            w = w.clip(lower=0)

        # Leverage cap per row
        gross = w.abs().sum(axis=1)
        scale = (max_leverage / gross).clip(upper=1.0)
        w = w.mul(scale, axis=0)

        return w

    # ==================================================================
    # Engine runners
    # ==================================================================
    def _run_vectorbt(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        *,
        fees: float,
        fixed_fees: float,
        slippage: float,
        rebalancing_freq,
        threshold,
        trading_days: int,
    ) -> dict[str, Any]:
        from quantbox.plugins.backtesting import compute_backtest_metrics, run_vectorbt

        pf = run_vectorbt(
            prices,
            weights,
            rebalancing_freq=rebalancing_freq,
            threshold=threshold,
            fees=fees,
            fixed_fees=fixed_fees,
            slippage=slippage,
        )

        metrics = compute_backtest_metrics(pf, trading_days=trading_days)
        returns = pf.returns()

        # Build portfolio_daily DataFrame
        equity = pf.value()
        if isinstance(equity, pd.DataFrame):
            equity = equity.iloc[:, 0]
        portfolio_daily = pd.DataFrame(
            {
                "date": equity.index,
                "portfolio_value": equity.values,
            }
        ).set_index("date")

        return {
            "returns": returns,
            "metrics": metrics,
            "portfolio_daily": portfolio_daily,
            "vbt_portfolio": pf,
        }

    def _run_rsims(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        funding: pd.DataFrame,
        *,
        fees: float,
        trade_buffer: float,
        initial_cash: float,
        margin: float,
        capitalise_profits: bool,
        equity_basis: str,
        trading_days: int,
    ) -> dict[str, Any]:
        from quantbox.plugins.backtesting import (
            compute_backtest_metrics,
            fixed_commission_backtest_with_funding,
        )

        results_df = fixed_commission_backtest_with_funding(
            prices=prices,
            target_weights=weights,
            funding_rates=funding,
            trade_buffer=trade_buffer,
            initial_cash=initial_cash,
            margin=margin,
            commission_pct=fees,
            capitalise_profits=capitalise_profits,
            equity_basis=equity_basis,
        )

        # Build equity curve (same as quantlab validation script)
        margin_totals = results_df.groupby(results_df.index)["Margin"].sum().to_frame("TotalMargin")
        cash_balance = results_df[results_df["ticker"] == "Cash"][["Value"]].rename(columns={"Value": "Cash"})
        equity_curve = cash_balance.join(margin_totals, how="left")
        equity_curve["TotalMargin"] = equity_curve["TotalMargin"].fillna(0)
        equity_curve["portfolio_value"] = equity_curve["Cash"] + equity_curve["TotalMargin"]

        returns = equity_curve["portfolio_value"].ffill().pct_change(fill_method=None).dropna()
        returns.name = None

        metrics = compute_backtest_metrics(returns, trading_days=trading_days)

        portfolio_daily = equity_curve[["portfolio_value"]].copy()
        portfolio_daily.index.name = "date"

        return {
            "returns": returns,
            "metrics": metrics,
            "portfolio_daily": portfolio_daily,
            "rsims_results": results_df,
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

        Accepts (in priority order):
          1. ``params['frequency']`` — full spec, str or dict
             - dict: ``{'bar_size': '1h', 'calendar': 'NYSE'}``
             - str: ``'1h'`` (calendar falls through to ``market_calendar`` or '24/7')
          2. ``prices.frequency`` + optional ``params['market_calendar']`` shorthand
          3. Default: ``Frequency('1d', '24/7')`` — preserves pre-PR-B crypto-friendly behaviour

        The derived `bars_per_year` is used as the DEFAULT for `trading_days`
        and is injected into each strategy's params as `_pipeline_annualize`,
        so the two cannot silently drift apart.
        """
        explicit = params.get("frequency")
        if explicit is not None:
            return Frequency.parse(explicit)

        bar_size = prices_params.get("frequency", "1d")
        calendar = params.get("market_calendar", "24/7")
        return Frequency.parse({"bar_size": bar_size, "calendar": calendar})
