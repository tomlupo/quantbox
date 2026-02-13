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

logger = logging.getLogger(__name__)


@dataclass
class BacktestPipeline:
    meta = PluginMeta(
        name="backtest.pipeline.v1",
        kind="pipeline",
        version="0.1.0",
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
                    "description": "Rebalancing frequency: int, string (1D/1W/1M), or null for buy-and-hold.",
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
                    "default": 0.05,
                    "description": "Maintenance margin rate (rsims only).",
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
        trading_days = int(params.get("trading_days", 365))

        # --- Stage 1: Universe & Market Data ---
        universe_params = params.get("universe", {})
        prices_params = params.get("prices", {"lookback_days": 365})
        universe = data.load_universe(universe_params)
        market_data_dict = data.load_market_data(universe, asof, prices_params)

        store.put_parquet("universe", universe)

        market_data: dict[str, Any] = {"universe": universe}
        market_data.update(market_data_dict)
        for key in ("prices", "volume", "market_cap", "funding_rates"):
            market_data.setdefault(key, pd.DataFrame())

        prices_wide = market_data["prices"]
        logger.info(
            "Data loaded: %d dates x %d tickers, range %s to %s",
            prices_wide.shape[0],
            prices_wide.shape[1],
            prices_wide.index[0] if len(prices_wide) else "?",
            prices_wide.index[-1] if len(prices_wide) else "?",
        )

        # --- Stage 2: Strategy Execution ---
        strategies_cfg = params.get("_strategies_cfg", params.get("strategies", []))
        if strategies:
            strategy_results = self._run_strategy_plugins(strategies, strategies_cfg, market_data)
        else:
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

        # Drop rows with any NaN prices (can't simulate without prices)
        valid_rows = bt_prices.notna().all(axis=1)
        bt_prices = bt_prices.loc[valid_rows]
        bt_weights = bt_weights.loc[valid_rows]

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
                margin=float(params.get("margin", 0.05)),
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
        results: dict[str, dict[str, Any]] = {}
        for i, strat in enumerate(strategy_plugins):
            strat_cfg = strategies_cfg[i] if i < len(strategies_cfg) else {}
            weight = float(strat_cfg.get("weight", 1.0))
            strat_params = strat_cfg.get("params", {})

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

        returns = equity_curve["portfolio_value"].pct_change().dropna()
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
