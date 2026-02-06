"""Full trading pipeline plugin.

Orchestrates: strategy execution -> aggregation -> risk transforms ->
order generation -> execution -> artifact storage.

Ported from the quantlab ``trading.py`` workflow, ``orders.py``, and
``portfolio.py`` into a single PipelinePlugin that the quantbox runner
can invoke via ``pipeline.run()``.
"""
from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal, getcontext
from typing import Any, Callable, Dict, List, Optional, Tuple

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


def _adjust_quantity(qty: float, step_size: float) -> float:
    """Round *qty* down to the nearest *step_size* (Binance-style)."""
    if step_size <= 0:
        return qty
    step_str = f"{step_size:.8f}"
    decimal_places = step_str.rstrip("0").split(".")[-1]
    precision = len(decimal_places)
    getcontext().rounding = ROUND_DOWN
    return float(Decimal(qty).quantize(Decimal("1." + "0" * precision)))


def _get_lot_size_and_min_notional(
    symbol_info: Optional[Dict],
) -> Tuple[float, float, float]:
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


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


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
        params: Dict[str, Any],
        data: DataPlugin,
        store: ArtifactStore,
        broker: Optional[BrokerPlugin],
        risk: List[RiskPlugin],
        strategies: Optional[List[StrategyPlugin]] = None,
        rebalancer: Optional[RebalancingPlugin] = None,
        **kwargs,
    ) -> RunResult:
        if mode in ("paper", "live") and broker is None:
            raise ValueError("broker_required_for_paper_or_live")

        # Resolve strategies config: from injected plugins or from pipeline params
        strategies_cfg = params.get("_strategies_cfg", params.get("strategies", []))
        if not strategies and not strategies_cfg:
            raise ValueError("params.strategies is required and must be non-empty")

        # Resolve aggregator and rebalancer config from runner injection
        aggregator_cfg = params.get("_aggregator_cfg", {})
        rebalancer_cfg = params.get("_rebalancer_cfg", {})

        # --- Stage 1: Universe & Prices ---
        universe_params = params.get("universe", {})
        prices_params = params.get("prices", {"lookback_days": 365})
        universe = data.load_universe(universe_params)

        # Token policy filtering (universe scope)
        token_policy = self._load_token_policy(universe_params)
        token_policy_notes: Dict[str, Any] = {}
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

            universe = universe[
                universe["symbol"].isin(allowed)
            ].reset_index(drop=True)
            logger.info(
                "TokenPolicy: %d/%d symbols allowed (mode=%s)",
                len(allowed),
                len(all_symbols),
                token_policy.mode,
            )

        market_data_dict = data.load_market_data(universe, asof, prices_params)
        store.put_parquet("universe", universe)
        # Store prices as wide-format parquet
        prices_wide = market_data_dict.get("prices", pd.DataFrame())
        if not prices_wide.empty:
            store.put_parquet("prices", prices_wide.reset_index() if isinstance(prices_wide.index, pd.DatetimeIndex) else prices_wide)
        else:
            store.put_parquet("prices", pd.DataFrame())

        # Build data dict for strategies (quantlab convention)
        market_data = self._build_market_data(market_data_dict, universe)

        # --- Stage 2: Strategy Execution ---
        if strategies:
            strategy_results = self._run_strategy_plugins(
                strategies, strategies_cfg, market_data,
            )
        else:
            strategy_results = self._run_strategies(market_data, strategies_cfg, params)

        # Save per-strategy weights
        strat_weights_records: List[Dict[str, Any]] = []
        for sname, sinfo in strategy_results.items():
            w = sinfo["result"].get("weights")
            if w is not None and not w.empty:
                last_row = w.iloc[-1] if isinstance(w, pd.DataFrame) else w
                for ticker, wt in last_row.items():
                    strat_weights_records.append(
                        {"strategy": sname, "symbol": str(ticker), "weight": float(wt)}
                    )
        a_strat_w = store.put_parquet(
            "strategy_weights", pd.DataFrame(strat_weights_records)
        )

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
        agg_records = [
            {"symbol": str(k), "weight": float(v)} for k, v in final_weights.items()
        ]
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
            final_weights = self._apply_risk_transforms(
                final_weights, strategy_results, params
            )

        if mode == "backtest" or broker is None:
            # In backtest mode, just save targets with no execution
            targets = pd.DataFrame(
                [
                    {"symbol": s, "weight": w, "asof": asof}
                    for s, w in final_weights.items()
                ]
            )
            a_targets = store.put_parquet("targets", targets)
            empty_orders = pd.DataFrame(columns=["symbol", "side", "qty", "price", "asof"])
            a_orders = store.put_parquet("orders", empty_orders)
            a_fills = store.put_parquet("fills", pd.DataFrame(columns=["symbol", "side", "qty", "price"]))
            a_rebal = store.put_parquet("rebalancing", pd.DataFrame())
            portfolio_daily = pd.DataFrame(
                [{"asof": asof, "cash_usd": 0.0, "portfolio_value_usd": 0.0}]
            )
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
        risk_findings: List[Dict[str, Any]] = []
        for rp in risk:
            try:
                risk_findings.extend(rp.check_targets(targets, params.get("risk", {})))
                exec_orders = orders_df[orders_df.get("Executable", pd.Series(dtype=bool))].copy() if "Executable" in orders_df.columns else orders_df
                risk_findings.extend(rp.check_orders(exec_orders, params.get("risk", {})))
            except Exception as exc:
                logger.warning("Risk check failed: %s", exc)

        # --- Stage 7: Execution ---
        trading_enabled = bool(params.get("trading_enabled", True))
        execution_report = self._execute_orders(
            broker=broker,
            orders_df=orders_df,
            stable_coin=stable_coin,
            trading_enabled=trading_enabled,
            mode=mode,
        )

        fills_data = []
        for detail in execution_report.get("orders_details", []):
            if detail.get("status") == "FILLED":
                fills_data.append({
                    "symbol": detail.get("symbol", ""),
                    "side": str(detail.get("side", detail.get("action", ""))).lower(),
                    "qty": float(detail.get("executed_quantity", 0)),
                    "price": float(detail.get("executed_price", 0)),
                })
        fills = pd.DataFrame(fills_data) if fills_data else pd.DataFrame(columns=["symbol", "side", "qty", "price"])
        a_fills = store.put_parquet("fills", fills)

        # Apply funding rates to open positions (if broker supports it)
        funding_charge = 0.0
        if broker is not None and hasattr(broker, "apply_funding"):
            funding_charge = broker.apply_funding()
            logger.info("Applied funding charge: %.2f", funding_charge)

        # Portfolio snapshot
        portfolio_value_post = total_value
        cash_usd_post = 0.0
        try:
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
        except Exception:
            pass

        portfolio_daily = pd.DataFrame([{
            "asof": asof,
            "cash_usd": float(cash_usd_post),
            "portfolio_value_usd": float(portfolio_value_post),
        }])
        a_port = store.put_parquet("portfolio_daily", portfolio_daily)

        # Collect fee/funding metrics from broker
        cumulative_fees = 0.0
        if broker is not None and hasattr(broker, "_cumulative_fees"):
            cumulative_fees = float(broker._cumulative_fees)

        # --- Stage 8: Artifacts ---
        # Collect broker cost config for artifact
        broker_costs: Dict[str, Any] = {}
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
        a_trade = store.put_json("trade_history", {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "trades": artifact_payload.get("execution_summary", {}).get("executed_orders", []),
        })

        metrics = {
            "n_strategies": float(len(strategy_results)),
            "n_assets": float(len(final_weights)),
            "portfolio_value_usd_pre": float(total_value),
            "portfolio_value_usd_post": float(portfolio_value_post),
            "n_orders": float(len(orders_df)),
            "n_fills": float(len(fills)),
            "total_executed": float(execution_report.get("summary", {}).get("total_executed", 0)),
            "total_failed": float(execution_report.get("summary", {}).get("total_failed", 0)),
            "funding_charge": float(funding_charge),
            "cumulative_fees": cumulative_fees,
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
                **token_policy_notes,
            },
        )

    # ==================================================================
    # Stage 1 helper: build market data dict for strategies
    # ==================================================================
    def _build_market_data(
        self,
        market_data_dict: Dict[str, pd.DataFrame],
        universe: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Build the data dict that strategies expect.

        Takes the wide-format dict from ``DataPlugin.load_market_data()``
        and adds universe + defaults for missing keys.
        """
        result: Dict[str, Any] = {"universe": universe}
        result.update(market_data_dict)
        for key in ("prices", "volume", "market_cap", "funding_rates"):
            result.setdefault(key, pd.DataFrame())
        return result

    # ==================================================================
    # Stage 2: Strategy execution
    # ==================================================================
    def _run_strategies(
        self,
        market_data: Dict[str, Any],
        strategies_cfg: List[Dict[str, Any]],
        pipeline_params: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Import and run each strategy, collecting results."""
        results: Dict[str, Dict[str, Any]] = {}

        for strat_cfg in strategies_cfg:
            name = strat_cfg["name"]
            weight = float(strat_cfg.get("weight", 1.0))
            strat_params = strat_cfg.get("params", {})

            try:
                module = importlib.import_module(
                    f"quantbox.plugins.strategies.{name}"
                )
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
        strategy_plugins: List[StrategyPlugin],
        strategies_cfg: List[Dict[str, Any]],
        market_data: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Run injected StrategyPlugin instances, collecting results."""
        results: Dict[str, Dict[str, Any]] = {}

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
        strategy_results: Dict[str, Dict[str, Any]],
        params: Dict[str, Any],
    ) -> Dict[str, float]:
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
            combined = pd.concat(
                weight_dfs, axis=1, keys=names[:len(weight_dfs)], names=["strategy"]
            )
            acct_w = pd.Series(
                account_weights,
                index=pd.Index(names[:len(weight_dfs)], name="strategy"),
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
            agg: Dict[str, float] = {}
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
        weights: Dict[str, float],
        strategy_results: Dict[str, Dict[str, Any]],
        params: Dict[str, Any],
    ) -> Dict[str, float]:
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
                        account_weights.append(
                            float(weight_overrides.get(sname, sinfo["weight"]))
                        )

                if len(weight_dfs) == 1:
                    full_ts = weight_dfs[0] * account_weights[0]
                else:
                    combined = pd.concat(
                        weight_dfs,
                        axis=1,
                        keys=names[:len(weight_dfs)],
                        names=["strategy"],
                    )
                    acct_w = pd.Series(
                        account_weights,
                        index=pd.Index(names[:len(weight_dfs)], name="strategy"),
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
            logger.warning(
                "Leverage %.4f exceeds max_leverage %.1f, scaling down", gross, max_leverage
            )
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
        weights: Dict[str, float],
        capital_at_risk: float,
        stable_coin: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
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
        current_holdings: Dict[str, float] = {}
        if positions_df is not None and not positions_df.empty:
            for _, row in positions_df.iterrows():
                sym = str(row.get("symbol", ""))
                qty = float(row.get("qty", 0))
                if qty != 0:
                    current_holdings[sym] = qty

        # Get prices via broker market snapshot
        all_symbols = sorted(set(list(weights.keys()) + list(current_holdings.keys())))
        all_symbols = [s for s in all_symbols if s not in exclusions]

        price_map: Dict[str, Optional[float]] = {}
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

        def get_price(asset: str) -> Optional[float]:
            return price_map.get(asset)

        # Portfolio value
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
        target_positions: Dict[str, float] = {}
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
        current_holdings: Dict[str, float],
        target_positions: Dict[str, float],
        get_price: Callable[[str], Optional[float]],
        total_value: float,
        strategy_weights: Dict[str, float],
        exclusions: List[str],
    ) -> pd.DataFrame:
        """Build the rebalancing analysis DataFrame.

        Ported from quantlab ``orders.py:generate_rebalancing_dataframe()``.
        """
        assets = sorted(set(current_holdings.keys()) | set(target_positions.keys()))
        assets = [a for a in assets if a not in exclusions]

        rows: List[Dict[str, Any]] = []
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

            rows.append({
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
            })

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
            return pd.DataFrame(columns=[
                "Asset", "Symbol", "Action", "Raw Quantity", "Adjusted Quantity",
                "Price", "Notional Value", "Min Notional", "Min Qty", "Step Size",
                "Scaling Factor", "Order Status", "Reason", "Executable",
            ])

        order_records: List[Dict[str, Any]] = []

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
                    action == "sell"
                    and row.get("Target Weight", 0) == 0
                    and row.get("Current Quantity", 0) > 0
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
                            status = "Below min notional"
                            reason = f"Notional {notional_value:.4f} < min_notional {min_notional:.4f}"
                            adjusted_qty = 0.0
                        elif min_qty > 0 and adjusted_qty < min_qty:
                            status = "Below min qty"
                            reason = f"Qty {adjusted_qty:.8f} < min_qty {min_qty:.8f}"
                            adjusted_qty = 0.0
                        else:
                            status = "To be placed"
                            reason = ""
                    elif action == "buy":
                        status = "Pending scaling"
                        reason = ""

            order_records.append({
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
            })

        # --- Buy scaling (ported from quantlab) ---
        buy_indices = [
            i for i, rec in enumerate(order_records)
            if rec["Action"] == "Buy" and rec["Order Status"] == "Pending scaling"
        ]
        total_buy_value = sum(
            order_records[i]["Raw Quantity"] * (order_records[i]["Price"] or 0)
            for i in buy_indices
        )
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
                scaling_factor, scaling_factor_min,
            )

        columns = [
            "Asset", "Symbol", "Action", "Raw Quantity", "Adjusted Quantity",
            "Price", "Notional Value", "Min Notional", "Min Qty", "Step Size",
            "Scaling Factor", "Order Status", "Reason",
        ]
        order_df = pd.DataFrame(order_records, columns=columns)
        order_df["Executable"] = (order_df["Adjusted Quantity"] > 0) & (order_df["Order Status"] == "To be placed")
        return order_df

    # ==================================================================
    # Stage 7: Execution
    # ==================================================================
    def _execute_orders(
        self,
        broker: BrokerPlugin,
        orders_df: pd.DataFrame,
        stable_coin: str,
        trading_enabled: bool,
        mode: str,
    ) -> Dict[str, Any]:
        """Execute orders via broker with sell-before-buy ordering.

        Ported from quantlab ``orders.py:execute_orders()``.
        """
        report: Dict[str, Any] = {
            "executed_orders": [],
            "failed_orders": [],
            "summary": {
                "total_executed": 0,
                "total_failed": 0,
                "total_value": 0.0,
                "total_cost": 0.0,
            },
            "orders_details": [],
        }

        if orders_df is None or orders_df.empty or not trading_enabled:
            return report

        executable = orders_df[orders_df.get("Executable", pd.Series(dtype=bool))].copy() if "Executable" in orders_df.columns else orders_df
        if executable.empty:
            return report

        # Sell before buy: sort by Action descending (Sell > Buy)
        executable = executable.sort_values(by="Action", ascending=False)

        # Convert to broker-compatible format
        broker_orders_data: List[Dict[str, Any]] = []
        for _, row in executable.iterrows():
            action = str(row.get("Action", "")).lower()
            side = "sell" if action == "sell" else "buy"
            broker_orders_data.append({
                "symbol": str(row.get("Asset", "")),
                "side": side,
                "qty": float(row.get("Adjusted Quantity", 0)),
                "price": float(row.get("Price", 0)),
            })

        broker_orders = pd.DataFrame(broker_orders_data)
        if broker_orders.empty:
            return report

        try:
            fills = broker.place_orders(broker_orders)
        except Exception as exc:
            logger.error("Broker place_orders failed: %s", exc)
            for _, row in broker_orders.iterrows():
                report["orders_details"].append({
                    "symbol": row["symbol"],
                    "action": row["side"],
                    "status": "FAILED",
                    "error": str(exc),
                })
                report["summary"]["total_failed"] += 1
            return report

        # Process fills
        if fills is not None and not fills.empty:
            for _, fill_row in fills.iterrows():
                detail = {
                    "symbol": str(fill_row.get("symbol", "")),
                    "action": str(fill_row.get("side", "")),
                    "side": str(fill_row.get("side", "")),
                    "executed_quantity": float(fill_row.get("qty", 0)),
                    "executed_price": float(fill_row.get("price", 0)),
                    "fee": float(fill_row.get("fee", 0) or 0),
                    "status": "FILLED",
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

        report["summary"]["total_value"] = sum(
            d.get("executed_quantity", 0) * d.get("executed_price", 0)
            for d in report["orders_details"]
            if d.get("status") == "FILLED"
        )

        return report

    # ==================================================================
    # Artifact payload builder (for publisher plugins)
    # ==================================================================
    def _build_artifact_payload(
        self,
        rebalancing_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        execution_report: Dict[str, Any],
        final_weights: Dict[str, float],
        total_value: float,
        mode: str,
        funding_charge: float = 0.0,
        cumulative_fees: float = 0.0,
        broker_costs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build structured artifact payload for publishers.

        Ported from quantlab ``trading.py:_build_artifact_payload()``.
        """
        paper_trading = mode == "paper"

        # Rebalancing table
        rebalancing_table: List[Dict[str, Any]] = []
        if rebalancing_df is not None and not rebalancing_df.empty:
            col_map = {
                "Asset": "asset", "Current Quantity": "current_qty",
                "Current Value": "current_value", "Current Weight": "current_weight",
                "Target Quantity": "target_qty", "Target Value": "target_value",
                "Target Weight": "target_weight", "Weight Delta": "weight_delta",
                "Delta Quantity": "delta_qty", "Price": "price",
                "Trade Action": "trade_action",
            }
            for _, row in rebalancing_df.iterrows():
                entry: Dict[str, Any] = {}
                for src, dst in col_map.items():
                    val = row.get(src)
                    if val is not None and isinstance(val, (int, float, np.floating)):
                        entry[dst] = round(float(val), 6)
                    elif val is not None:
                        entry[dst] = str(val)
                rebalancing_table.append(entry)

        # Execution summary
        summary = execution_report.get("summary", {})
        executed_orders: List[Dict[str, Any]] = []
        failed_orders: List[Dict[str, Any]] = []
        total_order_fees = 0.0
        for detail in execution_report.get("orders_details", []):
            if detail.get("status") == "FILLED":
                order_fee = float(detail.get("fee", 0) or 0)
                total_order_fees += order_fee
                executed_orders.append({
                    "symbol": str(detail.get("symbol", "")),
                    "action": str(detail.get("action", "")),
                    "quantity": detail.get("executed_quantity"),
                    "executed_price": detail.get("executed_price"),
                    "reference_price": detail.get("reference_price"),
                    "spread_bps": round(float(detail.get("spread_pct", 0) or 0) * 10000, 1),
                    "fee": round(order_fee, 4),
                    "status": "FILLED",
                })
            elif detail.get("status") == "FAILED":
                failed_orders.append({
                    "symbol": str(detail.get("symbol", "")),
                    "action": str(detail.get("action", "")),
                    "error": str(detail.get("error", "")),
                    "status": "FAILED",
                })

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
                "fees_this_run": round(total_order_fees, 4),
                "cumulative_fees": round(cumulative_fees, 4),
                "funding_charge": round(funding_charge, 4),
                **(broker_costs or {}),
            },
        }

    # ==================================================================
    # Token policy helpers
    # ==================================================================
    def _load_token_policy(self, universe_params: Dict[str, Any]):
        """Create TokenPolicy from universe params or config file.

        Returns ``None`` when no token-policy configuration is present
        (backward-compatible).
        """
        from quantbox.plugins.trading.quantlab.token_policy import TokenPolicy

        tp_cfg = universe_params.get("token_policy")
        tp_file = universe_params.get("token_policy_file")

        if tp_cfg:
            return TokenPolicy.from_dict({"token_policy": tp_cfg})
        elif tp_file:
            return TokenPolicy.from_config(tp_file)
        return None

    def _detect_new_tokens(self, policy, universe: pd.DataFrame) -> list:
        """Build a minimal rankings DF from universe and run detection."""
        if not policy.alert_on_new or universe.empty:
            return []

        symbols = universe["symbol"].tolist()
        rankings_data = [
            {"symbol": sym, "rank": rank, "market_cap": 0}
            for rank, sym in enumerate(symbols, 1)
        ]
        rankings_df = pd.DataFrame(rankings_data)
        return policy.detect_new_tokens(rankings_df, policy.top_n_monitor)
