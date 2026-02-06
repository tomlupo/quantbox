#!/usr/bin/env python3
"""
Step-by-step comparison: quantbox TradingPipeline vs quantlab trading.py

Runs both systems with identical inputs and compares intermediate outputs
at each stage:
  1. Data loading (universe, prices)
  2. Strategy execution (per-strategy weights)
  3. Aggregation (final combined weights)
  4. Risk transforms (leverage, short clamp, tranching)

Usage:
    uv run --with requests --with ccxt --with python-binance \
        python scripts/test_pipeline_stepwise.py
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Setup paths — quantbox-core first
# ------------------------------------------------------------------
QUANTBOX_SRC = Path(__file__).resolve().parent.parent / "packages" / "quantbox-core" / "src"
sys.path.insert(0, str(QUANTBOX_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger("stepwise_test")

# ------------------------------------------------------------------
# Config — mirrors run_trading_multi_strategy.yaml
# ------------------------------------------------------------------
ASOF = "2026-02-05"
STRATEGIES = [
    {"name": "crypto_trend", "weight": 0.6, "params": {"lookback_days": 365}},
]
RISK_CFG = {"tranches": 1, "max_leverage": 1, "allow_short": False}
SYMBOLS = ["BTC", "ETH", "SOL", "BNB"]  # small set for fast test
LOOKBACK_DAYS = 90  # shorter for speed


def sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ==================================================================
# STAGE 1: Data Loading
# ==================================================================
def test_data_loading():
    sep("STAGE 1: Data Loading")

    from quantbox.plugins.datasources.binance_data import BinanceDataFetcher
    from quantbox.plugins.datasources import BinanceDataPlugin

    # ---- A) Via DataPlugin protocol (what TradingPipeline uses) ----
    dp = BinanceDataPlugin()
    universe = dp.load_universe({"symbols": SYMBOLS})
    prices_long = dp.load_prices(universe, ASOF, {"lookback_days": LOOKBACK_DAYS})
    print(f"\n[DataPlugin] Prices long: {prices_long.shape}, cols={list(prices_long.columns)}")
    print(f"  Date range: {prices_long['date'].min()} → {prices_long['date'].max()}")
    print(f"  NaN in close: {prices_long['close'].isna().sum()}")

    # ---- B) Via BinanceDataFetcher directly (what quantlab uses) ----
    fetcher = BinanceDataFetcher(quote_asset="USDT")
    raw = fetcher.get_market_data(tickers=SYMBOLS, lookback_days=LOOKBACK_DAYS, end_date=ASOF)
    prices_wide = raw["prices"]
    volume_wide = raw["volume"]
    market_cap_wide = raw["market_cap"]

    print(f"\n[Fetcher direct] Prices wide: {prices_wide.shape}")
    print(f"  Volume wide: {volume_wide.shape}")
    print(f"  Market cap wide: {market_cap_wide.shape}")
    print(f"  NaN in prices: {prices_wide.isna().sum().sum()}")

    print("\nLast 3 rows of wide prices:")
    print(prices_wide.tail(3).to_string())

    # ---- C) Compare: DataPlugin long → wide should match fetcher wide ----
    plugin_wide = prices_long.pivot_table(index="date", columns="symbol", values="close")
    diff = (plugin_wide - prices_wide).abs()
    max_diff = diff.max().max() if not diff.empty else 0
    print(f"\nDataPlugin vs Fetcher price diff: max={max_diff:.2e}")

    # Use fetcher data (has market_cap) for strategy testing
    market_data = {
        "prices": prices_wide,
        "volume": volume_wide,
        "market_cap": market_cap_wide,
        "universe": universe,
    }
    return market_data, prices_long


# ==================================================================
# STAGE 2: Strategy Execution
# ==================================================================
def test_strategies(market_data: dict):
    sep("STAGE 2: Strategy Execution")

    results = {}
    for strat_cfg in STRATEGIES:
        name = strat_cfg["name"]
        weight = strat_cfg["weight"]
        strat_params = strat_cfg.get("params", {})

        module = importlib.import_module(f"quantbox.plugins.strategies.{name}")
        result = module.run(data=market_data, params=strat_params)

        weights_df = result.get("weights", pd.DataFrame())
        print(f"\n--- Strategy: {name} (account weight: {weight}) ---")
        print(f"  Weights shape: {weights_df.shape}")
        print(f"  Columns nlevels: {weights_df.columns.nlevels}")

        # Normalize multi-level columns (same as both quantlab & quantbox)
        if isinstance(weights_df, pd.DataFrame) and weights_df.columns.nlevels > 1:
            weights_df = weights_df.T.groupby("ticker").sum().T
            result["weights"] = weights_df

        last_row = weights_df.iloc[-1] if isinstance(weights_df, pd.DataFrame) else weights_df
        nonzero = last_row[last_row != 0].sort_values(ascending=False)
        print(f"  Non-zero positions (last row): {len(nonzero)}")
        for sym, w in nonzero.head(10).items():
            print(f"    {sym:6s}: {w:+.6f}")

        results[name] = {"result": result, "weight": weight}

    return results


# ==================================================================
# STAGE 3: Aggregation
# ==================================================================
def test_aggregation(strategy_results: dict):
    sep("STAGE 3: Aggregation")

    names = list(strategy_results.keys())

    # --- quantlab method (inline in trading.py lines 432-455) ---
    strategies_asset_weights = pd.concat(
        [strategy_results[k]["result"]["weights"] for k in names],
        axis=1,
        keys=names,
        names=["strategy"],
    )
    account_strategies_weights = pd.Series(
        [strategy_results[k]["weight"] for k in names],
        index=pd.Index(names, name="strategy"),
    )
    account_asset_weights = strategies_asset_weights.mul(
        account_strategies_weights, level="strategy"
    ).droplevel(0, axis=1)
    ql_final = account_asset_weights.iloc[-1]
    ql_final = ql_final.sort_values(ascending=False)

    print("\nQuantlab aggregation (last row):")
    for sym, w in ql_final[ql_final != 0].head(10).items():
        print(f"  {sym:6s}: {w:+.6f}")

    # --- quantbox method (_aggregate_strategies) ---
    from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline

    tp = TradingPipeline()
    qb_final = tp._aggregate_strategies(
        strategy_results, {"strategies": STRATEGIES}
    )

    print("\nQuantbox aggregation:")
    for sym, w in sorted(qb_final.items(), key=lambda x: -x[1])[:10]:
        if w != 0:
            print(f"  {sym:6s}: {w:+.6f}")

    # --- Compare ---
    print("\nDifferences:")
    all_syms = set(ql_final.index) | set(qb_final.keys())
    max_diff = 0.0
    for sym in sorted(all_syms):
        ql_w = float(ql_final.get(sym, 0))
        qb_w = float(qb_final.get(str(sym), 0))
        diff = abs(ql_w - qb_w)
        max_diff = max(max_diff, diff)
        if diff > 1e-8:
            print(f"  {sym:6s}: QL={ql_w:+.6f}  QB={qb_w:+.6f}  delta={diff:.2e}")
    if max_diff < 1e-8:
        print("  EXACT MATCH")
    else:
        print(f"  Max diff: {max_diff:.2e}")

    return ql_final, qb_final


# ==================================================================
# STAGE 4: Risk Transforms
# ==================================================================
def test_risk_transforms(strategy_results: dict, ql_aggregated: pd.Series):
    sep("STAGE 4: Risk Transforms")

    # --- quantlab method (trading.py lines 445-455) ---
    tranches = RISK_CFG["tranches"]
    max_leverage = RISK_CFG["max_leverage"]
    allow_short = RISK_CFG["allow_short"]

    names = list(strategy_results.keys())
    strategies_asset_weights = pd.concat(
        [strategy_results[k]["result"]["weights"] for k in names],
        axis=1,
        keys=names,
        names=["strategy"],
    )
    account_strategies_weights = pd.Series(
        [strategy_results[k]["weight"] for k in names],
        index=pd.Index(names, name="strategy"),
    )
    account_asset_weights = strategies_asset_weights.mul(
        account_strategies_weights, level="strategy"
    ).droplevel(0, axis=1)

    ql_final = account_asset_weights.rolling(window=tranches).mean().iloc[-1]
    if ql_final.sum() > max_leverage:
        ql_final = ql_final / ql_final.sum() * max_leverage
    if not allow_short:
        ql_final = ql_final.clip(lower=0)
    ql_final = ql_final.sort_values(ascending=False)
    ql_final = ql_final[ql_final != 0]

    print("\nQuantlab risk-transformed weights:")
    for sym, w in ql_final.head(10).items():
        print(f"  {sym:6s}: {w:+.6f}")
    print(f"  Sum={ql_final.sum():.6f}  Gross={ql_final.abs().sum():.6f}")

    # --- quantbox method ---
    from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline

    tp = TradingPipeline()
    qb_agg = tp._aggregate_strategies(
        strategy_results, {"strategies": STRATEGIES}
    )
    qb_final = tp._apply_risk_transforms(
        qb_agg,
        strategy_results,
        {"strategies": STRATEGIES, "risk": RISK_CFG},
    )

    print("\nQuantbox risk-transformed weights:")
    for sym, w in sorted(qb_final.items(), key=lambda x: -x[1])[:10]:
        print(f"  {sym:6s}: {w:+.6f}")
    print(f"  Sum={sum(qb_final.values()):.6f}  Gross={sum(abs(v) for v in qb_final.values()):.6f}")

    # --- Compare ---
    print("\nDifferences:")
    all_syms = set(ql_final.index) | set(qb_final.keys())
    max_diff = 0.0
    for sym in sorted(all_syms):
        ql_w = float(ql_final.get(sym, 0))
        qb_w = float(qb_final.get(str(sym), 0))
        diff = abs(ql_w - qb_w)
        max_diff = max(max_diff, diff)
        if diff > 1e-8:
            print(f"  {sym:6s}: QL={ql_w:+.6f}  QB={qb_w:+.6f}  delta={diff:.2e}")
    if max_diff < 1e-8:
        print("  EXACT MATCH")
    else:
        print(f"  Max diff: {max_diff:.2e}")

    return ql_final, qb_final


# ==================================================================
# Main
# ==================================================================
def main():
    print("Step-by-step comparison: quantbox vs quantlab logic")
    print(f"Symbols: {SYMBOLS}")
    print(f"Asof: {ASOF}")
    print(f"Strategies: {[s['name'] for s in STRATEGIES]}")
    print(f"Risk: {RISK_CFG}")

    market_data, prices_long = test_data_loading()
    strategy_results = test_strategies(market_data)
    ql_agg, qb_agg = test_aggregation(strategy_results)
    ql_risk, qb_risk = test_risk_transforms(strategy_results, ql_agg)

    sep("SUMMARY")
    print("All stages completed. Review per-stage diffs above.")


if __name__ == "__main__":
    main()
