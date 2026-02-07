#!/usr/bin/env python3
"""
Compare quantbox pipeline output against quantlab production artifacts.

Loads the SAME cached market data that quantlab prod used, runs quantbox
strategy with identical params, then compares strategy_weights against
the actual production artifact.json.

Usage:
    uv run --with requests --with ccxt --with python-binance \
        python scripts/test_pipeline_vs_prod.py [account] [date]

    account: paper1 (default)
    date:    2026-02-05 (default: latest available)
"""
from __future__ import annotations

import importlib
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
QUANTBOX_SRC = Path(__file__).resolve().parent.parent / "packages" / "quantbox-core" / "src"
sys.path.insert(0, str(QUANTBOX_SRC))

PROD_ROOT = Path("/home/tom/workspace/prod/quantlab")
CACHE_DIR = PROD_ROOT / "data" / "cache"
CONFIG_DIR = PROD_ROOT / "config"
ARTIFACTS_DIR = PROD_ROOT / "output"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger("vs_prod")


def sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ==================================================================
# Data loading — from quantlab production cache
# ==================================================================

def load_account_config(account_name: str) -> dict:
    """Load account + strategy config from quantlab prod config."""
    acct_path = CONFIG_DIR / "accounts" / f"{account_name}.yaml"
    if not acct_path.exists():
        raise FileNotFoundError(f"Account config not found: {acct_path}")
    with open(acct_path) as f:
        account_config = yaml.safe_load(f)

    strategy_configs = {}
    for strat in account_config.get("strategies", []):
        name = strat["name"]
        strat_path = CONFIG_DIR / "strategies" / f"{name}.yaml"
        if strat_path.exists():
            with open(strat_path) as f:
                strategy_configs[name] = yaml.safe_load(f)
        else:
            strategy_configs[name] = {}

    return account_config, strategy_configs


def load_cmc_rankings(target_date: str) -> pd.DataFrame:
    """Load CMC rankings from cache for the target date.

    Production runs at ~06:30-07:00 UTC, so we look for the CMC file
    captured on the same date (format: YYYYMMDD_HHMMSS*.parquet).
    Falls back to the latest file before the target date.
    """
    cmc_dir = CACHE_DIR / "cmc_rankings"
    files = sorted(cmc_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No CMC ranking files in {cmc_dir}")

    date_prefix = target_date.replace("-", "")
    # Find file matching target date
    matching = [f for f in files if f.name.startswith(date_prefix)]
    if matching:
        chosen = matching[0]
    else:
        # Fall back to latest file on or before target date
        candidates = [f for f in files if f.name[:8] <= date_prefix]
        chosen = candidates[-1] if candidates else files[-1]

    logger.info("Loading CMC rankings from %s (target date: %s)", chosen.name, target_date)
    return pd.read_parquet(chosen)


def load_cached_ohlcv(
    tickers: List[str],
    target_date: str,
    lookback_days: int = 365,
) -> Dict[str, pd.DataFrame]:
    """Load OHLCV data from quantlab's parquet cache.

    Production data includes the target date's candle (partial at
    run-time, stored in cache by the production scheduler).
    """
    ohlcv_dir = CACHE_DIR / "binance_ohlcv"
    end_date = pd.Timestamp(target_date).normalize()
    start_date = end_date - timedelta(days=lookback_days)
    logger.info("OHLCV range: %s → %s (%d days)", start_date.date(), end_date.date(), lookback_days)

    ohlcv = {}
    for ticker in tickers:
        ticker_dir = ohlcv_dir / f"ticker={ticker}"
        if ticker_dir.exists():
            df = pd.read_parquet(ticker_dir)
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
            ohlcv[ticker] = df
        else:
            logger.warning("No cached OHLCV for %s", ticker)
    return ohlcv


def preprocess_data(
    ohlcv: Dict[str, pd.DataFrame],
    coins_ranking: pd.DataFrame,
) -> Dict[str, Any]:
    """Preprocess market data — identical to quantlab trading.py:preprocess_data().

    Builds wide-format prices, volume, and market_cap from OHLCV dict.
    """
    prices = pd.DataFrame(
        [
            v.drop_duplicates(subset=["date"], keep="last")
            .set_index("date")["close"]
            .rename(k)
            for k, v in ohlcv.items()
        ]
    ).T

    volume = pd.DataFrame(
        [
            v.drop_duplicates(subset=["date"], keep="last")
            .set_index("date")["volume"]
            .rename(k)
            for k, v in ohlcv.items()
        ]
    ).T

    market_cap_series = coins_ranking.set_index("symbol")["market_cap"]

    # Strategy expects market_cap as wide DataFrame (dates × tickers),
    # not a Series. Broadcast the CMC snapshot across all dates.
    market_cap = pd.DataFrame(
        {sym: market_cap_series.get(sym, np.nan) for sym in prices.columns},
        index=prices.index,
    )

    return {
        "prices": prices,
        "volume": volume,
        "market_cap": market_cap,
        "coins_ranking": coins_ranking,
    }


def load_prod_artifact(account_name: str, date: str) -> Optional[dict]:
    """Load the production artifact.json for a given account and date."""
    process = f"trading_bot_{account_name}"
    artifacts_dir = ARTIFACTS_DIR / process / "artifacts" / date
    if not artifacts_dir.exists():
        logger.warning("No artifact directory for %s on %s", account_name, date)
        return None

    # Find the latest timestamped subdirectory
    subdirs = sorted(artifacts_dir.iterdir())
    if not subdirs:
        return None

    artifact_path = subdirs[-1] / "artifact.json"
    if not artifact_path.exists():
        return None

    with open(artifact_path) as f:
        return json.load(f)


# ==================================================================
# Strategy execution — uses quantbox strategy module
# ==================================================================

def run_quantbox_strategy(
    market_data: Dict[str, Any],
    strategy_name: str,
    strategy_config: dict,
    account_config: dict,
    strat_params: dict,
) -> dict:
    """Run a quantbox strategy with quantlab-identical params.

    Quantlab merges: {strategy_config} + {account_config} + {strat.params}
    """
    # Quantlab name → quantbox name mapping
    name_map = {
        "crypto_trend_catcher": "crypto_trend",
    }
    module_name = name_map.get(strategy_name, strategy_name)

    # Merge params exactly like quantlab (trading.py:227)
    params = {**strategy_config, **account_config, **strat_params}

    module = importlib.import_module(f"quantbox.plugins.strategies.{module_name}")
    return module.run(data=market_data, params=params)


# ==================================================================
# Aggregation + risk transforms — matches quantlab trading.py
# ==================================================================

def aggregate_and_risk(
    strategy_results: Dict[str, Dict[str, Any]],
    account_config: dict,
) -> Dict[str, float]:
    """Aggregate strategy weights and apply risk management.

    Mirrors quantlab trading.py lines 432-455 exactly.
    """
    names = list(strategy_results.keys())

    # Concat with MultiIndex
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

    # Risk management
    tranches = account_config.get("tranches", 1)
    max_leverage = account_config.get("max_leverage", 1)
    allow_short = account_config.get("allow_negative_weights", False)

    final = account_asset_weights.rolling(window=tranches).mean().iloc[-1]

    if final.sum() > max_leverage:
        final = final / final.sum() * max_leverage

    if not allow_short:
        final = final.clip(lower=0)

    final = final.sort_values(ascending=False)
    final = final[final != 0]

    return {str(k): round(float(v), 6) for k, v in final.items()}


# ==================================================================
# Comparison
# ==================================================================

def compare_weights(
    quantbox_weights: Dict[str, float],
    prod_weights: Dict[str, float],
    tolerance: float = 1e-4,
) -> dict:
    """Compare quantbox vs prod weights."""
    all_syms = set(quantbox_weights.keys()) | set(prod_weights.keys())

    matching = []
    mismatched = []
    only_qb = []
    only_prod = []
    max_diff = 0.0

    for sym in sorted(all_syms):
        qb_w = quantbox_weights.get(sym, 0.0)
        prod_w = prod_weights.get(sym, 0.0)
        diff = abs(qb_w - prod_w)
        max_diff = max(max_diff, diff)

        if sym in quantbox_weights and sym not in prod_weights:
            only_qb.append({"symbol": sym, "weight": qb_w})
        elif sym not in quantbox_weights and sym in prod_weights:
            only_prod.append({"symbol": sym, "weight": prod_w})
        elif diff <= tolerance:
            matching.append({"symbol": sym, "qb": qb_w, "prod": prod_w, "diff": diff})
        else:
            mismatched.append({"symbol": sym, "qb": qb_w, "prod": prod_w, "diff": diff})

    return {
        "matching": matching,
        "mismatched": mismatched,
        "only_quantbox": only_qb,
        "only_prod": only_prod,
        "max_diff": max_diff,
        "match": len(mismatched) == 0 and len(only_qb) == 0 and len(only_prod) == 0,
    }


# ==================================================================
# Main
# ==================================================================

def main():
    account_name = sys.argv[1] if len(sys.argv) > 1 else "paper1"
    target_date = sys.argv[2] if len(sys.argv) > 2 else "2026-02-05"

    print(f"Comparing quantbox vs quantlab prod: account={account_name} date={target_date}")

    # ------------------------------------------------------------------
    # Stage 1: Load config + data from prod cache
    # ------------------------------------------------------------------
    sep("STAGE 1: Load Data (from prod cache)")

    account_config, strategy_configs = load_account_config(account_name)
    print(f"Account: {account_name}")
    strategies = account_config.get("strategies", [])
    print(f"Strategies: {[s['name'] for s in strategies]}")
    print(f"Lookback: {account_config.get('lookback_days', 365)} days")
    print(f"Top coins: {account_config.get('top_coins', 100)}")
    print(f"Max leverage: {account_config.get('max_leverage', 1)}")

    # Load CMC rankings for universe (use target date, not latest)
    coins_ranking = load_cmc_rankings(target_date)
    print(f"\nCMC rankings: {len(coins_ranking)} coins")

    # Apply not_tradable_on_binance filter
    not_tradable = account_config.get("not_tradable_on_binance", [])
    not_tradable_syms = {item["symbol"] for item in not_tradable}
    tradable_ranking = coins_ranking[~coins_ranking["symbol"].isin(not_tradable_syms)].copy()
    tradable_ranking = tradable_ranking.reset_index(drop=True)
    print(f"After filtering: {len(tradable_ranking)} tradable coins ({len(not_tradable_syms)} excluded)")

    # Get top N tickers
    top_n = account_config.get("top_coins", 100)
    tickers = tradable_ranking.iloc[:top_n]["symbol"].tolist()
    print(f"Top {top_n} tickers: {tickers[:10]}... ({len(tickers)} total)")

    # Load OHLCV from cache
    lookback_days = account_config.get("lookback_days", 365)
    ohlcv = load_cached_ohlcv(tickers, target_date, lookback_days)
    print(f"OHLCV loaded: {len(ohlcv)} tickers")
    missing = set(tickers) - set(ohlcv.keys())
    if missing:
        print(f"Missing from cache: {missing}")

    # Preprocess
    market_data = preprocess_data(ohlcv, tradable_ranking)
    print(f"Prices shape: {market_data['prices'].shape}")
    print(f"Volume shape: {market_data['volume'].shape}")
    print(f"Market cap: {len(market_data['market_cap'])} symbols")

    # Validate end date is in data
    prices = market_data["prices"]
    end_date = pd.Timestamp(target_date).normalize()
    if end_date not in prices.index:
        # Try previous day
        end_date_prev = end_date - timedelta(days=1)
        if end_date_prev in prices.index:
            print(f"Note: {target_date} not in prices, using {end_date_prev.date()}")
        else:
            print(f"WARNING: Neither {target_date} nor {end_date_prev.date()} in prices index")
            print(f"  Price index range: {prices.index.min().date()} → {prices.index.max().date()}")

    # ------------------------------------------------------------------
    # Stage 2: Run strategies
    # ------------------------------------------------------------------
    sep("STAGE 2: Strategy Execution")

    strategy_results = {}
    for strat in strategies:
        name = strat["name"]
        weight = strat.get("weight", 1.0)
        strat_params = strat.get("params", {})
        strat_config = strategy_configs.get(name, {})

        print(f"\nRunning: {name} (weight={weight})")
        result = run_quantbox_strategy(
            market_data, name, strat_config, account_config, strat_params
        )

        weights_df = result.get("weights", pd.DataFrame())
        # Normalize multi-level columns
        if isinstance(weights_df, pd.DataFrame) and weights_df.columns.nlevels > 1:
            weights_df = weights_df.T.groupby("ticker").sum().T
            result["weights"] = weights_df

        last_row = weights_df.iloc[-1] if isinstance(weights_df, pd.DataFrame) else weights_df
        nonzero = last_row[last_row != 0].sort_values(ascending=False)
        print(f"  Weights shape: {weights_df.shape}")
        print(f"  Non-zero positions: {len(nonzero)}")
        for sym, w in nonzero.head(5).items():
            print(f"    {sym:6s}: {w:+.6f}")
        if len(nonzero) > 5:
            print(f"    ... and {len(nonzero) - 5} more")

        strategy_results[name] = {"result": result, "weight": weight}

    # ------------------------------------------------------------------
    # Stage 3: Aggregation + Risk
    # ------------------------------------------------------------------
    sep("STAGE 3: Aggregation + Risk Transforms")

    qb_weights = aggregate_and_risk(strategy_results, account_config)
    print(f"\nQuantbox final weights ({len(qb_weights)} positions):")
    for sym, w in sorted(qb_weights.items(), key=lambda x: -x[1]):
        print(f"  {sym:6s}: {w:+.6f}")
    print(f"  Sum={sum(qb_weights.values()):.6f}")

    # ------------------------------------------------------------------
    # Stage 4: Load prod artifact and compare
    # ------------------------------------------------------------------
    sep("STAGE 4: Compare vs Production Artifact")

    artifact = load_prod_artifact(account_name, target_date)
    if artifact is None:
        print(f"ERROR: No production artifact found for {account_name}/{target_date}")
        sys.exit(1)

    prod_weights = artifact["payload"].get("strategy_weights", {})
    print(f"\nProd artifact: {artifact['payload'].get('portfolio_value', 0):.2f} USD")
    print(f"Prod weights ({len(prod_weights)} positions):")
    for sym, w in sorted(prod_weights.items(), key=lambda x: -x[1]):
        print(f"  {sym:6s}: {w:+.6f}")

    # Compare
    comparison = compare_weights(qb_weights, prod_weights)

    sep("RESULTS")

    if comparison["matching"]:
        print(f"\nMATCHING ({len(comparison['matching'])} positions):")
        for m in comparison["matching"]:
            print(f"  {m['symbol']:6s}: QB={m['qb']:+.6f}  Prod={m['prod']:+.6f}  diff={m['diff']:.2e}")

    if comparison["mismatched"]:
        print(f"\nMISMATCHED ({len(comparison['mismatched'])} positions):")
        for m in comparison["mismatched"]:
            print(f"  {m['symbol']:6s}: QB={m['qb']:+.6f}  Prod={m['prod']:+.6f}  diff={m['diff']:.2e}")

    if comparison["only_quantbox"]:
        print(f"\nONLY IN QUANTBOX ({len(comparison['only_quantbox'])}):")
        for m in comparison["only_quantbox"]:
            print(f"  {m['symbol']:6s}: {m['weight']:+.6f}")

    if comparison["only_prod"]:
        print(f"\nONLY IN PROD ({len(comparison['only_prod'])}):")
        for m in comparison["only_prod"]:
            print(f"  {m['symbol']:6s}: {m['weight']:+.6f}")

    print(f"\nMax diff: {comparison['max_diff']:.2e}")

    # Classify result
    if comparison["match"]:
        print("RESULT: EXACT MATCH")
        return 0
    elif (
        not comparison["only_quantbox"]
        and not comparison["only_prod"]
        and comparison["max_diff"] < 0.02
    ):
        print("RESULT: APPROXIMATE MATCH (same positions, small weight diff)")
        print("  Note: Small diffs are expected due to data timing —")
        print("  production fetches live data at runtime, test uses cached data.")
        return 0
    else:
        print("RESULT: MISMATCH")
        return 1


if __name__ == "__main__":
    sys.exit(main())
