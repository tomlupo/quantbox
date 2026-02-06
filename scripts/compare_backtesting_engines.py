"""
Compare quantlab (original) vs quantbox (ported) backtesting engines.

Uses real Binance perpetual futures data from the Robot Wealth validation
dataset.  Runs both the original rsims engine and the ported quantbox
version with identical inputs and checks for exact numerical match.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import BOTH engines
# ---------------------------------------------------------------------------

# 1. Quantlab original (add to sys.path)
QUANTLAB_RSIMS = Path("/home/tom/workspace/prod/quantlab/lab/playground/backtester_validation")
sys.path.insert(0, str(QUANTLAB_RSIMS))
import backtester_rsims_py as quantlab_rsims

# 2. Quantbox ported
from quantbox.plugins.backtesting.rsims_engine import (
    fixed_commission_backtest_with_funding as quantbox_rsims_backtest,
    positions_from_no_trade_buffer as quantbox_ntb,
)

# Also test vectorbt engine import
from quantbox.plugins.backtesting import run_vectorbt, compute_backtest_metrics

# ---------------------------------------------------------------------------
# Load & prepare data
# ---------------------------------------------------------------------------
DATA_CSV = QUANTLAB_RSIMS / "binance_perp_daily.csv"

print("=" * 70)
print("QUANTLAB vs QUANTBOX — BACKTESTING ENGINE COMPARISON")
print("=" * 70)

print(f"\nLoading data from {DATA_CSV}...")
raw = pd.read_csv(DATA_CSV, parse_dates=["date"])
print(f"  Raw rows: {len(raw):,}, tickers: {raw['ticker'].nunique()}")

# --- Prepare data like Robot Wealth blog ---
# Filter to top 30 coins by trailing 30-day volume (excluding stablecoins)
STABLECOINS = {"USDCUSDT", "BUSDUSDT", "USDPUSDT", "TUSDUSDT", "DAIUSDT", "FDUSDUSDT"}
raw = raw[~raw["ticker"].isin(STABLECOINS)].copy()

# Trailing 30-day average dollar volume
raw = raw.sort_values(["ticker", "date"])
raw["vol_30d"] = raw.groupby("ticker")["dollar_volume"].transform(
    lambda s: s.rolling(30, min_periods=1).mean()
)

# Rank by volume within each date, keep top 30
raw["vol_rank"] = raw.groupby("date")["vol_30d"].rank(ascending=False, method="first")
filtered = raw[raw["vol_rank"] <= 30].copy()

# Find first date with 30 tickers
date_counts = filtered.groupby("date")["ticker"].nunique()
start_date = date_counts[date_counts >= 30].index.min()
filtered = filtered[filtered["date"] >= start_date].copy()

# Pivot to wide format
prices = filtered.pivot(index="date", columns="ticker", values="close")
funding = filtered.pivot(index="date", columns="ticker", values="funding_rate").fillna(0)

# Simple equal-weight target
tickers_per_date = prices.notna().sum(axis=1)
weights = prices.notna().astype(float).div(tickers_per_date, axis=0)

# Align all three frames
common_idx = prices.index
common_cols = prices.columns
prices = prices.reindex(index=common_idx, columns=common_cols)
weights = weights.reindex(index=common_idx, columns=common_cols).fillna(0)
funding = funding.reindex(index=common_idx, columns=common_cols).fillna(0)

# Forward-fill prices and drop any leading NaN rows
prices = prices.ffill().dropna(how="all")
common_idx = prices.index
weights = weights.loc[common_idx]
funding = funding.loc[common_idx]

print(f"  Prepared: {prices.shape[0]} dates x {prices.shape[1]} tickers")
print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# ---------------------------------------------------------------------------
# Test 1: positions_from_no_trade_buffer
# ---------------------------------------------------------------------------
print("\n" + "-" * 70)
print("TEST 1: positions_from_no_trade_buffer()")
print("-" * 70)

np.random.seed(42)
test_positions = np.random.randn(5) * 10
test_prices = np.abs(np.random.randn(5)) * 100 + 50
test_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
test_equity = 10000.0
test_buffer = 0.02

result_quantlab = quantlab_rsims.positions_from_no_trade_buffer(
    test_positions.copy(), test_prices, test_weights, test_equity, test_buffer
)
result_quantbox = quantbox_ntb(
    test_positions.copy(), test_prices, test_weights, test_equity, test_buffer
)

diff_ntb = np.max(np.abs(result_quantlab - result_quantbox))
print(f"  Max absolute diff: {diff_ntb:.2e}")
print(f"  Match: {'EXACT' if diff_ntb == 0.0 else 'CLOSE' if diff_ntb < 1e-10 else 'MISMATCH'}")

# ---------------------------------------------------------------------------
# Test 2: rsims engine — Robot Wealth parameters (cost-free)
# ---------------------------------------------------------------------------
PARAM_SETS = [
    {
        "name": "Cost-free, no buffer",
        "trade_buffer": 0.0,
        "initial_cash": 10000,
        "margin": 0.05,
        "commission_pct": 0.0,
        "capitalise_profits": False,
    },
    {
        "name": "With 2% buffer",
        "trade_buffer": 0.02,
        "initial_cash": 10000,
        "margin": 0.05,
        "commission_pct": 0.0,
        "capitalise_profits": False,
    },
    {
        "name": "Reinvesting profits",
        "trade_buffer": 0.0,
        "initial_cash": 10000,
        "margin": 0.05,
        "commission_pct": 0.0,
        "capitalise_profits": True,
    },
    {
        "name": "With commissions",
        "trade_buffer": 0.0,
        "initial_cash": 10000,
        "margin": 0.05,
        "commission_pct": 0.001,
        "capitalise_profits": False,
    },
]

for i, params in enumerate(PARAM_SETS, start=2):
    name = params.pop("name")
    print(f"\n{'=' * 70}")
    print(f"TEST {i}: rsims engine — {name}")
    print("=" * 70)

    # Run quantlab original
    ql_result = quantlab_rsims.fixed_commission_backtest_with_funding(
        prices=prices.copy(),
        target_weights=weights.copy(),
        funding_rates=funding.copy(),
        **params,
    )

    # Run quantbox ported
    qb_result = quantbox_rsims_backtest(
        prices=prices.copy(),
        target_weights=weights.copy(),
        funding_rates=funding.copy(),
        **params,
    )

    # Compare shapes
    print(f"  Shape: quantlab={ql_result.shape}, quantbox={qb_result.shape}")
    assert ql_result.shape == qb_result.shape, "Shape mismatch!"

    # Compare column by column
    numeric_cols = ["Close", "Position", "Value", "Margin", "Funding",
                    "PeriodPnL", "Trades", "TradeValue", "Commission"]
    bool_cols = ["MarginCall", "ReducedTargetPos"]

    print(f"  {'Column':<18} {'Max Abs Diff':>14} {'Max Rel Diff':>14} {'Match':>8}")
    print(f"  {'-'*18} {'-'*14} {'-'*14} {'-'*8}")

    all_match = True
    for col in numeric_cols:
        ql_vals = ql_result[col].values.astype(float)
        qb_vals = qb_result[col].values.astype(float)
        # Treat NaN==NaN as equal (IEEE NaN!=NaN would cause false diffs)
        both_nan = np.isnan(ql_vals) & np.isnan(qb_vals)
        valid = ~(np.isnan(ql_vals) | np.isnan(qb_vals))
        nan_mismatch = np.isnan(ql_vals) != np.isnan(qb_vals)
        if nan_mismatch.any():
            abs_diff = float("inf")
            rel_diff = float("inf")
        elif valid.any():
            abs_diff = np.max(np.abs(ql_vals[valid] - qb_vals[valid]))
            denom = np.maximum(np.abs(ql_vals[valid]), 1e-12)
            rel_diff = np.max(np.abs(ql_vals[valid] - qb_vals[valid]) / denom)
        else:
            abs_diff = 0.0
            rel_diff = 0.0
        match = "EXACT" if abs_diff == 0.0 else "OK" if abs_diff < 1e-8 else "DIFF"
        if match == "DIFF":
            all_match = False
        print(f"  {col:<18} {abs_diff:>14.2e} {rel_diff:>14.2e} {match:>8}")

    for col in bool_cols:
        ql_vals = ql_result[col].values
        qb_vals = qb_result[col].values
        match = "EXACT" if np.all(ql_vals == qb_vals) else "DIFF"
        if match == "DIFF":
            all_match = False
        n_diff = np.sum(ql_vals != qb_vals)
        print(f"  {col:<18} {'':>14} {n_diff:>14d} {match:>8}")

    # Equity comparison
    def calc_equity(df):
        margin_tot = df.groupby(df.index)["Margin"].sum()
        cash = df[df["ticker"] == "Cash"]["Value"]
        return (cash.values + margin_tot.values)

    ql_equity = calc_equity(ql_result)
    qb_equity = calc_equity(qb_result)
    equity_diff = np.max(np.abs(ql_equity - qb_equity))
    final_ql = ql_equity[-1]
    final_qb = qb_equity[-1]

    print(f"\n  Final equity: quantlab=${final_ql:,.2f}, quantbox=${final_qb:,.2f}")
    print(f"  Equity max diff: ${equity_diff:.2e}")
    print(f"  Overall: {'PASS — EXACT MATCH' if all_match else 'DIFFERENCES FOUND'}")

# ---------------------------------------------------------------------------
# Test 6: VectorBT engine smoke test with same data (subset)
# ---------------------------------------------------------------------------
print(f"\n{'=' * 70}")
print("TEST 6: VectorBT engine — weekly rebalancing on subset")
print("=" * 70)

# Use a smaller subset — pick tickers with full price history for this window
vbt_subset = prices.iloc[:200].dropna(axis=1)  # only tickers with no NaN
vbt_prices = vbt_subset.iloc[:, :10]
n_tickers = vbt_prices.shape[1]
vbt_weights = pd.DataFrame(1.0 / n_tickers, index=vbt_prices.index, columns=vbt_prices.columns)

pf = run_vectorbt(vbt_prices, vbt_weights, rebalancing_freq="1W", fees=0.001)
metrics = compute_backtest_metrics(pf)

print(f"  Final value: ${pf.final_value():.2f}")
print(f"  Total return: {metrics['total_return']:.4f}")
print(f"  Sharpe: {metrics['sharpe']:.4f}")
print(f"  Max drawdown: {metrics['max_drawdown']:.4f}")
print(f"  VBT engine: OK")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)
print("  positions_from_no_trade_buffer: TESTED")
print("  rsims engine (4 param sets):    TESTED")
print("  vectorbt engine:                TESTED")
print("  metrics module:                 TESTED")
print("=" * 70)
