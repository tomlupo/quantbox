# Backtesting

QuantBox includes two backtesting engines accessible through the `backtest.pipeline.v1` pipeline plugin. Both use the same strategy configs as live trading — swap the pipeline name to go from backtest to production.

## Quick start

```bash
quantbox run -c configs/run_backtest_crypto_trend.yaml
```

## Engines

### vectorbt (spot / equity)

Numba-accelerated portfolio simulation. Best for spot strategies without leverage.

```yaml
plugins:
  pipeline:
    name: "backtest.pipeline.v1"
    params:
      engine: vectorbt
      fees: 0.001               # 10 bps per trade
      rebalancing_freq: 1       # every N days (or "1W", "1M")
      # threshold: 0.05         # uncomment for rebalancing-bands mode
      trading_days: 365
```

**Rebalancing modes:**
- `rebalancing_freq: N` — periodic rebalancing every N days
- `threshold: 0.05` — rebalance when any weight drifts more than 5% from target

### rsims (futures)

Daily step simulator with funding rates, margin, leverage, and no-trade buffer. Best for futures strategies.

```yaml
plugins:
  pipeline:
    name: "backtest.pipeline.v1"
    params:
      engine: rsims
      fees: 0.001
      rebalancing_freq: 1
      trading_days: 365

      risk:
        tranches: 1
        max_leverage: 2
        allow_short: true
```

**Additional rsims features:**
- Funding rate simulation (long/short asymmetry)
- Margin and leverage tracking
- No-trade buffer to reduce turnover

## Configuration

### Full example

```yaml
run:
  mode: backtest
  asof: "2026-02-06"
  pipeline: "backtest.pipeline.v1"

artifacts:
  root: "./artifacts"

plugins:
  pipeline:
    name: "backtest.pipeline.v1"
    params:
      engine: vectorbt
      fees: 0.001
      rebalancing_freq: 1
      trading_days: 365

      universe:
        top_n: 100

      prices:
        lookback_days: 365

  strategies:
    - name: "strategy.crypto_trend.v1"
      weight: 0.6
      params:
        lookback_days: 365

    - name: "strategy.carver_trend.v1"
      weight: 0.4
      params:
        lookback_days: 365
        target_vol: 0.50

  aggregator:
    name: "strategy.weighted_avg.v1"
    params: {}

  data:
    name: "binance.live_data.v1"
    params_init:
      quote_asset: USDT
```

### Parameters reference

| Parameter | Default | Description |
|---|---|---|
| `engine` | `vectorbt` | `"vectorbt"` or `"rsims"` |
| `fees` | `0.001` | Trading fee per side (0.001 = 10 bps) |
| `rebalancing_freq` | `1` | Rebalance every N days, or `"1W"`, `"1M"` |
| `threshold` | (none) | Drift threshold for rebalancing-bands mode |
| `trading_days` | `365` | Days per year for annualization |
| `universe.top_n` | — | Universe size (top N by volume/mcap) |
| `prices.lookback_days` | — | Price history window |
| `risk.max_leverage` | `1` | Max leverage (rsims only) |
| `risk.allow_short` | `false` | Allow negative weights (rsims only) |
| `risk.tranches` | `1` | Number of rebalancing tranches (rsims only) |

## Outputs

Artifacts are written to `artifacts/<run_id>/`:

| Artifact | Description |
|---|---|
| `strategy_weights` | Per-strategy weight time series |
| `aggregated_weights` | Final blended weights after aggregation |
| `weights_history` | Full weight history across all rebalancing points |
| `portfolio_daily` | Daily portfolio value series |
| `returns` | Daily return series |
| `metrics` | Summary statistics (Sharpe, drawdown, etc.) |

## Metrics

The `metrics` artifact includes:

- **Sharpe ratio** — annualized risk-adjusted return
- **Max drawdown** — peak-to-trough decline
- **CAGR** — compound annual growth rate
- **Volatility** — annualized return standard deviation
- **Calmar ratio** — CAGR / max drawdown
- **Rolling Sharpe** — time-varying Sharpe windows

## Research to production

The same strategy params work in both backtesting and live trading. To go live:

1. **Backtest:**
   ```yaml
   pipeline:
     name: "backtest.pipeline.v1"
   ```

2. **Live (swap pipeline, add broker + risk + rebalancer):**
   ```yaml
   pipeline:
     name: "trade.full_pipeline.v1"
   broker:
     name: "hyperliquid.perps.v1"
   rebalancing:
     name: "rebalancing.futures.v1"
   risk:
     - name: "risk.trading_basic.v1"
   ```

Strategy and data sections stay the same.
