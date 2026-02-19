# Plugin Catalog

All built-in plugins registered in `plugins/builtins.py`. Use `uv run quantbox plugins info --name <id>`
for detailed metadata on any plugin.

## Pipelines

| ID | Class | Description |
|---|---|---|
| `fund_selection.simple.v1` | FundSelectionPipeline | Research pipeline: loads data, runs strategy, selects top-N assets |
| `backtest.pipeline.v1` | BacktestPipeline | Historical backtesting with vectorbt or rsims engine |
| `trade.full_pipeline.v1` | TradingPipeline | Full trading: multi-strategy, risk checks, execution, artifacts |
| `trade.allocations_to_orders.v1` | AllocationsToOrdersPipeline | Bridge: converts pre-computed allocations into broker orders |

**When to use which:**
- Research / screening: `fund_selection.simple.v1`
- Historical performance: `backtest.pipeline.v1`
- Paper or live trading: `trade.full_pipeline.v1`
- External allocations: `trade.allocations_to_orders.v1`

## Data Sources

| ID | Class | Description |
|---|---|---|
| `local_file_data` | LocalFileDataPlugin | Loads from local Parquet files (wide format). `params_init: {prices_path: "..."}` |
| `binance.live_data.v1` | BinanceDataPlugin | Binance REST API spot data. No API key needed. |
| `binance.futures_data.v1` | BinanceFuturesDataPlugin | Binance futures OHLCV data |
| `hyperliquid.data.v1` | HyperliquidDataPlugin | Hyperliquid perpetuals data |
| `data.synthetic.v1` | SyntheticDataPlugin | Generates synthetic data via GBM (geometric Brownian motion) |

## Strategies

| ID | Class | Description |
|---|---|---|
| `strategy.crypto_trend.v1` | CryptoTrendStrategy | Donchian breakout with volatility targeting. Multi-asset trend following. |
| `strategy.carver_trend.v1` | CarverTrendStrategy | Robert Carver-style trend following for futures |
| `strategy.momentum_long_short.v1` | MomentumLongShortStrategy | Cross-sectional momentum, long winners / short losers |
| `strategy.cross_asset_momentum.v1` | CrossAssetMomentumStrategy | Time-series momentum across asset classes |
| `strategy.crypto_regime_trend.v1` | CryptoRegimeTrendStrategy | Regime detection (HMM) combined with trend signals |
| `strategy.beglobal.v1` | BeGlobalStrategy | Global multi-asset (equity, bonds, commodities) |
| `strategy.portfolio_optimizer.v1` | PortfolioOptimizerStrategy | Mean-variance optimization (max Sharpe, min variance, risk parity) |
| `strategy.ml_prediction.v1` | MLPredictionStrategy | ML classifier/regressor for return prediction |
| `strategy.weighted_avg.v1` | WeightedAverageAggregator | Aggregator: blends multiple strategy weights by config weight |

**Aggregator note**: When using multiple strategies, include `strategy.weighted_avg.v1` as
the `aggregator` to blend their outputs.

## Brokers

| ID | Class | Mode | Description |
|---|---|---|---|
| `sim.paper.v1` | SimPaperBroker | paper | Simple paper broker for spot trading |
| `sim.futures_paper.v1` | FuturesPaperBroker | paper | Paper broker for futures with leverage |
| `ibkr.paper.stub.v1` | IBKRPaperBrokerStub | paper | IBKR simulator stub |
| `binance.paper.stub.v1` | BinancePaperBrokerStub | paper | Binance simulator stub |
| `ibkr.live.v1` | IBKRBroker | live | IBKR live trading (requires ib_insync) |
| `binance.live.v1` | BinanceBroker | live | Binance live spot (requires python-binance) |
| `binance.futures.v1` | BinanceFuturesBroker | live | Binance live futures |
| `hyperliquid.perps.v1` | HyperliquidBroker | live | Hyperliquid perpetuals |

**Env vars for live brokers:**
- Binance: `API_KEY_BINANCE`, `API_SECRET_BINANCE`
- Hyperliquid: `HYPERLIQUID_WALLET`, `HYPERLIQUID_PRIVATE_KEY`
- IBKR: Requires TWS/Gateway running locally

## Rebalancing

| ID | Class | Description |
|---|---|---|
| `rebalancing.standard.v1` | StandardRebalancer | Spot rebalancing with risk transforms and min trade filter |
| `rebalancing.futures.v1` | FuturesRebalancer | Futures rebalancing with leverage and contract sizing |

## Risk

| ID | Class | Description |
|---|---|---|
| `risk.trading_basic.v1` | TradingRiskManager | Position limits, leverage caps, notional checks |
| `risk.stress_test.v1` | StressTestRiskManager | Stress testing with synthetic scenarios |

**Common risk params:**
```yaml
params:
  max_position_pct: 0.25     # No single position > 25%
  max_leverage: 1.0           # No leverage (spot)
  max_notional: 50000         # Max notional per position
```

## Publishers

| ID | Class | Description |
|---|---|---|
| `telegram.publisher.v1` | TelegramPublisher | Sends run summary to Telegram |

**Telegram setup:**
```yaml
params_init:
  token_env: "TELEGRAM_TOKEN"       # env var name, NOT the token
  chat_id_env: "TELEGRAM_CHAT_ID"   # env var name, NOT the chat ID
```

## Artifact Schemas

JSON schemas in `schemas/` validate pipeline outputs:

| Schema | Validates |
|---|---|
| `prices.schema.json` | Price data (OHLCV) |
| `universe.schema.json` | Universe definition |
| `strategy_weights.schema.json` | Per-strategy weights |
| `aggregated_weights.schema.json` | Blended weights |
| `targets.schema.json` | Target allocations |
| `orders.schema.json` | Generated orders |
| `fills.schema.json` | Execution fills |
| `rebalancing.schema.json` | Rebalancing output |
| `portfolio_daily.schema.json` | Daily portfolio values |
| `allocations.schema.json` | External allocations |
| `scores.schema.json` | Strategy scores |
| `rankings.schema.json` | Asset rankings |
| `trade_history.schema.json` | Historical trades |
| `fx.schema.json` | FX rates |
