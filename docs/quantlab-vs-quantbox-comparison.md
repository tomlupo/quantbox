# Quantlab vs Quantbox: Comprehensive Trading Pipeline Comparison

## Overview

| Aspect | Quantlab (prod) | Quantbox (quantbox-core) |
|--------|-----------------|--------------------------|
| **Location** | `/home/tom/workspace/prod/quantlab/` | `/home/tom/workspace/projects/quantbox/packages/quantbox-core/` |
| **Architecture** | Monolithic: modules import each other directly | Plugin-based: Protocol contracts + registry + dataclass plugins |
| **Python version** | >=3.12 | >=3.10 |
| **Package manager** | uv | uv |
| **Total declared deps** | ~50 packages (pyproject.toml) | 5 core deps + optional extras |
| **Config format** | YAML (accounts/ + strategies/) | YAML (single run config + manifest profiles) |
| **Entry point** | `run_trading_bot_{account}.py` -> `engine.run_process()` | `runner.py:run_from_config()` |

---

## Step-by-Step Pipeline Comparison

### Step 0: Entry Point & Config Loading

| Aspect | Quantlab | Quantbox |
|--------|----------|----------|
| **Entry script** | `run_trading_bot_paper1.py` calls `engine.run_process(process_name=...)` | `runner.py:run_from_config(config_path)` |
| **Config loading** | `load_yaml()` on `config/accounts/{name}.yaml` + per-strategy `config/strategies/{name}.yaml` | Single YAML config; profiles resolved via `plugins/manifest.yaml` |
| **Env vars** | `python-dotenv` loads `.env` (API keys, tokens) | Same, via `.env` or env vars |
| **Dynamic dispatch** | `importlib.import_module(f'workflow.{task_module}')` + `getattr(module, task_function)` | `PluginRegistry.discover()` merges builtins + entry points; plugins instantiated via `params_init` |
| **Orchestrator** | `engine/core.py` with `@log_execution()` decorator (timing, error handling, artifact saving) | `runner.py` with event log (JSONL), schema validation, run manifest |
| **Packages** | `yaml`, `dotenv`, `importlib`, `luigi` (optional), `utils` (custom) | `yaml`, `hashlib`, `json`, `pathlib`, `importlib` |

---

### Step 1: Market Data Fetching - Universe / Rankings

| Aspect | Quantlab | Quantbox |
|--------|----------|----------|
| **Module** | `workflow/data_fetcher.py:CryptoDataFetcher` | `plugins/datasources/binance_data.py:BinanceDataFetcher` |
| **Market cap source** | **CoinMarketCap API** (`dq.coinmarketcap.fetch_cmc_rankings(limit=100)`) - live API call with API key | **Hardcoded circulating supply estimates** (`_estimate_market_cap()`) - price * fixed supply values for ~20 major coins, default 1B for others |
| **CMC caching** | `FastParquetCache` (DuckDB-backed), 4h fresh / 28h fallback TTL | N/A (no CMC integration) |
| **Universe candidates** | CMC top 100 by market cap | Either explicit `symbols` list or top N by 24h Binance volume |
| **Stablecoin list** | `['USDT','USDC','BUSD','TUSD','DAI','MIM','USTC','FDUSD','USD1']` (9) | `DEFAULT_STABLECOINS` list (16 coins including wrapped/synthetic) |
| **Validation** | `basic_validate_ohlcv()` - checks columns, negative prices, high<low | Minimal validation |
| **Packages** | `requests`, `duckdb`, `pyarrow`, `pandas`, `tqdm` | `requests`, `pandas`, `time` |

**Key Difference**: Quantlab uses **live CoinMarketCap API** for accurate, real-time market cap rankings. Quantbox uses **hardcoded circulating supply approximations**, making market cap rankings potentially inaccurate for smaller/newer coins.

---

### Step 2: Market Data Fetching - OHLCV Prices

| Aspect | Quantlab | Quantbox |
|--------|----------|----------|
| **Module** | `data_fetcher.py:fetch_ohlcv()` | `binance_data.py:get_ohlcv()` + `get_market_data()` |
| **OHLCV source** | **ccxt** (`binance` exchange) - daily candles, 1000/batch | **ccxt** (optional) or **requests** to Binance REST API |
| **Default lookback** | 730 days (2 years) | Configurable via params (typically 365 days) |
| **Pair resolution** | `get_valid_binance_pairs()` - tries USDT, BUSD, BTC quotes via `requests` to `/api/v3/exchangeInfo` | Similar: tries primary quote, then USDC, BUSD, BTC fallbacks |
| **Caching** | `FastParquetCache` - DuckDB-backed Parquet files in `data/cache/` | In-memory TTL cache (60s default) |
| **End date** | `pd.Timestamp.now().normalize() - timedelta(days=1)` (yesterday) | Configurable `asof` date from config |
| **Output format** | Dict[ticker -> DataFrame(date, OHLCV)] -> preprocessed to wide DataFrames | Dict with `prices`, `volume`, `market_cap` as wide DataFrames |
| **Rate limiting** | Via ccxt `enableRateLimit: True` | 100ms between requests, 3 retries |
| **Packages** | `ccxt`, `requests`, `duckdb`, `pyarrow`, `pandas`, `tqdm` | `ccxt` (optional), `requests`, `pandas`, `time` |

**Key Difference**: Both use ccxt for OHLCV. Quantlab has a robust DuckDB-backed Parquet cache; Quantbox uses simple in-memory caching. Quantlab always fetches 2 years; Quantbox is configurable.

---

### Step 3: Data Preprocessing

| Aspect | Quantlab | Quantbox |
|--------|----------|----------|
| **Module** | `trading.py:preprocess_data()` | Done inside `DataPlugin.load_prices()` and `TradingPipeline._build_market_data()` |
| **Price extraction** | Manual: `pd.DataFrame([df.set_index('date')['close'].rename(ticker) for ticker, df in ohlcv.items()]).T` | Internal to data plugin: `_wide_to_long()` conversion |
| **Volume extraction** | Same pattern as prices | Same - extracted from OHLCV data |
| **Market cap** | From CMC rankings DataFrame `coins_ranking.set_index('symbol')['market_cap']` | Estimated from `price * circulating_supply` |
| **Validation** | Shape checks, prices for end_date, completeness | Minimal |
| **Output** | `processed_data` dict: `{ohlcv, tickers, coins_ranking, prices, volume, market_cap}` | Dict: `{prices, volume, market_cap}` as DataFrames |
| **Packages** | `pandas` | `pandas` |

---

### Step 4: Universe Selection (within Strategy)

| Aspect | Quantlab | Quantbox |
|--------|----------|----------|
| **Module** | `strategies/crypto_trend_catcher.py:select_universe()` | `strategies/crypto_trend.py:select_universe_vectorized()` or `select_universe_duckdb()` |
| **Step 1** | Filter stablecoins (9 coins) | Filter stablecoins (16 coins) + custom exclude list |
| **Step 2** | Rank by market cap -> top `filtered_coins_market_cap` (default 30) | `market_cap.rank(axis=1, ascending=False)` -> top `top_by_mcap` (default 30) |
| **Step 3** | Within MC-filtered, rank by volume -> top `portfolio_coins_max` (default 10) | `dollar_vol = prices * volume`, rank -> top `top_by_volume` (default 10) |
| **Dollar volume calc** | Implicit in volume ranking | Explicit: `dollar_vol = prices * volume` (more correct for cross-asset comparison) |
| **DuckDB acceleration** | No | Optional via `select_universe_duckdb()` with SQL windowed RANK() |
| **Output** | Binary mask DataFrame (0/1) | Binary mask DataFrame (0/1) |
| **Packages** | `pandas`, `numpy` | `pandas`, `numpy`, optionally `duckdb` |

**Key Difference**: Quantbox explicitly uses dollar volume (price * volume) for volume ranking, which is more correct. Quantlab uses raw volume. Quantbox also offers DuckDB-accelerated universe selection for large datasets.

---

### Step 5: Signal Generation (Donchian Breakout)

| Aspect | Quantlab | Quantbox |
|--------|----------|----------|
| **Module** | `strategies/crypto_trend_catcher.py:generate_signals()` | `strategies/crypto_trend.py:generate_ensemble_signals()` |
| **Algorithm** | Donchian channel breakout with trailing stop | Same: Donchian channel breakout with trailing stop |
| **Lookback windows** | `[5, 10, 20, 30, 60, 90, 150, 250, 360]` (9 windows) | Same: `[5, 10, 20, 30, 60, 90, 150, 250, 360]` (9 windows) |
| **Breakout condition** | `price >= rolling_high` | `price >= rolling_high` |
| **Trailing stop** | `price < midpoint` (mid = (high+low)/2) | Same: `price < midpoint` |
| **Ensemble** | Average across windows | Same: `np.mean([signal_w1, ..., signal_w9])` |
| **Implementation** | Loop-based with vectorized rolling | `compute_donchian_breakout_vectorized()` - fully vectorized + stateful trailing stop |
| **Alternative** | N/A | `compute_donchian_simple_vectorized()` - no trailing stop (faster) |
| **Packages** | `pandas`, `numpy` | `pandas`, `numpy` |

**Key Difference**: Identical algorithm. Quantbox has two variants (with/without trailing stop) and a more explicitly vectorized implementation.

---

### Step 6: Volatility Targeting

| Aspect | Quantlab | Quantbox |
|--------|----------|----------|
| **Module** | `crypto_trend_catcher.py:volatility_scaler()` | `crypto_trend.py:compute_volatility_scalers()` |
| **Volatility calc** | `pct_change().rolling(vol_lookback).std() * sqrt(365)` | Same: `returns.rolling(vol_lookback).std() * sqrt(365)` |
| **Default vol lookback** | 60 days | 60 days |
| **Default vol targets** | `[0.5]` (50% annualized) | `[0.25, 0.50]` (25% and 50%) |
| **Scaler formula** | `target_vol / realized_vol` | Same: `target_vol / realized_vol` |
| **Clipping** | Not documented | Clipped to [0.1, 10.0] |
| **Packages** | `pandas`, `numpy` | `pandas`, `numpy` |

**Key Difference**: Quantbox tests two vol targets by default (0.25 and 0.50) vs quantlab's single 0.50. Quantbox also clips extreme scalers.

---

### Step 7: Portfolio Construction (Weight Computation)

| Aspect | Quantlab | Quantbox |
|--------|----------|----------|
| **Module** | `crypto_trend_catcher.py:construct_weights()` | `crypto_trend.py:construct_weights()` |
| **For each (vol_target, tranche)** | | |
| **Signal scaling** | `sig_scaled = signals * vol_scaler` | Same |
| **Tranching** | `rolling(window=tranche).mean()` (default tranches: `[5]`) | Same: `rolling(t, min_periods=1).mean()` (default: `[1, 5, 21]`) |
| **Universe masking** | `w = sig_tranched * universe` | Same |
| **Normalization** | `w / universe.sum(axis=1)` | Same: `w.div(universe.sum(axis=1), axis=0)` |
| **Output** | MultiIndex DataFrame (vol_target x tranche x ticker) | Same: MultiIndex DataFrame |
| **Default tranches** | `[5]` (single) | `[1, 5, 21]` (three - more combos) |
| **Output period** | Last 30 days (`last_x_days`) | Configurable via `output_periods` |
| **Packages** | `pandas` | `pandas` |

**Key Difference**: Quantbox explores more tranche combinations (1, 5, 21 days) vs quantlab's single 5-day tranche. This means quantbox generates more weight variants for analysis.

---

### Step 8: Strategy Aggregation & Risk Transforms

| Aspect | Quantlab | Quantbox |
|--------|----------|----------|
| **Module** | `trading.py` (inline, lines 351-375) | `trading_pipeline.py:_aggregate_strategies()` or aggregator plugin |
| **Multi-strategy stacking** | `pd.concat([weights], keys=strategy_names)` | Same: `pd.concat([weights], keys=names)` |
| **Strategy weighting** | `.mul(account_strategies_weights, level='strategy')` | Same: `.mul(acct_w, level="strategy")` |
| **Aggregation** | `.droplevel(0, axis=1)` (drop strategy level) | `.groupby(level=-1, axis=1).sum()` per ticker |
| **Tranching (risk)** | `rolling(window=tranches).mean().iloc[-1]` (account-level) | `_apply_risk_transforms()`: rolling mean over N days |
| **Max leverage** | `if sum > max_lev: weights = weights / sum * max_lev` | Same: scale down if `sum(abs(weights)) > max_leverage` |
| **Short clamping** | `.clip(lower=0)` if not `allow_negative_weights` | Same: clamp negatives to 0 if `allow_short=False` |
| **Zero removal** | `final.loc[final != 0]` | Filter `> 0.001` (small threshold) |
| **Pluggable aggregator** | No | Yes: `WeightedAvgAggregator` plugin or custom |
| **Packages** | `pandas` | `pandas`, `numpy` |

**Key Difference**: Quantbox has a pluggable aggregator (can swap in custom logic) and uses a small threshold (0.001) instead of exact zero for filtering. The core aggregation logic is equivalent.

---

### Step 9: Order Generation

| Aspect | Quantlab | Quantbox |
|--------|----------|----------|
| **Module** | `trading_bot/orders.py:generate_portfolio_orders()` | `trading_pipeline.py:_generate_orders()` or rebalancer plugin |
| **Current holdings** | `get_current_holdings(client, account_config)` - live Binance API or paper trading state | `broker.get_positions()` via BrokerPlugin protocol |
| **Cash query** | `get_cash_available(current_holdings, stable_coin_symbol)` | `broker.get_cash()` |
| **Portfolio value** | `get_portfolio_value(holdings, price_fn, stable_coin, exclusions)` | `cash + sum(qty * price)` from broker |
| **Capital at risk** | `weight * capital_at_risk` scaling | Same scaling |
| **Target qty** | `(total_value * weight) / price` | Same |
| **Rebalancing DF** | `generate_rebalancing_dataframe()` - columns: Asset, Current/Target Qty/Value/Weight, Delta, Action | `_build_rebalancing()` - same columns |
| **Lot/step size** | `get_symbol_info(client, symbol)` -> `get_lot_size_and_min_notional()` | `broker.get_market_snapshot(symbols)` -> min_qty, step_size, min_notional |
| **Qty adjustment** | `adjust_quantity(qty, step_size)` via `Decimal` | `_adjust_quantity()` via `Decimal(ROUND_DOWN)` |
| **Min notional check** | `notional < min_notional` -> DROPPED | Same |
| **Buy scaling** | Scale down if cash insufficient (per-order) | Batch scaling: `scaling_factor = min(1.0, cash / total_buy)`, apply if >= `scaling_factor_min` (0.9) |
| **Price source** | `cached_get_price()` - TTL 30s, tries direct pair then BTC intermediate | Via `broker.get_market_snapshot()` |
| **Pluggable** | No - hardcoded | Yes: `StandardRebalancer` plugin or custom |
| **Packages** | `pandas`, `decimal`, `python-binance` | `pandas`, `numpy`, `decimal` |

**Key Difference**: Quantlab scales buy orders individually as cash runs out. Quantbox does batch scaling - calculates total buy value, computes a single scaling factor, and applies uniformly. Quantbox's approach is more predictable but less optimal for partial fills. Quantbox also abstracts broker interaction behind a Protocol.

---

### Step 10: Risk Checks

| Aspect | Quantlab | Quantbox |
|--------|----------|----------|
| **Module** | Inline in `trading.py` (max leverage, clip negatives) | `plugins/risk/trading_risk.py:TradingRiskManager` |
| **Pre-execution checks** | Max leverage, clip negatives (applied as transforms, not checks) | `check_targets()`: max_leverage, max_concentration, negative_weights |
| **Order-level checks** | Min notional (in order generation) | `check_orders()`: min_notional, max_order_notional |
| **Output format** | Transforms applied directly (mutates weights) | Returns `List[Dict]` with `{level, rule, detail}` findings |
| **Pluggable** | No | Yes: list of RiskPlugin instances, extensible |
| **Concentration limit** | Not checked | `abs(weight) > max_concentration` per asset |
| **Packages** | `pandas` | `pandas`, `dataclasses` |

**Key Difference**: Quantlab applies risk as inline transforms (mutate then proceed). Quantbox separates risk into a dedicated plugin that returns findings (advisory) without blocking execution. Quantbox also checks per-asset concentration limits.

---

### Step 11: Order Execution

| Aspect | Quantlab | Quantbox |
|--------|----------|----------|
| **Module** | `trading_bot/orders.py:execute_orders()` + `trading_bot/binance.py` | `trading_pipeline.py:_execute_orders()` -> `broker.place_orders()` |
| **Sort order** | Sells before buys (descending Action) | Same: sells before buys |
| **Live execution** | `place_market_order(client, symbol, side, qty)` via `python-binance` Client | `broker.place_orders(orders_df)` via BrokerPlugin |
| **Paper execution** | `paper_order_executor()` -> `simulate_market_execution()` with spread/slippage/impact | `FuturesPaperBroker.place_orders()` with spread/slippage/fees |
| **Paper slippage** | 0.05% (PAPER_TRADING_SLIPPAGE) | Configurable `slippage_bps` |
| **Paper spread** | 0.1% (PAPER_TRADING_SPREAD) | Configurable `spread_bps` |
| **Paper commission** | 0.1% (PAPER_TRADING_COMMISSION) | Configurable `maker_fee_bps` / `taker_fee_bps` |
| **Paper price impact** | `min(qty * 0.0001, 0.002)` proportional to size | Not modeled separately |
| **Paper delay** | Random 0.1-2.0s `time.sleep()` | No simulated delay |
| **Fill tracking** | Parse Binance `fills` array (price, qty, commission, commissionAsset) | Fills DataFrame with symbol, side, qty, executed_price, fee |
| **Spread analytics** | `spread = abs(fill_price - reference_price)` in bps | Same: spread vs reference price |
| **Retry logic** | Exponential backoff (1s, 2s, 4s) on transient errors (503, rate limit -1003) | No retry in broker plugin (relies on upstream) |
| **BTC intermediate** | Falls back to `{asset}BTC * BTCUSDT` if direct pair unavailable | Similar fallback logic |
| **State persistence** | Paper: JSON file `data/paper_trading/paper_trading_{account}.json` | Paper: in-memory state (lost on restart) |
| **Funding rates** | Not modeled | `broker.apply_funding()` for futures positions |
| **Position limits** | Not modeled in spot | `position_limits` dict in FuturesPaperBroker |
| **Packages** | `python-binance`, `requests`, `time`, `random` | `pandas`, `dataclasses` |

**Key Differences**:
- Quantlab has more realistic paper trading (price impact, execution delay, persistent state).
- Quantbox supports **futures** (funding rates, position limits) which quantlab doesn't.
- Quantbox paper state is in-memory only; quantlab persists to JSON.
- Quantlab has retry logic with backoff; quantbox doesn't retry failed orders.

---

### Step 12: Notifications / Publishing

| Aspect | Quantlab | Quantbox |
|--------|----------|----------|
| **Module** | `trading_bot/telegram.py` + `utils/messaging.py` + `engine/notifications.py` | `plugins/publisher/telegram.py:TelegramPublisher` |
| **Telegram** | `requests.post()` to Telegram Bot API | `urllib.request` to Telegram Bot API (no `requests` dep) |
| **Slack** | `slack_sdk.WebClient` with full Slack API | Not implemented |
| **Email** | `smtplib` with MIMEMultipart, HTML tables | Not implemented |
| **Message format** | HTML tables, emoji numbers, code blocks | HTML formatted summary |
| **Content** | Execution summary, rebalancing table, executed orders, failed orders | Same: execution summary, portfolio rebalancing, strategy weights |
| **Table formatting** | `tabulate` library, custom `df_to_html_email()`, `df_to_slack_table()` | Custom HTML formatting functions |
| **Pluggable** | No - hardcoded channels | Yes: PublisherPlugin protocol, list of publishers |
| **Packages** | `requests`, `slack_sdk`, `smtplib`, `tabulate`, `json` | `urllib.request`, `json`, `os` |

**Key Difference**: Quantlab has three notification channels (Telegram, Slack, Email). Quantbox has only Telegram but uses stdlib `urllib.request` to avoid the `requests` dependency. Quantbox publishers are pluggable.

---

### Step 13: Artifact Storage

| Aspect | Quantlab | Quantbox |
|--------|----------|----------|
| **Module** | `engine/core.py` `@log_execution()` decorator + inline in `trading.py` | `store.py:FileArtifactStore` |
| **Path pattern** | `output/{process}/artifacts/{date}/{time}/artifact.json` | `{root}/{run_id}/{name}.parquet` or `.json` |
| **JSON artifact** | Single `artifact.json` with full payload | `run_meta.json`, `trade_history.json`, `run_manifest.json` |
| **Parquet files** | Not used for artifacts | `universe.parquet`, `prices.parquet`, `targets.parquet`, `rebalancing.parquet`, `orders.parquet`, `fills.parquet`, `portfolio_daily.parquet` |
| **Event log** | Python logging to file | JSONL event log (`events.jsonl`) with RUN_START/RUN_END |
| **Schema validation** | Not present | JSON schema validation (best-effort) via `jsonschema` |
| **Run ID** | `{date}/{timestamp}` | Hash of `asof + pipeline + config` |
| **Packages** | `json`, `logging` | `json`, `pathlib`, `pandas`, `jsonschema` |

**Key Difference**: Quantbox stores structured Parquet files for every pipeline stage (enabling downstream analysis). Quantlab stores a single JSON artifact. Quantbox also does schema validation.

---

## Package Dependency Comparison

| Package | Quantlab | Quantbox Core | Notes |
|---------|----------|---------------|-------|
| **pandas** | >=2.0.0 | >=2.0 | Both core |
| **numpy** | >=1.24.0 | (transitive via pandas) | Quantlab explicit dep |
| **pyarrow** | >=12.0.0 | >=14.0 | Both for Parquet |
| **duckdb** | >=0.8.0 | >=0.10.0 | Both for fast queries |
| **pyyaml** | >=6.0 | >=6.0 | Both for config |
| **jsonschema** | - | >=4.19 | Quantbox only (artifact validation) |
| **ccxt** | >=4.0.0 | *(not in core deps)* | Quantlab explicit; quantbox uses but not declared |
| **python-binance** | >=1.0.19 | optional `[binance]` | Quantlab explicit; quantbox optional |
| **requests** | >=2.31.0 | *(not in core deps)* | Quantlab explicit; quantbox uses `urllib.request` in core |
| **scipy** | >=1.10.0 | - | Quantlab only |
| **scikit-learn** | >=1.3.0 | - | Quantlab only |
| **plotly** | >=5.15.0 | - | Quantlab only (visualization) |
| **matplotlib** | >=3.7.0 | - | Quantlab only |
| **seaborn** | >=0.12.0 | - | Quantlab only |
| **luigi** | >=3.3.0 | - | Quantlab only (workflow orchestration) |
| **slack_sdk** | >=3.21.0 | - | Quantlab only (Slack notifications) |
| **python-dotenv** | >=1.0.0 | - | Quantlab only |
| **tqdm** | >=4.65.0 | - | Quantlab only (progress bars) |
| **quantstats** | >=0.0.62 | - | Quantlab only (perf metrics) |
| **vectorbt** | >=0.25.0 | - | Quantlab only (backtesting) |
| **tabulate** | >=0.9.0 | - | Quantlab only (ASCII tables) |
| **numba** | >=0.58.0 | - | Quantlab only (JIT) |
| **statsmodels** | >=0.14.0 | - | Quantlab only |
| **PyPortfolioOpt** | >=1.5.0 | - | Quantlab only |
| **yfinance** | ==0.2.28 | - | Quantlab only |
| **dill** | >=0.3.6 | - | Quantlab only |
| **arch** | >=6.0.0 | - | Quantlab only (GARCH) |
| **psycopg2** | >=2.9.0 | - | Quantlab only (Postgres) |
| **sqlalchemy** | >=2.0.0 | - | Quantlab only |
| **diskcache** | >=5.6.0 | - | Quantlab only |

---

## Architectural Differences Summary

| Dimension | Quantlab | Quantbox |
|-----------|----------|----------|
| **Extensibility** | Hard to extend - modify source directly | Plugin protocols - swap any component |
| **Testing** | Tightly coupled - hard to unit test | Protocol-based - easy to mock any plugin |
| **Market cap accuracy** | Accurate (live CMC API) | Approximate (hardcoded supply) |
| **Data caching** | Production-grade DuckDB Parquet cache | In-memory TTL only |
| **Paper trading realism** | Price impact, execution delay, persistent state | Configurable costs, no price impact, no state persistence |
| **Futures support** | No (spot only) | Yes (FuturesPaperBroker with funding, position limits) |
| **Notification channels** | Telegram + Slack + Email | Telegram only (pluggable) |
| **Artifact richness** | Single JSON | Parquet per stage + JSON + schema validation |
| **Strategy variants** | 1 default combo per strategy | Multiple vol_target x tranche combos explored |
| **Additional strategies** | cross_asset_momentum | carver_trend, momentum_long_short, weighted_avg_aggregator |
| **Retry/resilience** | Exponential backoff on exchange errors | No retry in broker layer |
| **Dependencies** | ~50 packages (heavy) | 5 core packages (minimal) |

---

## Algorithm Match Status (Crypto Trend)

| Algorithm Component | Match? | Notes |
|--------------------|--------|-------|
| Donchian breakout logic | Exact match | Same rolling high/low/mid, same trailing stop |
| Lookback windows | Exact match | [5,10,20,30,60,90,150,250,360] |
| Ensemble method | Exact match | Average across windows |
| Universe selection flow | Equivalent | MC filter -> vol filter, same thresholds |
| Vol targeting formula | Exact match | target_vol / realized_vol |
| Annualization factor | Exact match | sqrt(365) |
| Weight normalization | Exact match | Divide by universe.sum(axis=1) |
| Default parameters | Mostly match | Vol targets differ (0.5 vs 0.25+0.50), tranches differ (5 vs 1+5+21) |
| Validated output | Exact match | Confirmed via `scripts/test_pipeline_vs_prod.py` on paper1 Feb 3-5 2026 |
