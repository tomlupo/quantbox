# Data Sources

Data plugins load market data for pipelines and strategies. All data is returned in
**wide format**: DataFrames with a DatetimeIndex and one column per symbol.

## When to Use

- You need to provide price/volume/market cap data to a pipeline
- You want to add a new exchange or data provider
- You need to understand the data format contract

## Available Data Plugins

<!-- BEGIN AUTO-GENERATED -->
| ID | Description | Tags |
|---|---|---|
| `binance.futures_data.v1` | USDM futures data from Binance (OHLCV + funding rates, no API key). | binance, crypto, futures, live |
| `binance.live_data.v1` | Live market data from Binance public API (no API key needed). | binance, crypto, live |
| `data.synthetic.v1` | Synthetic market data generator using stochastic models (GBM, jump diffusion, mean reversion). Useful for strategy research, stress-testing, and CI pipelines. | synthetic, simulation, research |
| `hyperliquid.data.v1` | Perpetual-futures data from Hyperliquid REST API (no API key). | hyperliquid, crypto, futures, live |
| `local_file_data` | Load market data from local Parquet/CSV files via DuckDB | - |
<!-- END AUTO-GENERATED -->

## Data Format Contract

`load_market_data()` returns `dict[str, pd.DataFrame]`:

| Key | Required | Format | Description |
|---|---|---|---|
| `"prices"` | Yes | DatetimeIndex x symbols | Close prices |
| `"volume"` | No | DatetimeIndex x symbols | Trading volume |
| `"market_cap"` | No | DatetimeIndex x symbols | Market capitalization |
| `"funding_rates"` | No | DatetimeIndex x symbols | Funding rates (perps) |

All DataFrames are **wide format**: rows = dates, columns = symbol names.

## Config Examples

### Local Parquet files
```yaml
plugins:
  data:
    name: "local_file_data"
    params_init:
      prices_path: "./data/curated/prices.parquet"
      # Optional:
      # volume_path: "./data/curated/volume.parquet"
      # market_cap_path: "./data/curated/market_cap.parquet"
```

### Binance spot (no API key)
```yaml
plugins:
  data:
    name: "binance.live_data.v1"
    params_init:
      quote_currency: "USDT"
```

### Synthetic data (for testing)
```yaml
plugins:
  data:
    name: "data.synthetic.v1"
    params_init:
      n_symbols: 20
      n_days: 500
```

## Next Steps

- **Create a custom data plugin**: Load [api.md](api.md)
- **Debug data issues**: Load [gotchas.md](gotchas.md)
