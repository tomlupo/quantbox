# Data Sources

Data plugins load market data for pipelines and strategies. All data is returned in
**wide format**: DataFrames with a DatetimeIndex and one column per symbol.

## When to Use

- You need to provide price/volume/market cap data to a pipeline
- You want to add a new exchange or data provider
- You need to understand the data format contract

## Available Data Plugins

| ID | Source | API key needed | Description |
|---|---|---|---|
| `local_file_data` | Local files | No | Loads from Parquet files on disk |
| `binance.live_data.v1` | Binance REST | No | Spot OHLCV + market cap for top coins |
| `binance.futures_data.v1` | Binance REST | No | Futures OHLCV + funding rates |
| `hyperliquid.data.v1` | Hyperliquid | No | Perpetuals OHLCV data |
| `data.synthetic.v1` | Generated | No | Synthetic data via GBM for testing |

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
