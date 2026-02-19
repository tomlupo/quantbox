# Data Gotchas

## DataLoadError: API rate limit

**Cause:** Binance/Hyperliquid API rate limits exceeded when fetching large universes.

**Solution:** Reduce universe size or add delays between requests:
```yaml
plugins:
  pipeline:
    params:
      universe:
        top_n: 50    # reduce from 100+
```

## Empty DataFrame from load_market_data

**Cause:** No data available for the requested symbols or date range.

**Solution:** Check:
1. Symbols exist on the exchange (e.g., "BTC" not "BTCUSDT" for Binance plugin)
2. Date range is valid (not in the future, not before listing date)
3. Network connectivity is working

Test with a known-good symbol:
```python
data = plugin.load_market_data(
    pd.DataFrame({"symbol": ["BTC"]}),
    "2026-01-01",
    {"lookback_days": 30}
)
```

## Wide format vs long format confusion

**Cause:** Data source returns long-format (date, symbol, value) instead of wide (date x symbols).

**Solution:** Convert in your data plugin:
```python
# Long to wide conversion
wide_prices = long_df.pivot(index="date", columns="symbol", values="close")
wide_prices.index = pd.to_datetime(wide_prices.index)
```

## local_file_data can't read Parquet file

**Cause:** File path is wrong, file doesn't exist, or format isn't wide-format.

**Solution:**
1. Check path is correct: `ls -la ./data/curated/prices.parquet`
2. Verify it's wide format:
   ```python
   import pandas as pd
   df = pd.read_parquet("./data/curated/prices.parquet")
   print(df.head())  # Should show: DatetimeIndex x symbol columns
   ```

## Mismatched symbols between data and strategy

**Cause:** Data plugin returns symbols like "BTCUSDT" but strategy expects "BTC".

**Solution:** Normalize symbol names in your data plugin's `load_market_data()`:
```python
# Strip quote currency suffix
prices.columns = [c.replace("USDT", "") for c in prices.columns]
```

## Funding rates not available

**Cause:** Using `binance.live_data.v1` (spot) which doesn't provide funding rates.

**Solution:** Use `binance.futures_data.v1` for futures data with funding rates:
```yaml
plugins:
  data:
    name: "binance.futures_data.v1"  # has funding_rates
    params_init:
      quote_currency: "USDT"
```

## ccxt warning messages

**Cause:** `ccxt` package not installed. Some data plugins fall back to REST API.

**Solution:** These are warnings, not errors. To suppress, install ccxt:
```bash
uv sync --extra full
```
