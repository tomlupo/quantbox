# Strategy Gotchas

## Empty weights returned

**Cause:** Strategy's lookback period exceeds available data.

**Solution:** Ensure `lookback_days` in strategy params is less than the data range.
If using `binance.live_data.v1`, the data plugin fetches based on `params.lookback_days`
in the pipeline config, not the strategy config. Both must be sufficient.

```yaml
# Ensure pipeline lookback >= strategy lookback
plugins:
  pipeline:
    params:
      universe:
        lookback_days: 400   # must be >= strategy's lookback
  strategies:
    - name: "strategy.crypto_trend.v1"
      params:
        lookback_days: 365   # needs 365 days of data
```

## NaN values in weights

**Cause:** Missing data in input DataFrames (gaps, delistings, or insufficient history for
rolling calculations).

**Solution:** Forward-fill prices before signal computation, and handle edge cases:
```python
prices = data["prices"].ffill()
# Drop symbols with too many NaNs
valid = prices.isna().sum() < len(prices) * 0.1
prices = prices.loc[:, valid]
```

## Weights don't sum to 1.0

**Cause:** The `run()` method returns unnormalized signals or raw position sizes.

**Solution:** Normalize in the strategy:
```python
row_sum = weights.sum(axis=1).replace(0, np.nan)
weights = weights.div(row_sum, axis=0)
```

Or let the pipeline / rebalancer handle normalization (some pipelines normalize automatically).

## Wrong data format (long vs wide)

**Cause:** Strategy receives long-format DataFrame instead of wide-format.

**Solution:** QuantBox uses wide format everywhere. If you have long format:
```python
# Convert long to wide
wide_prices = long_df.pivot(index="date", columns="symbol", values="close")
```

The DataPlugin is responsible for providing wide-format data. If implementing a custom
DataPlugin, ensure `load_market_data()` returns wide DataFrames.

## `meta` as instance attribute

**Cause:** Defining `meta` inside `__init__` or `__post_init__` instead of as a class attribute.

**Solution:** `meta` must be a **class-level** attribute, not per-instance:
```python
# CORRECT
@dataclass
class MyStrategy:
    meta = PluginMeta(name="strategy.mine.v1", kind="strategy", ...)

# WRONG - will fail registration
@dataclass
class MyStrategy:
    def __post_init__(self):
        self.meta = PluginMeta(...)
```

## Strategy not appearing in plugin list

**Cause:** Missing one of the three registration steps.

**Solution:** All three are required:
1. Export from `plugins/strategies/__init__.py`
2. Import in `plugins/builtins.py`
3. Add to the `"strategy": _map(...)` line in `builtins()`

Verify with: `uv run quantbox plugins list | grep strategy`

## Param overrides not taking effect

**Cause:** The `run()` method doesn't apply `params` dict to instance attributes.

**Solution:** Add the standard param override pattern:
```python
def run(self, data, params=None):
    if params:
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
    # ... rest of strategy
```

## Optional dependency not available

**Cause:** Strategy requires optional packages (scikit-learn, quantstats, etc.)
that aren't installed.

**Solution:** Guard imports and fail gracefully:
```python
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# In run():
if not SKLEARN_AVAILABLE:
    raise ImportError("Install scikit-learn: uv sync --extra ml")
```

Install all optional deps: `uv sync --extra full`
