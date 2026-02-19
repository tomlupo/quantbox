# Broker Gotchas

## BrokerExecutionError: insufficient balance

**Cause:** Trying to place orders that exceed available cash or margin.

**Solution:** Check balances before execution:
```python
cash = broker.get_cash()
positions = broker.get_positions()
```

For paper brokers, increase starting capital:
```yaml
plugins:
  broker:
    name: "sim.paper.v1"
    params_init:
      cash: 500000  # increase from default
```

## API key not set

**Cause:** Live broker can't find environment variables.

**Solution:** Set env vars before running:
```bash
export API_KEY_BINANCE="your_key"
export API_SECRET_BINANCE="your_secret"
```

Or use a `.env` file (copy from `.env.example`).

Verify (without printing values):
```bash
python -c "import os; print(bool(os.getenv('API_KEY_BINANCE')))"
```

## Paper broker has wrong portfolio value

**Cause:** Using `get_cash() + sum(positions * prices)` for derivatives,
which gives wrong results for short positions and futures.

**Solution:** Use `get_equity()` if available:
```python
if hasattr(broker, "get_equity"):
    total_value = broker.get_equity()
else:
    total_value = sum(broker.get_cash().values())
```

Pipelines prefer `get_equity()` when available.

## Broker not needed but config requires it

**Cause:** Using `fund_selection.simple.v1` or `backtest.pipeline.v1` which don't
execute trades, but config has a `broker` section.

**Solution:** Remove the `broker` section for research/backtest pipelines, or leave it
(it will be ignored). The broker is only invoked when `mode` is `paper` or `live`.

## Futures broker with spot rebalancer

**Cause:** Using `sim.futures_paper.v1` with `rebalancing.standard.v1` (spot).

**Solution:** Match broker and rebalancer types:
- Spot broker (`sim.paper.v1`) + `rebalancing.standard.v1`
- Futures broker (`sim.futures_paper.v1`) + `rebalancing.futures.v1`

## Orders DataFrame has wrong columns

**Cause:** `place_orders()` receives DataFrame without expected columns.

**Solution:** Orders DataFrame must have: `symbol`, `side`, `qty`.
Optional: `order_type`, `price`, `leverage`.

```python
orders = pd.DataFrame([
    {"symbol": "BTC", "side": "buy", "qty": 0.1},
    {"symbol": "ETH", "side": "sell", "qty": 1.0},
])
```

## describe() not implemented

**Cause:** Custom broker doesn't implement the optional `describe()` method.

**Solution:** `describe()` is optional but recommended for LLM agents. Add it:
```python
def describe(self) -> dict:
    return {
        "plugin": self.meta.name,
        "paper_trading": self.paper_trading,
        "cash": self.get_cash(),
        "positions": dict(self._positions),
    }
```
