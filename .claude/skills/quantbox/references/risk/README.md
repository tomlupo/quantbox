# Risk

Risk plugins validate portfolio targets and orders against configurable limits.
They run as pre-trade checks: all risk plugins must pass before orders are executed.

## When to Use

- You need pre-trade validation (position limits, leverage caps)
- You want to add stress testing to your pipeline
- You need to implement custom risk rules

## Available Risk Plugins

<!-- BEGIN AUTO-GENERATED -->
| ID | Description | Tags |
|---|---|---|
| `risk.stress_test.v1` | Stress-test risk plugin. Runs Monte Carlo stress scenarios (2008 crisis, COVID crash, etc.) and flags portfolios whose VaR, CVaR, or drawdown breach thresholds. | risk, stress-test, simulation |
| `risk.trading_basic.v1` | Basic trading risk checks: leverage, concentration, negative weights, min notional, max order size. | trading, risk |
<!-- END AUTO-GENERATED -->

## Risk Flow

```
strategy outputs weights
  -> risk.check_targets(weights, params)     # validate weights
  -> rebalancer generates orders
  -> risk.check_orders(orders, params)       # validate orders
  -> if all pass: broker executes
  -> if any fail: pipeline logs findings, may skip execution
```

Both methods return `list[dict]` of findings. Empty list = all checks passed.

## Config Examples

### Basic risk limits
```yaml
plugins:
  risk:
    - name: "risk.trading_basic.v1"
      params:
        max_position_pct: 0.25     # no single position > 25%
        max_leverage: 1.0           # no leverage (spot)
        max_notional: 50000         # max notional per position
```

### Multiple risk plugins (both must pass)
```yaml
plugins:
  risk:
    - name: "risk.trading_basic.v1"
      params:
        max_position_pct: 0.30
        max_leverage: 3.0
    - name: "risk.stress_test.v1"
      params: {}
```

### No risk checks
```yaml
plugins:
  risk: []
```

## Next Steps

- **Create a custom risk plugin**: Load [api.md](api.md)
