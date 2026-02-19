# Config Gotchas

## ConfigValidationError: plugin not found

**Cause:** Plugin name in config doesn't match any registered plugin.

**Solution:** Check for typos. Get valid names:
```bash
uv run quantbox plugins list
```

Common mistakes:
- `publisher.telegram.v1` (wrong) -> `telegram.publisher.v1` (correct)
- `binance.data.v1` (wrong) -> `binance.live_data.v1` (correct)
- `paper.broker.v1` (wrong) -> `sim.paper.v1` (correct)

## run.pipeline doesn't match plugins.pipeline.name

**Cause:** The two pipeline references in the config are different.

**Solution:** They must be identical:
```yaml
run:
  pipeline: "trade.full_pipeline.v1"     # MUST match
plugins:
  pipeline:
    name: "trade.full_pipeline.v1"       # MUST match
```

## params vs params_init confusion

**Cause:** Putting constructor arguments in `params` instead of `params_init`,
or vice versa.

**Solution:**
- `params_init` = constructor arguments (file paths, API keys, cash, quote_currency)
- `params` = runtime arguments (lookback_days, top_n, universe symbols)

```yaml
# CORRECT
plugins:
  data:
    name: "local_file_data"
    params_init:                    # constructor args
      prices_path: "./data/prices.parquet"
  strategies:
    - name: "strategy.crypto_trend.v1"
      params:                       # runtime args
        lookback_days: 365

# WRONG - will fail or be ignored
plugins:
  data:
    name: "local_file_data"
    params:                         # wrong! prices_path is a constructor arg
      prices_path: "./data/prices.parquet"
```

## Missing required plugin section

**Cause:** Config is missing a required section (e.g., `data` is always required).

**Solution:** Every config needs at minimum:
```yaml
run:
  mode: ...
  asof: ...
  pipeline: ...
artifacts:
  root: ...
plugins:
  pipeline: { name: ... }
  data: { name: ... }
```

Optional sections: `strategies`, `broker`, `rebalancing`, `risk`, `publishers`, `aggregator`.

## Empty risk/publishers lists

**Cause:** Config has `risk:` or `publishers:` without any entries.

**Solution:** Use empty list syntax:
```yaml
plugins:
  risk: []         # valid - no risk checks
  publishers: []   # valid - no notifications
```

Not:
```yaml
plugins:
  risk:            # this is null, not empty list - may cause issues
```

## Profile overrides not working

**Cause:** Profile sets defaults, but explicit config sections should override them.

**Solution:** Explicit sections always win over profile defaults:
```yaml
plugins:
  profile: "trading"           # sets broker to sim.paper.v1
  broker:
    name: "binance.live.v1"    # overrides profile's broker
    params_init: {}
```

## YAML indentation errors

**Cause:** YAML is indentation-sensitive. Incorrect nesting breaks parsing.

**Solution:** Use consistent 2-space indentation. Validate:
```bash
uv run quantbox validate -c configs/my_config.yaml
```

Common mistake - strategies must be a list (note the `-`):
```yaml
# CORRECT (list)
strategies:
  - name: "strategy.crypto_trend.v1"
    weight: 1.0

# WRONG (not a list)
strategies:
  name: "strategy.crypto_trend.v1"
  weight: 1.0
```

## Secrets in YAML

**Cause:** API keys or tokens hardcoded in config files.

**Solution:** Never put secrets in YAML. Use env var references:
```yaml
plugins:
  publishers:
    - name: "telegram.publisher.v1"
      params_init:
        token_env: "TELEGRAM_TOKEN"        # env var NAME, not the value
        chat_id_env: "TELEGRAM_CHAT_ID"    # env var NAME, not the value
```
