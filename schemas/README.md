# Artifact Schemas

JSON schemas for all QuantBox run artifacts. Used by `runner.py` for
post-run validation and by LLM agents for contract inspection.

| Schema | Artifact | Written by |
|---|---|---|
| `universe.schema.json` | Symbol universe | data plugins |
| `prices.schema.json` | OHLCV prices | data plugins |
| `scores.schema.json` | Strategy scores | strategy plugins |
| `rankings.schema.json` | Ranked universe | fund selection pipeline |
| `allocations.schema.json` | Target allocations | fund selection pipeline |
| `strategy_weights.schema.json` | Target weights | strategy plugins |
| `aggregated_weights.schema.json` | Blended weights | weighted aggregator |
| `targets.schema.json` | Pre-trade targets | trading pipeline |
| `orders.schema.json` | Orders submitted | rebalancing plugins |
| `fills.schema.json` | Executed fills | broker plugins |
| `rebalancing.schema.json` | Rebalancing report | rebalancing plugins |
| `portfolio_daily.schema.json` | Daily portfolio snapshot | trading pipeline |
| `trade_history.schema.json` | Historical trades | broker plugins |
| `fx.schema.json` | FX rates | data plugins |
