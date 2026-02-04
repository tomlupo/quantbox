# trade.allocations_to_orders.v1

Trading bridge pipeline:
- input: `allocations.parquet` with columns `symbol, weight`
- uses broker positions/cash + latest close prices to generate `targets` and `orders`
- in paper/live mode: calls broker.place_orders -> `fills`
- writes `portfolio_daily` summary

This is intentionally simple (integer share sizing, USD-only).
