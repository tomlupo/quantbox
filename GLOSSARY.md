# Glossary

One term per concept. Code, docs and PRs use these words and no synonyms.

## Lot rules

A venue's rules for the quantity of one order on one instrument: the quantity
step, the minimum quantity, the minimum notional, and whether a reduce-only
order is exempt from both minimums. They also fix the ORDER in which the rules
apply: round toward zero to the step, then the minimum notional on the rounded
quantity, then the minimum quantity.

Owner: `quantbox.trade.lots.LotRules`. A caller builds the lot rules from the
venue's market data (a ccxt market dict, or a market-snapshot row) and asks
`accept(qty, price, reduce_only=...)`. The answer is the signed quantity the
venue accepts, or a skip with its reason.

_Avoid_: "precision" for the step (ccxt's word; it means a step or a number of
decimals depending on the mode), "filters" (Binance's word), "lot size" for the
whole set (one rule of the set in `trade.allocations_to_orders.v1`).
