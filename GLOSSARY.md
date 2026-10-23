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

## Residual

An exit the venue rejects at the book's size, so the order is not sent and the
exposure stays: a partial reduce below the venue minimum (TOM-1778), or a full
close the venue can never accept, dust below the minimum quantity with no
reduce-only exemption or a quantity that rounds to zero (TOM-2212). A residual
is listed on the run, never a freeze (Tom, 2026-10-12; 2026-10-23).

Owner: the rebalancer writes `RESIDUAL_STATUS` on the row
(`quantbox.plugins.rebalancing.futures_rebalancer`), and
`LotRules.never_accepts_close` says which full closes qualify.
`quantbox.trade.executor` lists every residual and reads a day with nothing
else trapped as a residual day.

_Avoid_: "trapped residual" for this. That is #81's word for an exit the broker
skipped although the rebalancer judged it placeable, and that still freezes.
