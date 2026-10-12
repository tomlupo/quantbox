---
adr: 0012
title: The trading pipeline owns the one alert channel; brokers send nothing, and per-order fill pings are dropped
status: accepted
date: 2026-10-12
supersedes:
superseded_by:
amends:
status_changes:
  - 2026-10-12: accepted with TOM-1720 (quantbox#282) — Tom, 2026-10-10 (P5 grilling Q2/Q12), card TOM-1720: "The brokers' own notify / send_telegram paths are removed"; Tom, 2026-10-12, asked whether fill pings move to the pipeline: "Drop them (card as written)"
---

# ADR-0012: The trading pipeline owns the one alert channel; brokers send nothing, and per-order fill pings are dropped

## Context

Before TOM-1720 the trading pipeline raised its alerts through the broker
(`getattr(broker, "notify")`). Only `HyperliquidBroker` had a `notify`, so on
Kraken, a live book, every HALT, FROZEN and order-failure alert was dropped
without a trace. Two brokers also sent Telegram messages of their own:
Hyperliquid a message per order outcome (FILLED, PARTIAL, FAILED) and on a failed
balance read; Binance futures a "submitted" message per order. Their standalone
`rebalance_to_weights` also sent summaries and a daily-loss HALT, but nothing in
quantbox calls that path.

Moving the alerts to the pipeline raised one question the card's words answer
but a reader of the code cannot: do the per-order fill and partial-fill messages
move to the pipeline too, or stop?

## Options considered

1. **The pipeline sends a fill / partial message per filled order, on every
   venue.** Hyperliquid keeps its messages; Kraken and Binance futures gain them.
2. **The same, Hyperliquid only.** The exact old behaviour, through the new
   channel.
3. **Drop them** (chosen). The card says the brokers' own send paths are removed.

## Decision

- `quantbox.trade.notifier.Alerts`, owned by the trading pipeline, is the one
  alert channel, on every venue. Its sites are the eight the pipeline already
  had: order_failures, recon_halt, frozen, trapped_residual, zero_fills,
  recon_preflight, recon_persist, recon_transition.
- Brokers send no message of their own: no `notify`, no `send_telegram`, no
  Telegram params.
- **Per-order fill and partial-fill messages are dropped, not moved** (Tom,
  2026-10-12: "Drop them (card as written)"). An alert is an exception a human
  must act on; a fill is not one. Fills are in quantbox-live's Tier-1 daily
  glance; a failed order still alerts through `order_failures`; a resting
  (WORKING) order is still never alerted as a failure.
- A failed balance read raises `BrokerExecutionError` and no longer sends its own
  message. The run either aborts (quantbox-live's `run_daily.sh` sends a
  pipeline-failure alert) or records a refused valuation as `pipeline_ok: false`,
  which quantbox-live's notifier turns into a Tier-2 "run failed" alert.

## Consequences

- (+) Kraken and Binance futures alert on HALT, FROZEN and order failures for the first time.
- (+) One place decides what reaches Tom's phone, and `notes.alerts` records every send, failure and "alerts off".
- (−) Accepted: carver-HL stops sending a message per filled order.
- The tests that asserted the broker's per-order messages
  (`test_hyperliquid_fill_still_alerts` and its two siblings) retire with
  `_notify_order_outcome`; this ADR is the requirement change that retires them.
  The WORKING-is-not-a-failure property survives on the pipeline's channel
  (`tests/test_trade_notifier.py::test_a_working_order_is_not_alerted_as_a_failure`).
