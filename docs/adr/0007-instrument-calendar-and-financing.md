---
adr: 0007
title: Per-instrument calendar at the engine seam, and venue.financing for borrowed and idle cash
status: proposed
date: 2026-10-04
---

# ADR-0007: Per-instrument calendar at the engine seam, and `venue.financing` for borrowed and idle cash

## Context

The TSMOM replication (quantbox-lab `research/replication-tsmom-mop2012`,
`results/engine-parity/`, TOM-1429) compared `backtest.pipeline.v1` with an
independent reference on identical data. It confirmed two engine-made defects
worth -0.13 Sharpe between them:

1. **A missing price zeroed the target.** `_align_for_engine` ran
   `weights.where(prices.notna(), 0.0)`. In a panel of instruments with
   different calendars a NaN is usually a holiday (1 January prints for some
   indices and not others), and on a rebalance bar the zero is held for the
   whole holding period: 478 of 5920 instrument-months, -0.10 Sharpe. The
   comment said the intent was "not yet listed". The same function also dropped
   any instrument with under 50% price coverage, silently.
2. **vectorbt cannot borrow.** `Portfolio.from_orders(cash_sharing=True)` never
   takes cash below zero, so when the traded book's net exposure is above 1 the
   last buys in the call sequence (buy-backs of shorts included) are cut
   silently: 58 of 409 rebalances, -0.03 Sharpe.

A third, latent: `get_rebalancing_dates` built calendar dates with
`pd.date_range` and the engine matched them with `index.isin`, so a scheduled
date that is not a bar (a holiday `BMS`, a weekend month-end, every `W-SUN` on
weekday data) silently skipped that period's rebalance.

Tom approved the design on 2026-10-04 (Telegram, msg 12553).

## Decision

### 1. A per-instrument life window, applied once at the engine seam

`quantbox.instrument_calendar.apply_instrument_calendar` gives each instrument
a life window, from its first valid price to its last:

- **inside** the window a missing bar is forward-filled explicitly, the target
  weight is KEPT, and the bar is counted (`ffilled_bars`);
- **outside** it (before listing, after delisting) the weight is forced to 0;
  prices are filled only to mark a flat book, never into a tradable position.
  A non-zero target there is counted per instrument
  (`targeted_outside_window_bars`, `max_abs_weight_outside_window`) and logged
  as a warning.

Rows where nothing prints and columns that never print are dropped and counted.
The 50%-coverage column drop is deleted: a short history is traded inside its
window.

It runs in `BacktestPipeline._align_with_calendar`, after the execution lag
and before the engine branch. Both engines (vectorbt `from_orders`, vectorbt
order-func, rsims) and both flows (single run, variants) receive the same book.
It is not in a data plugin, because strategies must keep seeing the OBSERVED
prices: the TSMOM plugin's stale-price filter and volatility estimate read the
gaps. "Data layer" here means the data the ENGINE reads.

**Recorded.** Every backtest run writes `data_validation.json`
(`quantbox/data-validation@1`). It has one section per check: today
`calendar`, with a policy, totals and per-instrument rows. TOM-1430 adds its
own sections beside it. run@1 minor 3 adds `data_validation: {schema, file,
calendar: <totals>}` to the manifest. `metrics.json` carries
`calendar_ffilled_bars` and `calendar_targeted_outside_window_bars`.

### 2. `venue.financing`: rf + spread, through synthetic cash legs

```yaml
venue:
  allow_shorts: true
  financing:
    rate: "LT12TRUU Index"   # a ticker in the loaded prices (a cash TR index; its bar return is the rate)
                             # or a number: a constant annual rate, ACT/365 (0.0 = free)
    borrow_spread_bps: 0     # borrowed cash pays rate + this
    lend_spread_bps: 0       # idle cash earns rate - this
```

**Why `venue`, not `execution`.** `execution` is timing: when a decided weight
fills. `venue` answers "could this book have existed?". `allow_shorts` already
answers that. A levered book exists only where something lends, and the price
of lending is the venue's (or the broker's). Like `venue`, `financing` is
run-level: a variant cannot override it.

**Mechanism** (`quantbox.financing.add_cash_legs`). This is engine-agnostic and
runs after the calendar. Two synthetic assets are appended to the engine's
book:

- `LEND`, held at `max(1 - sum(w), 0)`, priced by compounding `rate - lend_spread`;
- `BORROW`, held at `min(1 - sum(w), 0)` (short), priced by compounding
  `rate + borrow_spread`.

The engine's book then sums to exactly 1, so vectorbt never runs out of cash.
The financing P&L is the legs' mark-to-market. Splitting the residual by sign
makes the asymmetric spread exact: a leg never changes sign. The legs trade
free of fees and slippage (`fee_free=` on both engine primitives). They are not
part of `traded_weights.parquet`, which stays the real book. The realised
financing (mean, min and max cash weight, borrow-bar share) goes in
`RunResult.notes["financing"]` and `financing_*` metrics. The resolved block is
`venue.financing` in run@1 and explain@1 (`null` when undeclared).

A ticker rate must have printed on or before the first backtest bar. An unknown
rate is refused, never assumed to be 0.

### 3. No `financing` block: a book that needs borrowing is REFUSED on vectorbt

Without `venue.financing`, a vectorbt run is refused before the engine runs
when the traded book's net exposure is above 1 (+1e-6) on any rebalance bar.
The error names the block and the count. `rate: 0.0` restores "borrowing is
free, idle cash earns nothing".

We chose refusal over keeping the old behaviour with a warning and a counter,
for three reasons:

- A cut book is not the book the strategy designed. The run would publish
  numbers for some other strategy, which is exactly what ADR-0005 refused for
  same-bar fills. A warning above a result enforces nothing.
- Rejecting at `validate` time is impossible: net exposure is a property of the
  strategy's output, not of the config. The refusal fires after the strategies
  and before the engine, which is the earliest point the fact exists.
- The fix is one line, and the error message spells it out.

rsims is not refused. It is a margin (notional) simulator with no cash floor,
so it cuts nothing. Its idle cash earns nothing and its borrowing is free unless
`financing` says otherwise. With `financing`, rsims receives the same legs.

**Measured, not assumed.** Every vectorbt run without a threshold now records
`engine_underfilled_rebalances` and `engine_max_fill_gap`. On each rebalance
bar these compare the book the engine HELD against the target it was handed.
The tolerance is 1e-6 plus twice the cost times turnover. Any underfill is
logged as a warning.

### 4. The rebalance schedule is bars

`quantbox.frequency.rebalancing_dates` is engine-free. A calendar or explicit
date that is not a bar snaps FORWARD to the first bar on or after it, which is
the first moment an order could trade. `vectorbt_engine.get_rebalancing_dates`
delegates to it.

## Consequences

- **Numbers change.**
  - Books with holiday rebalance bars now hold their targets.
  - Short-history instruments are no longer dropped.
  - Weekly and month-end schedules on data without those dates now rebalance.
  - Financed books earn rf on idle cash.
  - The TSMOM line with `financing: {rate: "LT12TRUU Index"}`: IS (1991-02..2009-12)
    Sharpe in excess of LT12TRUU is 1.087, against the reference's lagged
    1.075 and 0.899 as run on v0.9.0. On total returns it is 1.314, because
    idle cash now earns rf.
- **A levered vectorbt config without `venue.financing` now fails loudly.** The
  TSMOM line's `tsmom` arm is one: net exposure is above 1 on 39 of 409
  rebalances.
- Declaring `financing` requires a `venue` block, and `venue` requires
  `allow_shorts`, as before.
- `risk.max_leverage` still caps GROSS exposure before the lag. It does not
  cap net, and the legs do not count towards it.
- Two cases still cut by fees alone. A book at net exactly 1 with fees, or one
  with financing whose fees exceed its cash, loses the fee amount from its last
  buy. That is pre-existing fee drag, not leverage. The fill-gap tolerance
  allows for it.
- The L1 helpers `backtest()` / `optimize()` call the vectorbt primitive
  directly and get neither the calendar nor financing. vectorbt's own
  `validate_prices` + ffill already keeps holiday targets there, but they still
  cut net > 1 silently. That is a follow-up.

## Alternatives considered

- **Ffill prices in the data plugin.** Rejected. Strategies need the observed
  gaps (stale filters, volatility on observed returns), and the engine seam is
  the one place every engine passes through.
- **Let vectorbt borrow** (`lock_cash`, a negative cash floor). vectorbt 1.0's
  `from_orders` has no borrowing knob. An engine-specific fix would also tie
  the semantics to vectorbt, which TOM-1325's engine seam forbids.
- **One signed cash leg.** Rejected. The asymmetric spread would depend on the
  leg's sign, which only the engine knows between rebalances. Two legs, each
  with a fixed sign, make it exact without engine state.
- **Keep today's behaviour plus a warning and a counter when unfinanced.**
  Rejected for the reasons in section 3. The counter exists anyway, as a
  measurement on every vectorbt run.
