---
adr: 0007
title: Instrument and execution calendars, decision vs execution timing, venue.leverage and venue.financing
status: proposed
date: 2026-10-04
---

# ADR-0007: Instrument and execution calendars, decision vs execution timing, `venue.leverage` and `venue.financing`

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

Tom approved the design on 2026-10-04 (Telegram, msg 12553). Review round 1
(NO-SHIP at 9137ccf) found a fourth defect in the first fix, and Tom added three
decisions to the same change:

4. **Keeping the target on a bar the instrument did not print FILLED it at the
   stale price.** Decided at the 29 Dec close, lagged to 1 January (A not
   priced, B priced), the order filled at the forward-filled 29 Dec close and
   booked A's 2 January move — same-bar look-ahead for every multi-calendar
   book (478 TSMOM instrument-months rebalanced on such bars).
5. Instead of refusing an unfinanced levered vectorbt book: **`venue.leverage:
   normalize | borrow`** (Tom, msg 12557/12558).
6. **An explicit EXECUTION calendar** beside the per-instrument ones (msg 12564).
7. **Decision date and execution date kept apart** (msg 12566).

## Decision

### 1. A per-instrument life window, applied once at the engine seam

`quantbox.instrument_calendar.instrument_calendar` gives each instrument a life
window, from its first valid price to its last:

- **inside** the window a missing bar is forward-filled explicitly — to MARK
  the position, never to fill an order — and counted (`ffilled_bars`). The
  position is held across it. An order for the instrument on such a bar is
  DEFERRED to the instrument's own next printed bar (section 1c);
- **outside** it (before listing, after delisting) the target is forced to 0;
  prices are filled only to mark a flat book, never into a tradable position.
  A non-zero target there is counted per instrument
  (`targeted_outside_window_bars`, `max_abs_weight_outside_window`) and logged
  as a warning.

Rows where nothing prints and columns that never print are dropped and counted.
The 50%-coverage column drop is deleted: a short history is traded inside its
window. The columns the old rule (`< max(30, 50% of bars)` priced bars) WOULD
have dropped are counted in `calendar.legacy_coverage_drop`, so a result that
changed because of it says so.

**A trailing feed gap looks like a delisting.** The window ends at the last
print, so a feed that stops early is indistinguishable from a delisting: the
position is closed at the last printed price (the delisting return is assumed
0, a mild optimistic bias, unchanged from before). The per-instrument
`last_valid` in `data_validation.json` is where to look.

It runs once, in `BacktestPipeline._engine_book`, before the engine branch.
Both engines (vectorbt `from_orders`, vectorbt order-func, rsims) and both
flows (single run, variants) receive the same book. It is not in a data
plugin, because strategies must keep seeing the OBSERVED prices: the TSMOM
plugin's stale-price filter and volatility estimate read the gaps. "Data layer"
here means the data the ENGINE reads.

### 1b. The execution calendar

```yaml
execution:
  lag_bars: 1
  calendar: majority   # majority (default) | union | intersection | <ticker>
```

| value | a bar is an execution bar when |
|---|---|
| `majority` | at least 50% of the instruments inside their life window print on it |
| `union` | any instrument prints on it |
| `intersection` | every instrument inside its life window prints on it |
| `<ticker>` | that series prints on it (a reference index as the exchange calendar; it must be in the loaded prices, it need not carry a weight) |

PnL is still marked on every bar of the union index. Only DECISIONS and ORDERS
wait for execution bars. It is an `execution` key because it is timing: when a
decision can be taken and filled. No exchange-calendar dependency: the
calendar is read from the prices themselves (or from one series of them).

### 1c. decision vs execution timing

- **Decision bar.** The rebalance schedule (`rebalancing_freq`) picks decision
  bars ON THE EXECUTION CALENDAR (`quantbox.frequency.rebalancing_dates` over
  the execution bars). A period-END offset (`ME`, `BME`, `QE`, `YE`, `W-FRI`,
  `1W`) snaps BACKWARD to the last execution bar of the period — "monthly" is
  the last execution bar of the month, never a raw holiday row and never the
  first bar of the next month. Any other offset (`MS`, `BMS`, `D`) and an
  explicit date snap FORWARD. An integer `n` is every n-th execution bar;
  `None` (buy-and-hold) the first one. rsims decides on every execution bar.
  The weight decided on bar `d` is the strategy's row `d`, computed from data
  stamped on or before `d`.
- **Execution bar** = the decision bar plus `lag_bars` bars OF THE EXECUTION
  CALENDAR, not raw index rows. A lag of 1 raw row on a union index used to land
  on a holiday row and fill at stale prices.
- **Per instrument**, the execution bar is where the order is placed for every
  instrument that prints on it. One that is inside its window but did not print
  keeps its previous weight, and its order is DEFERRED to its own next printed
  bar (an execution bar or not). A later decision reaching the instrument first
  supersedes the deferred one. Counted per instrument (`deferred_trades`) and
  per rebalance (`deferred_instruments` in the schedule). The engines honour
  this through a per-cell order mask. rsims leaves the position as it is.
  vectorbt runs its flexible (order-function) path whenever the mask is given:
  an untouched cell sorts as a zero-value order (so sells still go before
  buys), and the financing legs are sized to the residual of what is ACTUALLY
  held after the bar's orders — a deferred position has drifted from its old
  target, and sizing the legs off the target starved the buys (118 underfilled
  TSMOM rebalances before this). The same path had valued the book at
  positions + FREE cash, which misstates every weight of a book with shorts;
  it now uses cash (this also corrects `threshold` runs that hold shorts).
- **Input staleness at decision.** For each instrument on each decision bar,
  the bars since its last real print — the age of the forward-filled input the
  signal saw. `data_validation.json` `staleness`: decisions on stale inputs,
  max and p95 age; per instrument `stale_decisions`, `max_staleness_bars`.
  Recorded, never blocking; TOM-1430 gates on it.
- **Decision weight age.** Price staleness says nothing about the WEIGHTS a
  decision trades: the strategy's newest row on or before the decision bar,
  forward-filled onto it. A strategy that stamps its weights on dates that are
  not execution bars — calendar month-ends that fall on a weekend, taken from
  a wider panel — writes its period's weights AFTER the decision bar; the
  engine seam drops that row and the previous period's weights are held
  until the next decision, silently (the lab's TSMOM re-run: 71 of 409
  executed decisions). The rule is about where weights are STAMPED, and it is
  deliberately the narrowest one that catches that bug. A decision is STALE
  when, strictly between the decision bar and the next execution bar, the
  strategy writes a STEP on a non-execution bar — weights that differ (by
  more than 1e-12 in any instrument) from the ones the decision traded and
  then stay unchanged up to that next execution bar, i.e. the strategy
  forward-fills its own stamp — AND the next decision comes after that
  execution bar, so the step is held for a period rather than traded at once
  (on an `int` 1 schedule or rsims, the next bar's decision trades it: never
  stale) — or the price/weight intersection (below) dropped price bars from
  that gap, so the shrunk calendar's "next execution bar" is a later stamp,
  not the next bar. It is read on the strategy's own rows, before the engine seam drops
  the ones that are not bars. It needs no notion of a period, so it holds for
  any schedule (a period-end offset, an `int`, explicit dates, rsims), and it
  reads only the gap after each decision bar, so a book that moves on
  execution bars inside the period (a weekday vol scaler on a month-end
  signal) does not mask it. Executed decisions only: one past the last bar
  never trades. `data_validation.json` `weight_age` (minor 1): executed
  decisions, the stale count, the first five stale decision bars and the
  stamps each missed, and `stale_held_bars_*` — bars from a stale decision's
  execution to the next decision's execution (max, total); metrics
  `decision_stale_weights`, `decision_stale_weights_held_bars_max` (0 when
  none is stale); a loud `TIMING:` warning. Recorded, never refused —
  TOM-1430 gates on it, so a false alarm costs more than a miss.
  History: round 1 asked when the weights CHANGED (unchanged since the period
  start AND changed later in it) — blind on a `BME` schedule and under any
  in-period move, false on `int` / `2W` schedules; round 2 flagged ANY changed
  non-execution row — 104 of 104 weekly decisions on a constant book times a
  7-day vol scaler, and on a daily signal over a crypto+equity panel.
  **Known false negative:** a stamped step with daily variation on top over
  the non-execution bars (a calendar month-end signal times a vol scaler that
  also moves on weekends) reads as drift and is NOT counted — weights that
  keep moving over a weekend cannot be told apart from a legitimate daily
  signal without knowing the strategy, and a noisy gate is worse than one
  honest blind spot. The reverse edge: a single non-execution bar in the gap
  (a mid-week holiday) is trivially "unchanged to the next bar", so a daily
  signal that moves on it counts when the schedule holds it for a period.
  **Known limit:** on intraday bars the rule compares timestamps exactly, but
  the calendar periods that pick decision bars are day-normalised; a stamp
  later on the decision bar's own day is not after it. Daily bars only, for
  now.
- **Index alignment.** The engine runs on the bars where prices AND a strategy
  weight row exist (the intersection is unchanged here — a separate card). A
  strategy that writes rows only on its stamp dates shrinks the whole panel
  to them. `data_validation.json` `index_alignment` (minor 3): price bars,
  weight rows, bars used, price bars dropped before the strategy's first row
  (warm-up, not warned), any OTHER dropped price bar (count, first five
  dates), weight rows on dates with no price bar, and of those the ones whose
  weights differ from the last kept row (count, first five dates); metrics
  `index_price_bars_dropped`, `index_weight_rows_dropped`,
  `index_weight_rows_dropped_changed`; a loud `INDEX:` warning for dropped
  price bars and, separately, for CHANGED dropped weight rows. A dropped row
  that repeats the last kept row is counted but not warned: it loses nothing
  (every 7-day-panel strategy has them), while a changed one is weights that
  are never traded.
- **Recorded.** `rebalance_schedule.parquet` has one row per executed decision:
  `decision_date`, `execution_date`, `deferred_instruments` (`;`-joined).
  `traded_weights.parquet` is now the book HELD after each bar's orders: the
  ordered cells at their targets AFTER `venue.leverage` (section 3), every
  deferred cell at its previous weight (before, it was the lagged decided
  weights, which between rebalances the vectorbt engine never held). The
  engine is checked against it on every run (`engine_underfilled_rebalances`).

**Recorded** (all of section 1). Every backtest run writes
`data_validation.json` (`quantbox/data-validation@1`, schema
`artifact_schemas/data_validation.schema.json`): `calendar` (policy, totals,
per-instrument rows), `execution_calendar` (calendar, execution bars against
total bars, non-execution bars per year), `timing`, `staleness`, `weight_age`
(minor 1), `index_alignment` (minor 3), `leverage` (its `max_net_exposure_unscaled`,
`scaled_after_deferral` and `buys_zeroed_dates`, written since minor 0, are required
from minor 2).
TOM-1430 adds its own sections beside them. run@1 minor 3 adds
`execution.calendar`, `venue.leverage` and `data_validation: {schema, file,
calendar, execution_calendar, timing, staleness, leverage}` (summaries) to the
manifest; explain@1 records `execution.calendar` and `venue.leverage` before
any data is loaded; run@1 minor 4 adds the `weight_age` and `index_alignment`
summaries. `metrics.json` carries `calendar_*`, `execution_calendar_*`,
`decision_*`, `index_*` and `leverage_*` counters.

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
rate is refused, never assumed to be 0. A ticker that stops printing before the
backtest ends reads as rate 0 after its last price; those bars are counted
(`rate_stale_bars_at_end`) and warned about.

### 3. `venue.leverage`: what a decision with net exposure above 1 becomes

```yaml
venue:
  allow_shorts: true
  leverage: normalize   # normalize | borrow; default normalize on vectorbt, borrow on rsims
```

- **`normalize`** (the vectorbt default) bounds the book HELD after each bar's
  orders, not the decided row: a deferred cell keeps its previous weight, so a
  rotation out of an instrument that did not print would otherwise ask for net
  above 1 (review round 2). On every bar whose held net exposure — ordered
  cells at target plus deferred cells at their weight — is above 1 (+1e-6),
  the ORDERED cells are scaled proportionally until the held net is 1. When
  the deferred cells alone are at or above 1, every buy ordered on that bar is
  set to 0 (`buys_zeroed_rebalances`, with dates, warned). With no deferral
  this is the decided row scaled to net 1. Recorded in `data_validation.json`
  `leverage`, the manifest and `leverage_*` metrics, and warned about:
  `rebalances_above_net_1` (on the held book), `scaled_rebalances`
  (`scaled_after_deferral` of them on a bar with a deferred cell), the mean,
  min and max scale, `max_net_exposure_decided` / `_unscaled` / `_held`.
- **`borrow`**: the decision is held as decided, financed by `venue.financing`
  (section 2). Without a financing block the rate is ASSUMED to be 0:
  `venue.financing` in run@1 / explain@1 is then `{rate: {annual: 0.0}, ...,
  assumed: true}`, and a run whose HELD book goes above net 1 (a deferral
  included) warns loudly.
  `risk.max_leverage` (a gross cap) still applies before.

Round 1 refused an unfinanced levered vectorbt book instead. The reviewer
counted about 84 levered vectorbt configs in quantbox-lab that would start
failing; Tom chose the two explicit modes (msg 12557). `normalize` is the
vectorbt default because vectorbt cannot borrow (it would cut the last buys
silently); `borrow` is the rsims default because rsims is a margin (notional)
simulator with no cash floor — that is what it has always done, so the 26 lab
rsims configs keep their leverage behaviour. With `borrow` and no block, rsims gets no cash
legs (an assumed rate of 0 is its own behaviour); vectorbt gets zero-rate legs,
so it can hold the book.

**Measured, not assumed.** Every vectorbt run records
`engine_underfilled_rebalances` and `engine_max_fill_gap` — a threshold run
too, on the bars the engine traded (it skips the others by design). On each
rebalance bar they compare the book the engine HELD against `traded_weights`,
on the cells that were ordered. The tolerance is 1e-6 plus twice the cost times
turnover. Any underfill is logged as a warning. It is the backstop. Leverage
applied to the held book removes the deferral case by construction, but the
held book is a TARGET path: under `normalize` with no financing legs, a
deferred position that DRIFTS above its old target can still leave too little
cash for a funding buy (review round 3: 5 of 250 bars, max gap 0.15%, at 1%
daily vol with 13 deferrals). Small, and loud when it happens — not
guaranteed 0.

### 4. The rebalance schedule is bars

`quantbox.frequency.rebalancing_dates` is engine-free and returns members of
the bars it is given — in the pipeline, the execution bars (section 1c). A
calendar date that is not one snaps backward (period-end offsets) or forward
(everything else). `vectorbt_engine.get_rebalancing_dates` delegates to it.

## Consequences

`feat!`: configs that ran before change their numbers. The commit that lands
round 1 carries the `BREAKING CHANGE:` footer.

- **Numbers change.**
  - Holiday bars hold their weight; an order on a bar the instrument did not
    print waits for its next print.
  - Decisions sit on the execution calendar and execute `lag_bars` execution
    bars later. A schedule now names the DECISION bar: `rebalancing_freq: 5`
    decides on bars 0, 5, 10 and trades on 1, 6, 11 (before: traded on 0, 5,
    10 on the previous bar's decision); `BMS` decides on the first bar of the
    month and trades on the second. A config that meant "decide at month-end,
    trade on the first day" says `ME`. The canonical `momentum` golden moved
    from -13.9% to +9.9% total return for this phase shift alone (checked: the
    old phase as an explicit decision list reproduces -0.138805 exactly).
  - Net exposure above 1 is scaled to 1 on vectorbt unless `venue.leverage:
    borrow`.
  - Short-history instruments are no longer dropped (counted:
    `legacy_coverage_drop`).
  - Period-end schedules on data without those dates decide on the last bar of
    the period; start schedules snap to the next bar.
  - Financed books earn rf on idle cash.
  - The TSMOM line (quantbox-lab `replication-tsmom-mop2012`, decisions `ME`,
    financing LT12TRUU at spread 0; monthly Sharpe, excess of LT12TRUU):

    | variant | IS 1991-02..2009-12 | full 1991-02..2025-01 |
    |---|---|---|
    | `leverage: borrow` | 1.045 | 0.751 |
    | `leverage: normalize` (52 held-book rebalances scaled, 10 with buys zeroed; mean 0.82) | 1.006 | 0.722 |
    | constant scale (borrow / mean gross 12.54, at rf) | 1.044 | 0.746 |

    Reference (DEFR, lagged): 1.075 IS. Round 1 (which filled at the stale
    price) gave 1.087; this round — 520 orders deferred to a printed bar,
    decisions and lag on the execution calendar — gives 1.045. Not attributed
    further: the pieces were not run separately. `normalize` read 1.047 /
    0.755 until review round 2: deferral had silently lifted its held net
    above 1. These runs predate the decision weight age (above), which found
    the lab strategy trading the previous month's weights on 71 of its 409
    executed decisions (410 decided; the last falls past the data and never
    executes); the numbers are the strategy's as written, not a corrected
    replication.
- `traded_weights.parquet` is the held book; `traded_*` metrics (turnover,
  flat-bar share) measure it.
- Declaring `financing` or `leverage` requires a `venue` block, and `venue`
  requires `allow_shorts`, as before.
- `risk.max_leverage` still caps GROSS exposure before scheduling. It does not
  cap net, and the cash legs do not count towards it.
- Two cases still cut by fees alone. A book at net exactly 1 with fees, or one
  with financing whose fees exceed its cash, loses the fee amount from its last
  buy. That is pre-existing fee drag, not leverage. The fill-gap tolerance
  allows for it.
- `quantbox sweep` refuses `execution.calendar`: the sweep engine trades the
  bars it is given.
- The L1 helpers `backtest()` / `optimize()` call the vectorbt primitive
  directly and get neither the calendars nor `venue.leverage`/financing; they
  still cut net > 1 silently. A follow-up card, not this change.

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
- **Refuse an unfinanced levered vectorbt book** (round 1). Replaced by
  `venue.leverage` (section 3): a refusal broke ~84 lab configs that only need
  a declared choice.
- **Fill at the forward-filled price on a bar the instrument did not print**
  (round 1). Rejected by review: same-bar look-ahead.
- **An exchange-calendar library** (`exchange_calendars`,
  `pandas_market_calendars` sessions) for the execution calendar. Rejected: no
  new dependency, and the data's own prints are the calendar the backtest can
  actually trade; `<ticker>` covers "this index IS the exchange calendar".
