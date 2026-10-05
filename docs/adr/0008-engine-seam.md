---
adr: 0008
title: Book simulation sits behind one engine seam, with vectorbt and rsims as adapters
status: accepted
date: 2026-10-04
supersedes:
superseded_by:
amends: "ADR-0001 § the adapter rule (a re-export, never an opaque wrapper): book simulation is the one capability that sits behind a seam; the native object stays reachable. ADR-0007: rsims no longer decides on every execution bar, and the venue.leverage default is one value."
status_changes:
  - 2026-10-04: proposed with TOM-1447 (P3a of the quantbox 1.0 spec, TOM-1325)
  - 2026-10-05: amended and accepted with TOM-1450 (3d-1) — Tom, 2026-10-05: one book builder for every door (alternative B), the seam owns the rebalancing schedule, the engines stay separate under it
  - 2026-10-05: decision 11 added with TOM-1450 (3d-2) — the rebalancing policies and group limits are seam semantics
  - 2026-10-05: decision 12 added with TOM-1500 — the same defaults on every engine (compounding, starting cash), and every cost charged or refused
---

# ADR-0008: Book simulation sits behind one engine seam, with vectorbt and rsims as adapters

## Context

Turning decided weights into a traded book was implemented five times: the
single run and the variants flow of `backtest.pipeline.v1`, the parameter
sweep, `backtest()` and `optimize()`. Each door had its own copy of the
lag, the NaN policy and the engine call. Code compared the engine name
(`engine == "rsims"`, `engine != "vectorbt"`) in the pipeline, the
schedule, the NaN policy, the leverage default and the run manifest.
The variants flow refused rsims. The sweep, `backtest()` and `optimize()`
ran vectorbt only.

Two engines exist and both are needed: vectorbt (open source, the
`[vectorbt]` extra; research, lab, live) and rsims (numpy, in core; perps
with funding and margin; the only engine a client install will carry). Two
implementations of one capability is what makes a seam real.

ADR-0001 says an adapter is a thin re-export and never hides the library.
That rule fits a library with one implementation. It does not fit a
capability with two, where the caller should pick one by name and get the
same shape back.

The first version of this ADR (TOM-1447) kept two book builders behind the
seam: the scheduled book of ADR-0007 for `quantbox run`, and a bar-grid book
for `backtest()`, `optimize()` and the sweep. Each adapter also kept its own
schedule semantics (rsims decided on every bar and ignored `rebalancing_freq`
and `threshold`), its own NaN policy (vectorbt held, rsims went flat) and its
own `venue.leverage` default (normalize on vectorbt, borrow on rsims). The
same config therefore gave a different book when only `engine` changed, and
the same strategy gave a different book through `backtest()` than through
`quantbox run`. Tom decided on 2026-10-05 (TOM-1450) to close both gaps.

## Decision

1. **One seam: `quantbox.engine`, one book function: `simulate()`.** Decided
   weights + prices + execution timing + the rebalancing schedule + costs go
   in; a `TradedBook` comes out: `returns`, `value`, `weights`, `orders`,
   `turnover`, `trades`, `metrics` and `native`.
2. **Every door builds its book with `simulate()`**: the single run, the
   variants, `backtest()`, `optimize()`, the sweep and `run_grid`. It is the
   calendar-aware scheduled book of ADR-0007. A book with several strategy
   slices (a dict of frames, the sweep's MultiIndex columns) is batching
   inside the one function, one schedule per slice, one engine call.
3. **`execution.schedule: calendar | bars`.** `calendar` (the default) is
   the scheduled book. `bars` is the same function on a DEGENERATE calendar —
   every price bar is an execution bar on which every instrument prints, so
   there is no deferral, and `venue.leverage` is measured, never applied
   (declaring `venue.leverage` or `venue.financing` with it is refused). It is
   not a second code path. `simulate_weights` and its `leading='flat'|'drop'`
   modes are deleted.
4. **The seam owns the rebalancing schedule.** `rebalancing_freq` and
   `threshold` produce a per-cell ORDERS mask inside the seam
   (`quantbox.engine.schedule`); every engine follows it. rsims now trades on
   the same bars as vectorbt. `threshold` is a drift trigger the seam computes
   from the cost-free price drift of the weights held since the last placed
   rebalance (see the caveat below).
5. **Thin adapters.** An adapter only executes:
   `execute(prices, targets, orders, costs, funding, params) -> TradedBook`.
   It owns its parameters (unknown keys are refused; rsims keeps
   `trade_buffer` and its other own params) and the normalisation of its
   output, and it declares two capability differences that are real:
   `charges_funding` and `models_margin`. The flags `decides_every_bar`,
   `nan_policy` and `default_leverage` are deleted.
6. **Engine-independent semantics live in the seam, decided once.** The NaN
   policy is HOLD (a NaN cell keeps the last decided target, a leading NaN
   is 0) for every engine; adapters receive fully specified targets. The
   default `venue.leverage` is `normalize` for every engine
   (`quantbox.financing.DEFAULT_LEVERAGE`); a levered perps book declares
   `borrow`.
7. **No branch on the engine name outside an adapter.** The one table of
   names is `quantbox.engine.registry`. `tests/test_engine_seam.py` scans
   the source tree and fails on a comparison against an engine-name literal
   outside the adapter files; `tests/test_engine_parity.py` fails when an
   adapter flag other than `charges_funding` / `models_margin` is read
   outside the engine package.
8. **The execution lag is applied in ONE place:**
   `quantbox.engine._lag.lag_positions`, counted in execution-calendar bars,
   before any adapter sees a book. `quantbox.execution.apply_execution_lag`
   (for the L0 signal helpers) delegates to `lag_frame`. The ADR-0005/0006
   rules are unchanged: next-bar is mandatory, and same-bar needs the
   explicit override.
9. **The native object stays reachable:** `book.native` is the
   `vbt.Portfolio` on vectorbt and the long results frame on rsims.
   `backtest()` also returns it under its old key (`vbt_portfolio` or
   `rsims_results`).
10. **ADR-0001 is amended, not reversed.** Composing stays the rule and
    vectorbt is still re-exported at L0 (`quantbox.adapters.vectorbt`); the
    engine primitives keep their own API (`vectorbt_engine.run` still takes
    `rebalancing_freq` and `threshold`). Book simulation is the one capability
    that sits behind a seam, because it has two implementations. We do not
    build one combined backtester.
11. **Rebalancing policies and group limits are seam semantics (TOM-1450 3d-2).**
    `rebalancing_policy` declares one of four policies
    (`quantbox.engine.policy`), each an orders mask in the seam:
    `periodic` (today's `rebalancing_freq`), `tranche` (the targets are the
    mean of N staggered tranches, one refreshed per decision), `band` (today's
    `threshold`: the whole book trades when a held weight drifted past the
    band) and `corridor` (only the instruments outside their own
    `[target - below, target + above]` corridor trade; an exit always trades).
    Every policy takes a `frequency` (`weekly` / `monthly` / ... = the last
    execution bar of the period, or any `rebalancing_freq` form) and an
    optional market `calendar` (pandas-market-calendars): the execution bars
    are narrowed to that market's sessions, so decisions and the lag use
    sessions only. `rebalancing_freq` / `threshold` stay as the legacy
    spelling of `periodic` / `band` and write the same files; declaring both
    spellings is refused. `group_limits` (`quantbox.engine.groups`) keeps each
    group's gross weight inside `[min, max]` on every decided row, before the
    schedule; the groups come from universe metadata (`by:` a column of
    `load_universe()`), and an infeasible limit refuses the run. No adapter
    changed: `execute(...)` still receives targets and an orders mask.

12. **The same defaults on every engine; a cost is charged or refused (TOM-1500).**
    Tom, 2026-10-05: "same defaults for each engine". rsims compounds by default
    (`capitalise_profits: true`), as vectorbt does, and both engines start from
    the same `initial_cash` (10,000; vectorbt's own default was 100). Both charge
    every `Costs` field: the proportional fee, the fixed fee per order and
    slippage on the fill price (a buy fills at `close * (1 + slippage)`, the
    position is marked at the close). rsims follows vectorbt's order rules: a
    dust change (1e-9 relative) is no order, and a sell whose proceeds do not
    cover its fees is not placed. Each adapter names the costs it charges
    (`charged_costs()`); `simulate()` refuses a non-zero cost outside that set,
    naming the engine and the cost. A cost is never dropped silently.

### The threshold caveat

Drift depends on the held book, and the held book depends on past trades and
their costs. The seam tracks it cost-free: after a placed rebalance each
ordered cell holds its target and each untouched cell its drifted weight, and
every weight then drifts with its price against a cash remainder. An engine
that charges costs holds slightly less than that book, so vectorbt's old
in-engine threshold measured a slightly different drift. **A run with costs
can therefore place a rebalance on a slightly different bar than the
in-engine threshold did.** Without costs the two agree. Both engines now trade
on the bars the seam keeps (`tests/test_engine_parity.py` asserts identical
trade dates, and a known-answer drift case).

## Alternatives considered

### A. Keep one adapter (vectorbt) and call rsims directly where needed

**Rejected because:** that is the state that produced five doors with
different rules, and it leaves every client path without a backtest.

### B. One book builder for every door, the seam owning the schedule (chosen, 2026-10-05)

**Accepted because:**

- every door and every engine gets the same book from the same config: the
  calendars, the schedule, the NaN policy, the leverage default and the lag
  are decided once;
- rsims follows `rebalancing_freq` and `threshold`, so the engine choice is a
  choice of simulator, not of schedule;
- the adapters shrink to execution, which is what makes a third engine one
  class.

**Accepted cost: numbers move** for `backtest()`, `optimize()`, the sweep and
every rsims run with a non-daily schedule, a `threshold`, a NaN weight cell or
a net exposure above 1 without a declared `venue.leverage`. Pipeline goldens on
vectorbt do not move. Lab numbers that move are re-measured as findings in
TOM-1450 3e.

### C. The seam with two book builders and one lag (chosen 2026-10-04, superseded 2026-10-05)

**Was accepted because** golden and canonical numbers did not move.
**Superseded because** it kept a second builder and engine-specific schedule,
NaN and leverage semantics behind one interface: the interface was one, the
book was not.

## Consequences

### Intended

- `engine: rsims` works on every door, on the same schedule as vectorbt. A
  variants config on rsims charges the funding series it is handed, and the
  run manifest records `funding.modelled` from the primary variant's book.
- `quantbox sweep` takes `backtest.engine` and records `engine` in `sweep@1`.
- A new engine is one adapter class (`execute`, `stats`) plus one registry row.
- The policies of TOM-1450 3d (periodic on a market calendar, tranche, band,
  corridor) are orders masks in the seam; no adapter changed for them
  (decision 11). `tests/test_rebalancing_policies.py` holds a known answer per
  policy, the same orders and trades on both adapters, and a seeded
  randomized check that a group limit holds on every rebalance date.

### Unintended (and accepted)

- On rsims a sweep answers only the metric names it can compute from returns
  (`total_return`, `sharpe_ratio`, `sortino_ratio`, `annualized_return`,
  `annualized_volatility`, `max_drawdown`, `calmar_ratio`). Any other name is
  warned and left out.
- The threshold caveat above. It applies to `band` and `corridor` too.
- A corridor's held book mixes targets and drifted weights by design, so a
  group limit binds its TARGETS, not the held book between rebalances.
  `venue.leverage: normalize` scales a row above net 1 down proportionally:
  a group maximum still holds, a group minimum can fall below its bound by
  that scale.
- `venue.leverage: normalize` runs before the drift trigger (as it did for
  `threshold`), so it scales a bar as if every ordered cell trades. Under
  `band` that holds, because a placed bar trades every ordered cell. Under
  `corridor` only some cells trade, so the held net exposure of a levered book
  can end a bar above 1. A corridor book above net 1 should declare
  `venue.leverage: borrow` (review round 1 of #239; open for Tom).
- A tranche holds its target weights between refreshes (the book is the mean
  of the tranche targets), not a separately drifting sub-account.
- A `backtest()` weights frame stamped only on rebalance dates is carried onto
  the price bars from its first row (`_on_price_bars`), and a row stamped on a
  date with no price bar is decided on the next price bar.
- A non-zero weight on a ticker with no price column is refused on both
  schedules (TOM-1500; the calendar used to drop it with a `WEIGHTS:`
  warning, and the scheduled book before that dropped it silently).
- A multi-slice book carries no `data_validation` and no top-level `metrics`;
  each slice's metrics come from the adapter's `stats()`.
- The engine adapters refuse a parameter they do not own.

### Anti-patterns this rules out

- ❌ `if engine == "..."` anywhere outside `quantbox/engine/{vectorbt,rsims,registry}.py`.
- ❌ An adapter flag that decides WHAT is traded or WHEN (schedule, NaN, leverage).
- ❌ A door that shifts or schedules weights itself instead of through `simulate()`.
- ❌ An adapter that hides its native object.

## Notes

- Spec: TOM-1325 (quantbox 1.0); cards TOM-1447 (P3a, the seam) and TOM-1450
  (3d-1, this amendment; 3d adds the periodic/tranche/band/corridor policies on
  top). Parity suite: TOM-262, `tests/test_engine_parity.py` (its docstring
  states the common scope and the tolerance).
- Related: [ADR-0005](0005-next-bar-is-mandatory.md), [ADR-0006](0006-same-bar-explicit-override.md),
  [ADR-0007](0007-instrument-calendar-and-financing.md) (the scheduled book; amended here), and
  [ADR-0001](0001-library-not-framework.md) (amended).
