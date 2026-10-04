---
adr: 0008
title: Book simulation sits behind one engine seam, with vectorbt and rsims as adapters
status: proposed
date: 2026-10-04
supersedes:
superseded_by:
amends: "ADR-0001 § the adapter rule (a re-export, never an opaque wrapper): book simulation is the one capability that sits behind a seam; the native object stays reachable"
status_changes:
  - 2026-10-04: proposed with TOM-1447 (P3a of the quantbox 1.0 spec, TOM-1325)
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

## Decision

1. **One seam: `quantbox.engine`.** Decided weights + prices + execution
   timing + costs go in; a `TradedBook` comes out: `returns`, `value`,
   `weights`, `turnover`, `trades`, `metrics` and `native`.
2. **Two adapters behind it:** `VectorbtAdapter` and `RsimsAdapter`. Each one
   owns its parameters (unknown keys are refused), its NaN policy (`hold` or
   `flat`), its defaults (leverage, decision bars, funding, margin) and the
   normalisation of its output.
3. **Every door calls the seam.** The single run and variants call
   `simulate_book` (the scheduled book of ADR-0007). `backtest()`,
   `optimize()` and the sweep call `simulate_weights` (the bar-grid book).
   Each door takes `engine` and nothing else changes with it. Variants,
   `backtest()`, `optimize()` and the sweep now run on rsims as well.
4. **No branch on the engine name outside an adapter.** The one table of
   names is `quantbox.engine.registry`. `tests/test_engine_seam.py` scans
   the source tree and fails on a comparison against an engine-name literal
   outside the adapter files.
5. **The execution lag is applied in ONE place:**
   `quantbox.engine._lag.lag_positions`, before any adapter sees a book. The
   scheduled book counts it in execution-calendar bars, and the bar-grid book
   shifts by it. `quantbox.execution.apply_execution_lag` delegates to it. The
   ADR-0005/0006 rules are unchanged: next-bar is mandatory, and same-bar
   needs the explicit override. Deleting the line turns the timing tests red
   on every door and both engines.
6. **The native object stays reachable:** `book.native` is the
   `vbt.Portfolio` on vectorbt and the long results frame on rsims.
   `backtest()` also returns it under its old key (`vbt_portfolio` or
   `rsims_results`).
7. **ADR-0001 is amended, not reversed.** Composing stays the rule and
   vectorbt is still re-exported at L0 (`quantbox.adapters.vectorbt`). Book
   simulation is the one capability that sits behind a seam, because it has
   two implementations.

## Alternatives considered

### A. Keep one adapter (vectorbt) and call rsims directly where needed

**Rejected because:** that is the state that produced five doors with
different rules, and it leaves every client path without a backtest.

### B. One book builder for every door (route `backtest()` and the sweep through the scheduled book)

**Rejected for now because:** the scheduled book applies instrument
calendars, `venue.leverage: normalize`, deferral and the price/weight index
intersection. The helpers do none of these today, so their numbers would
move. The lag is already shared, and that is the timing guarantee. Merging
the builders is a separate, number-moving decision.

### C. The seam with two book builders and one lag (chosen)

**Accepted because:**

- every door gets the same timing, by construction and under a test;
- the engine choice is one value, and rsims reaches variants, the sweep and the helpers;
- golden and canonical numbers do not move: the adapters call the same primitives with the same arguments.

## Consequences

### Intended

- `engine: rsims` works on every door. A variants config on rsims charges the
  funding series it is handed, and the run manifest records
  `funding.modelled` from the primary variant's book.
- `quantbox sweep` takes `backtest.engine` and records `engine` in `sweep@1`.
- A new engine is one adapter class plus one registry row.

### Unintended (and accepted)

- On rsims a sweep answers only the metric names it can compute from returns
  (`total_return`, `sharpe_ratio`, `sortino_ratio`, `annualized_return`,
  `annualized_volatility`, `max_drawdown`, `calmar_ratio`). Any other name is
  warned and left out. Parity between the two engines is TOM-262's.
- rsims decides on every bar on every door: `rebalancing_freq` and
  `threshold` are vectorbt's schedule. This was already true of the pipeline.
- The engine adapters refuse a parameter they do not own. A sweep that
  passed a key to `vectorbt_engine.run` other than `use_numba`,
  `use_order_func` or `create_strategy_label` now fails loudly.

### Anti-patterns this rules out

- ❌ `if engine == "..."` anywhere outside `quantbox/engine/{vectorbt,rsims,registry}.py`.
- ❌ A door that shifts weights itself instead of through the seam.
- ❌ An adapter that hides its native object.

## Notes

- Spec: TOM-1325 (quantbox 1.0), card TOM-1447 (P3a). Parity suite: TOM-262.
- Related: [ADR-0005](0005-next-bar-is-mandatory.md), [ADR-0006](0006-same-bar-explicit-override.md),
  [ADR-0007](0007-instrument-calendar-and-financing.md) (the scheduled book), and
  [ADR-0001](0001-library-not-framework.md) (amended).
