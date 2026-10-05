# Backtesting

QuantBox includes two backtesting engines accessible through the `backtest.pipeline.v1` pipeline plugin. Both use the same strategy configs as live trading — swap the pipeline name to go from backtest to production.

## Quick start

```bash
quantbox run -c cookbook/configs/run_backtest_crypto_trend.yaml
```

## Engines

### vectorbt (spot / equity)

Numba-accelerated portfolio simulation. Best for spot strategies without leverage.

```yaml
plugins:
  pipeline:
    name: "backtest.pipeline.v1"
    params:
      engine: vectorbt
      fees: 0.001               # 10 bps per trade
      rebalancing_freq: 1       # every N days (or "1W", "1M")
      # threshold: 0.05         # uncomment for rebalancing-bands mode
      trading_days: 365
```

**Rebalancing modes** (the seam's schedule — both engines follow it, docs/adr/0008):
- `rebalancing_freq: N` — periodic rebalancing every N execution bars (or `"W-FRI"`, `"ME"`, ...)
- `threshold: 0.05` — a scheduled rebalance is placed only when a held weight drifted more than
  5% from its target. The seam measures the drift cost-free; with costs a rebalance near the band
  edge can fall on a slightly different bar than an in-engine band would.

**Rebalancing policies** (`rebalancing_policy`, TOM-1450; replaces the two keys above — declare
one spelling, not both). Every policy is an orders mask in the seam, the same on both engines
(`quantbox.engine.policy` has the full rules). A policy is a **cadence** (`periodic` or `tranche`)
times a **trigger** (`none`, `band` or `corridor`; TOM-1513):

```yaml
      rebalancing_policy: {cadence: tranche, tranches: 5, frequency: daily, trigger: corridor, width: 0.02}
      # rebalancing_policy: {cadence: periodic, frequency: monthly, calendar: NYSE}
      # the single-key spelling still works and means the same thing:
      # rebalancing_policy: {policy: periodic, frequency: monthly, calendar: NYSE}
      # rebalancing_policy: {policy: tranche, tranches: 4, frequency: weekly}
      # rebalancing_policy: {policy: band, band: 0.05}
      # rebalancing_policy: {policy: corridor, width: [0.02, 0.05], bounds: {SPY: [0.01, 0.03]}}
```

- `frequency` — the bars a rebalance is considered on: `daily`, `weekly`, `monthly`, `quarterly`,
  `yearly` (the last execution bar of the period), or any `rebalancing_freq` form.
- `calendar` — a pandas-market-calendars name. Decisions and the execution lag use that
  market's sessions only: March 2024 ends on Thursday the 28th (Good Friday), not Sunday the 31st.
- `tranche` — the book is the mean of N tranches; one tranche is refreshed on each decision.
- `band` — the whole book trades when a held weight drifted past the band (= `threshold`).
- `corridor` — when one instrument is outside its own `[target - below, target + above]`
  corridor, the whole book trades back to target. An exit to 0 is always a hit.
- `min_trade` (every policy, default 0 = off) — on a rebalance, a trade smaller than
  `min_trade` (absolute weight) is dropped; sells always execute; buys above the cash plus
  the sell proceeds are scaled down proportionally, so without `venue.leverage: borrow` the
  held net never goes above 1. A trade under `min_trade` cannot trigger a band or corridor.

The same `rebalancing_policy` block works on `trade.full_pipeline.v1` (TOM-1518): each live
run asks the same code whether a rebalance is due on its last decided bar and to which
targets, given the broker's held book (`quantbox.engine.policy.decide_rebalance`; what it
decided is in `notes["rebalancing_policy"]`). Live refuses an int `frequency` above 1 and
null (the live history window moves), and a declared policy's `min_trade` replaces
`min_trade_size`.

**Group limits** (`group_limits`) keep each group's gross weight inside `[min, max]` on every
decided row, before execution. The groups come from a universe metadata column (a
`local_file_data` universe file keeps its per-symbol columns, such as `asset_class`):

```yaml
      group_limits:
        by: asset_class
        limits: {equity: {max: 0.6}, bond: {min: 0.2, max: 0.5}}
        excess: redistribute     # or cash: weight cut from a capped group stays in cash
```

An infeasible limit refuses the run with the first dates (the minimums need more weight than
the row holds, or a group with a minimum holds nothing). `data_validation.json` records the
policy (`rebalancing`) and the limits (`groups`); `quantbox config explain` shows both.

### rsims (futures)

Daily step simulator with funding rates, margin, leverage, and no-trade buffer. Best for futures strategies.

```yaml
plugins:
  pipeline:
    name: "backtest.pipeline.v1"
    params:
      engine: rsims
      fees: 0.001
      rebalancing_freq: 1
      trading_days: 365

      risk:
        tranches: 1
        max_leverage: 2
        allow_short: true
```

rsims has the same defaults as vectorbt (TOM-1500): it compounds, starts from the
same `initial_cash` and charges `fees`, `slippage` and `fixed_fees` the same way. A
cost an engine cannot model is refused, never dropped.

**Additional rsims features:**
- Funding rate simulation (long/short asymmetry)
- Margin and leverage tracking
- No-trade buffer to reduce turnover

## Configuration

### Full example

```yaml
run:
  mode: backtest
  asof: "2026-02-06"
  pipeline: "backtest.pipeline.v1"

artifacts:
  root: "./artifacts"

plugins:
  pipeline:
    name: "backtest.pipeline.v1"
    params:
      engine: vectorbt
      fees: 0.001
      rebalancing_freq: 1
      trading_days: 365

      universe:
        top_n: 100

      prices:
        lookback_days: 365

  strategies:
    - name: "strategy.crypto_trend.v1"
      weight: 0.6
      params:
        lookback_days: 365

    - name: "strategy.carver_trend.v1"
      weight: 0.4
      params:
        lookback_days: 365
        target_vol: 0.50

  aggregator:
    name: "strategy.weighted_avg.v1"
    params: {}

  data:
    name: "binance.live_data.v1"
    params_init:
      quote_asset: USDT
```

### Reading a curated dataset

Datasets from `quantbox-datasets` are read **by name**, never by a sibling path:

```yaml
  data:
    name: "local_file_data"
    params_init:
      dataset: etf-daily
```

The build served is the one pinned for that name in the `datasets.lock` nearest the
config (this repo's root here); re-pin with
`quantbox-datasets pin <name> --lock <quantbox>/datasets.lock`, and commit the lock.
The bytes are read under `$QUANTBOX_DATASETS_ROOT`, never relative to the working
directory, so the same config run from the repo root or a worktree reads the same
build. `quantbox sweep` takes the same name as `data.dataset`.

`quantbox dataset resolve <name> -c <config> --json` prints what `run -c <config>`
will read — the same lock, so the same build — `path`, the pinned `sha256`, the `actual_sha256` of those bytes, `matches`, `market` and the
`funding_rates` file if any — and `run` records the same object under
`run_manifest.json` → `dataset.resolved`. When the bytes are not the pinned build and
quantbox-datasets cannot restore it from git history, both commands fail and name both
shas. The older inline style (`dataset_root` + `expected_prices_sha256`, as
`dataset.curated.v1` takes them) still runs but warns: it is the deprecated alias.

quantbox does **not** depend on `quantbox-datasets` (it is a private repo, and the
dependency would point the wrong way): install it from its clone, and set
`QUANTBOX_DATASETS_ROOT` to `<clone>/datasets` when it is not installed from one.
Reading a dataset without it raises an ImportError naming both.

### Parameters reference

| Parameter | Default | Description |
|---|---|---|
| `engine` | `vectorbt` | `"vectorbt"` or `"rsims"` |
| `fees` | `0.001` | Trading fee per side (0.001 = 10 bps), every engine |
| `slippage` | `0` | Proportional slippage on the fill price (0.0005 = 5 bps), every engine |
| `fixed_fees` | `0` | Fixed fee per order, in quote currency, every engine |
| `initial_cash` | `10000` | Starting cash, every engine (a fixed fee is a share of it) |
| `capitalise_profits` | `true` | rsims: size off current equity (compound), as vectorbt does; `false` sizes off `min(initial_cash, equity)` |
| `rebalancing_freq` | `1` | The DECISION schedule on the execution calendar: every N execution bars, or `"1W"`, `"ME"`, `"BMS"`; period-end offsets decide on the period's last execution bar, others on the next one; the trade follows `lag_bars` execution bars later ([ADR-0007](../adr/0007-instrument-calendar-and-financing.md)) |
| `threshold` | (none) | Drift band: a scheduled rebalance is placed only when a held weight drifted more than this (seam-computed, cost-free, every engine) |
| `trading_days` | `365` | Days per year for annualization |
| `universe.top_n` | — | Universe size (top N by volume/mcap) |
| `prices.lookback_days` | — | Price history window |
| `execution.lag_bars` | `1` | Bars between deciding a weight and filling it — see [Execution timing and venue constraints](#execution-timing-and-venue-constraints) |
| `venue.allow_shorts` | (unset) | Whether the venue can hold shorts — same section |
| `venue.financing` | (unset) | What borrowed / idle cash costs — [Missing prices and financing](#missing-prices-and-financing) |
| `venue.leverage` | `normalize` (every engine) | How the decision is normalised: a target row above net 1 is scaled to 1, or borrowed — [Missing prices and financing](#missing-prices-and-financing) |
| `execution.calendar` | `majority` | The execution calendar: `majority` \| `union` \| `intersection` \| a ticker — [Missing prices and financing](#missing-prices-and-financing) |
| `execution.schedule` | `calendar` | `calendar`: the scheduled book; `bars`: every price bar executes, no deferral, no `venue.leverage` ([ADR-0008](../adr/0008-engine-seam.md)) |
| `risk.max_leverage` | `1` | Gross cap per bar (`sum \|w\|`); only ever scales DOWN (both engines). The same default in trading, `backtest()`, `optimize()` and the sweep (`quantbox.decision.DEFAULT_MAX_LEVERAGE`, TOM-1525): a levered book declares it |
| `risk.allow_short` | `false` | Legacy short switch (both engines); prefer `venue.allow_shorts` |
| `risk.tranches` | `1` | DEPRECATED (TOM-1513): the tranche cadence, `rebalancing_policy: {cadence: tranche, tranches: N}`; warns |

### Execution timing and venue constraints

Both engines are **same-bar primitives**: the weight row they are handed for bar
`t` is filled at `close[t]`. Strategies decide `weights[t]` with data through
`close[t]`, so the pipeline — not the strategy, not the engine — owns the delay
between deciding and filling. It is applied in exactly one place, inside
the engine seam (`quantbox.engine._lag.lag_positions`, docs/adr/0008: after
aggregation, venue clipping and risk transforms, before any engine adapter),
so it holds for both engines, the variants flow, the sweep, `backtest()` and
`optimize()` alike — they all build the book with the one function
`quantbox.engine.simulate`. `quantbox sweep` (`analysis.parameter_grid`) uses the same setting,
and so do the Python helpers `backtest()` and `optimize()`
(`quantbox.plugins.backtesting`): keyword `lag_bars=`, same default, same
refusal of `0`, and the result carries the same `execution` record. The L1
signal helpers `quantbox.bt.run` and
`quantbox.adapters.vectorbt.from_signals_with_costs` lag their signals the
same way (`lag_bars=`, default 1, `0` refused).

```yaml
plugins:
  pipeline:
    name: backtest.pipeline.v1
    params:
      execution:
        lag_bars: 1          # int >= 1, default 1
      venue:
        allow_shorts: false  # bool, no default — absent means "not declared"
```

| Key | Type | Default | Meaning |
|---|---|---|---|
| `execution.lag_bars` | int ≥ 1 | `1` | Weights decided with data through bar `t` fill at the **close of bar `t + lag_bars`**. `0` (same-bar — the signal filled at the very close it was computed from, which no order could have achieved) is **refused** by every entry point and is an error in `quantbox validate` ([ADR-0005](../adr/0005-next-bar-is-mandatory.md)), unless `execution.same_bar` grants it (below). |
| `execution.same_bar` | `{allow: true, reason: str}` | — | The explicit same-bar override ([ADR-0006](../adr/0006-same-bar-explicit-override.md)): valid only next to `lag_bars: 0`, `reason` non-empty. The run is then **research, not a backtest** — see [Same-bar research runs](#same-bar-research-runs-the-explicit-override). |
| `venue.allow_shorts` | bool | — | `false`: negative **target** weights are clipped to `0` *before* the leverage cap, the group limits, the normalisation and tranching (the decision's first step, `quantbox.decision`). **The long side is not re-levered** — the book carries less gross. `true`: shorts pass through. Must not contradict an explicit `risk.allow_short` (the run refuses). |

Unknown keys, non-integers, booleans, `0` (without the override) and negative lags are **refused**
(`ConfigValidationError` from the runner, `ValueError` from the pipeline, before
any data is loaded) — a typo never falls back to a default. `execution` and
`venue` are run-level: a variant that tries to override either is refused.

**Strategies must not lag their own output.** A strategy emits the weights it
wants given data through `t`; internal lags inside a *signal* or an *estimator*
(e.g. `weights.shift(1) * returns` to estimate realised vol causally) are fine
and unaffected. A strategy that already shifts the weights it returns would be
lagged twice — remove that shift; `lag_bars: 0` is not a way out (and the
same-bar override is not one either: it is for data, not for strategies).

**Where it is recorded.** `run_manifest.json` carries
`execution: {lag_bars, fill: "close", same_bar, description}` (`same_bar` is
`true`, with `same_bar_reason`, only under the override), `run: {kind}`
(`backtest` or `research`) and `venue: {declared, allow_shorts, max_leverage, leverage, financing}`; `metrics.json` carries `execution_lag_bars`;
`summary.md` has an **Execution timing** line, the HTML report states it in the
masthead and the reproducibility appendix, and the CLI prints `EXECUTION: …`
under `METRICS:`. Sweep grids carry a `lag_bars` column.

#### Missing prices and financing

[ADR-0007](../adr/0007-instrument-calendar-and-financing.md) (TOM-1429). Each instrument gets a
**life window**, from its first valid price to its last:

- **Inside** the window, a bar with no price (a holiday, a gap) is forward-filled to MARK the
  position, which is held. No order fills at a price the instrument did not print: an order
  falling on such a bar is **deferred** to the instrument's own next printed bar, and counted.
- **Outside** it (before listing, after delisting), the target is forced to 0, and that
  override is counted and logged. A feed that stops early looks exactly like a delisting.

On top sits ONE **execution calendar**, the bars decisions are taken and orders placed on:

```yaml
      execution:
        lag_bars: 1
        calendar: majority   # majority (default) | union | intersection | "<ticker in the prices>"
```

`majority`: half of the live instruments print; `union`: any; `intersection`: all; a ticker:
that series prints. **Decision vs execution:** `rebalancing_freq` picks DECISION bars on that
calendar (`"ME"` = the last execution bar of the month), and the trade happens `lag_bars`
**execution** bars later. `rebalance_schedule.parquet` records `decision_date`,
`execution_date` and `deferred_instruments` for every rebalance. Before ADR-0007 the schedule
named the TRADE bar: a config that meant "decide at month-end, trade on the 1st" with `BMS`
now says `ME`.

```yaml
      venue:
        allow_shorts: true
        leverage: borrow           # normalize (the default, every engine) | borrow
        financing:
          rate: "LT12TRUU Index"   # ticker in the prices (cash TR index) | annual number (0.0 = free)
          borrow_spread_bps: 0     # borrowed cash: rate + spread
          lend_spread_bps: 0       # idle cash:     rate - spread
```

`venue.leverage` is part of the DECISION (TOM-1520, `quantbox.decision`): the target weights
are final before any rebalancing policy reads them, and live trading computes the same ones.
`leverage: normalize` scales every decided row whose net exposure is above 1 down to net 1
(after the short clip, the `risk.max_leverage` gross cap and the group limits; counted in
`data_validation.json` `decision`, warned). Execution then never borrows: on every placed
rebalance the buys are capped at the cash plus the sell proceeds, which binds only when a
deferred instrument still holds its old weight (`leverage.cash_capped_rebalances`, warned).
`leverage: borrow` holds it; with `financing` the residual `1 - sum(w)` is held as two
synthetic cash legs (idle cash earns `rate - lend_spread`, borrowed cash pays `rate +
borrow_spread`, both trade without fees). `borrow` without `financing` runs at an ASSUMED rate
of 0, recorded as `venue.financing.assumed: true` and warned about.

Everything is counted in `data_validation.json` (`quantbox/data-validation@1`, schema in
`artifact_schemas/`): `calendar` (per instrument: forward-filled bars, deferred trades,
targets outside the window, stale decisions; `legacy_coverage_drop` = columns the old
50%-coverage rule would have dropped), `execution_calendar` (execution vs total bars,
non-execution bars per year), `timing`, `staleness` (age of the inputs at each decision:
count, max, p95 — visible, not blocking), `weight_age` (on a calendar schedule — one period-end
offset such as `ME`, `BME`, `QE`, `W-FRI` — the decisions that MISSED their period's weights
because the strategy stamped them on a non-execution bar after the decision bar: stamp
weights on the decision bars, the EXECUTION calendar, never on calendar period-ends from a
wider panel; a `TIMING:` warning names them. Any other schedule reports `measured: false`
— the check does not cover it), `index_alignment` (what the price/weight index intersection
dropped; an `INDEX:` warning names price bars with no weight row), `leverage` (measured on the
held book, and the cash cap) and `decision` (the rules, and the rows the decision clipped,
capped and normalised). Summaries go to
`run_manifest.json` `data_validation` and `metrics.json`.

Every vectorbt run records `engine_underfilled_rebalances` and `engine_max_fill_gap` (on
the bars the seam ordered). A `threshold` run also records `threshold_skipped_rebalances`
and a `threshold` section in `data_validation.json`. They compare the book the engine held after each
rebalance with `traded_weights`.

#### Same-bar research runs: the explicit override

Next-bar is the rule (Tom, 2026-10-02: "always lag +1, never the same bar").
Same-bar is **an explicit allowance against best practice**, for data where it
is closer to reality than next-bar. Example: monthly-only data, where the
month-end close is the only price and the next month-end is further from any
real fill than the same one. It is asked for in the config, with a reason:

```yaml
execution:
  lag_bars: 0
  same_bar: {allow: true, reason: "monthly-only data: the month-end close is the only price"}
```

In Python: `backtest(prices, weights, lag_bars=0, allow_same_bar=True,
same_bar_reason="...")`, and the same keywords on `optimize()`.

What the run then is ([ADR-0006](../adr/0006-same-bar-explicit-override.md)):
**RESEARCH, not a backtest.** In Tom's words, *"to bardziej nie backtest, tylko
ogólny research"*.

- `run_manifest.json`: `execution.same_bar: true`, `execution.same_bar_reason`,
  `run: {kind: research}`; `config explain` plans the same. A `backtest()` /
  `optimize()` result carries `run: {kind: research}` too.
- `finding_report.json`: no hero card reports `backtest_sharpe`, cards are
  toned bad, the chart title says RESEARCH, and a failed "Execution timing"
  audit axis carries the reason.
- `quantbox gates … --returns <run_dir>/returns.parquet`: the verdict carries
  `run_kind: research` and the reason (a RESEARCH line in text mode).
- `run.strict` / `promotion` mode refuse it. `quantbox sweep` refuses the
  `same_bar` block, and `bt.run` / `from_signals_with_costs` refuse `0`.

Refused: `lag_bars: 0` alone, `allow: false`, an empty or missing `reason`,
and an override next to `lag_bars >= 1`.

**NaN weight rows — one policy, every engine.** A NaN weight cell mid-series
means "the strategy said nothing for this bar". The seam answers it once
(`quantbox.engine.materialise_nan`, docs/adr/0008): the cell HOLDS the last
decided target; a leading NaN is 0. Every engine receives the materialised
book, so `traded_weights` never contains NaN and the same config gives the same
book on both engines. (Until TOM-1450 rsims treated NaN as 0 — went flat — so an
rsims run with mid-series NaN weights moves.)

**Shorts are never silent.** Whatever the config says, every run measures
`target_short_gross_share` (the strategy's targets) and
`traded_short_gross_share` (the book the engine received) and logs a `VENUE —`
warning when (a) targets contain shorts that are being clipped — by
`venue.allow_shorts: false` or by the legacy `risk.allow_short: false` default —
or (b) shorts are traded and no `venue` block is declared. The pipeline cannot
tell a spot dataset from a perp one: the catalog's `market:` field does not
reach the `DatasetManifest`, so the venue has to be declared in the config.

> **MIGRATION — default changed from same-bar to next-bar.** Until this
> release `quantbox run -c` handed strategy weights to the engine unshifted,
> while `quantbox sweep` shifted them by one bar. **Every historical backtest
> number produced by `quantbox run -c` was same-bar.** v0.8.0 still let
> `execution.lag_bars: 0` reproduce such a number, with a warning; since
> [ADR-0005](../adr/0005-next-bar-is-mandatory.md) `0` is refused everywhere
> and an old same-bar number is reproduced only by pinning a quantbox release
> older than that — and is then a record of a look-ahead, not a result. The
> same-bar override ([ADR-0006](../adr/0006-same-bar-explicit-override.md)) is
> not a way to reproduce them either: it is for data where same-bar is closer
> to reality, and its runs are labelled research. `sweep`'s
> `shift_signal` (Python kwarg and `backtest.shift_signal` in sweep YAML) still
> works as a deprecated alias of `execution.lag_bars`; sweep numbers are unchanged.
> The L1 signal helpers `quantbox.bt.run` and
> `adapters.vectorbt.from_signals_with_costs` were same-bar too, with no lag
> setting at all; since ADR-0005 they fill next-bar, so **the same inputs return
> different numbers** — one bar later, which is the correct number.
>
> **Do not quote an old-vs-new delta as "the size of the look-ahead".** The lag
> sets the first `lag_bars` rows flat, and with an integer `rebalancing_freq`
> N > 1 row 0 *is* the first rebalance bar — so the book enters at bar N, not
> bar 1, and sits flat for N bars. On a short window that lost first period is
> mixed into the delta: the regenerated `momentum` canonical golden (30 bars,
> `rebalancing_freq: 5`, total return +6.85% → −13.88%) contains both effects.
> On the reviewer's toy the split was: same-bar −0.1792, next-bar −0.1589,
> same-bar with only the first rebalance zeroed −0.1553 — there the lost first
> period ALONE moves the number by more than the whole same-bar → next-bar delta.
> Buy-and-hold (`rebalancing_freq: null`) is the exception: its one decision is
> the first bar, and it fills `lag_bars` execution bars later.

### Arms: one base config, many runs

Near-identical configs that differ in one or two values are **arms** of one
batch, declared once: a base config plus named `overrides:` (dotted paths, list
indices allowed: `plugins.strategies.0.params.min_periods`) or a Cartesian
`grid:`. `quantbox arms -c arms.yaml` runs them in parallel within
`--max-workers` and a memory budget (`parallel.memory_budget_gb` /
`arm_memory_gb`; with no budget, the memory available now), one ordinary run —
and one `run@1` manifest — per arm. `arms_summary.json` (`quantbox/arms@1`)
lists every arm with a link to its manifest, `n_trials` (the number of arms, or
a larger honest count from the file) is stamped into every manifest, and a
failing arm fails the batch (exit 1) by name while the others' results stay.
The file format is the `quantbox.arms` module docstring.

Timing is batch-level: the arms file's `execution:` block is the same block
`quantbox sweep` reads, and an arm that overrides `execution`, `run.n_trials` or
`artifacts` is refused. `quantbox sweep` records its timing and `n_trials`
(one per grid row) in `<output_dir>/sweep_manifest.json`.

A `source: path/to/strategy.py:Class` strategy works in a single run, as an
arm, in a variant (`variants[].strategy.source`) and in a sweep
(`strategy: {source: ...}`, path relative to the sweep config). In the runner
the path is relative to the working directory.

## Outputs

Artifacts are written to `artifacts/<run_id>/`:

| Artifact | Description |
|---|---|
| `strategy_weights` | Per-strategy weight time series |
| `aggregated_weights` | Final blended weights after aggregation |
| `weights_history` | The strategy's **decided target** weights: aggregated, BEFORE venue clipping, risk transforms and the execution lag |
| `traded_weights` | The weights the engine actually received: after venue clipping, tranching, leverage cap, execution lag and missing-price masking. Report charts and attribution are built from these |
| `portfolio_daily` | Daily portfolio value series |
| `returns` | Daily return series |
| `metrics` | Summary statistics (Sharpe, drawdown, etc.) |

## Metrics

The `metrics` artifact includes:

- **Sharpe ratio** — annualized risk-adjusted return
- **Max drawdown** — peak-to-trough decline
- **CAGR** — compound annual growth rate
- **Volatility** — annualized return standard deviation
- **Calmar ratio** — CAGR / max drawdown
- **Rolling Sharpe** — time-varying Sharpe windows
- **`execution_lag_bars`** — the execution timing the numbers were produced under
- **Traded-book statistics**, computed from `traded_weights` (never from
  `weights_history`): `traded_mean_gross_exposure`, `traded_mean_net_exposure`,
  `traded_short_gross_share` (short gross / total gross over the run),
  `traded_mean_turnover` (mean per-bar `sum(|w[t] - w[t-1]|)`),
  `traded_flat_bar_share` (share of bars with zero gross). They describe the
  book held after the seam's orders; with `rebalancing_freq` ≠ 1 or a `threshold`
  the engine trades a subset of the bars and positions drift in between.
- **`target_short_gross_share`, `target_mean_net_exposure`** — the same
  statistics of the strategy's targets, so a clipped short book is visible.

## Research to production

The same strategy params work in both backtesting and live trading. To go live:

1. **Backtest:**
   ```yaml
   pipeline:
     name: "backtest.pipeline.v1"
   ```

2. **Live (swap pipeline, add broker + risk + rebalancer):**
   ```yaml
   pipeline:
     name: "trade.full_pipeline.v1"
   broker:
     name: "hyperliquid.perps.v1"
   rebalancing:
     name: "rebalancing.futures.v1"
   risk:
     - name: "risk.trading_basic.v1"
   ```

Strategy and data sections stay the same.
