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

**Rebalancing modes:**
- `rebalancing_freq: N` — periodic rebalancing every N days
- `threshold: 0.05` — rebalance when any weight drifts more than 5% from target

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

The build served is the one pinned for that name in `datasets.lock` at this repo's
root; re-pin with `quantbox-datasets pin <name> --lock <quantbox>/datasets.lock`, and
commit the lock. `quantbox sweep` takes the same name as `data.dataset`.

quantbox does **not** depend on `quantbox-datasets` (it is a private repo, and the
dependency would point the wrong way): install it from its clone, and set
`QUANTBOX_DATASETS_ROOT` to `<clone>/datasets` when it is not installed from one.
Reading a dataset without it raises an ImportError naming both.

### Parameters reference

| Parameter | Default | Description |
|---|---|---|
| `engine` | `vectorbt` | `"vectorbt"` or `"rsims"` |
| `fees` | `0.001` | Trading fee per side (0.001 = 10 bps) |
| `rebalancing_freq` | `1` | Rebalance every N days, or `"1W"`, `"1M"` |
| `threshold` | (none) | Drift threshold for rebalancing-bands mode |
| `trading_days` | `365` | Days per year for annualization |
| `universe.top_n` | — | Universe size (top N by volume/mcap) |
| `prices.lookback_days` | — | Price history window |
| `execution.lag_bars` | `1` | Bars between deciding a weight and filling it — see [Execution timing and venue constraints](#execution-timing-and-venue-constraints) |
| `venue.allow_shorts` | (unset) | Whether the venue can hold shorts — same section |
| `risk.max_leverage` | `99` | Gross cap per bar; only ever scales DOWN (both engines) |
| `risk.allow_short` | `false` | Legacy short switch (both engines); prefer `venue.allow_shorts` |
| `risk.tranches` | `1` | Rolling-mean tranching of target weights (both engines) |

### Execution timing and venue constraints

Both engines are **same-bar primitives**: the weight row they are handed for bar
`t` is filled at `close[t]`. Strategies decide `weights[t]` with data through
`close[t]`, so the pipeline — not the strategy, not the engine — owns the delay
between deciding and filling. It is applied in exactly one place
(`BacktestPipeline._align_for_engine`, after aggregation, venue clipping and
risk transforms, before the engine), so it holds for the vectorbt `from_orders`
branch, the vectorbt order-func (`threshold`) branch, rsims and the variants
flow alike. `quantbox sweep` (`analysis.parameter_grid`) uses the same setting.

```yaml
plugins:
  pipeline:
    name: backtest.pipeline.v1
    params:
      execution:
        lag_bars: 1          # int >= 0, default 1
      venue:
        allow_shorts: false  # bool, no default — absent means "not declared"
```

| Key | Type | Default | Meaning |
|---|---|---|---|
| `execution.lag_bars` | int ≥ 0 | `1` | Weights decided with data through bar `t` fill at the **close of bar `t + lag_bars`**. `0` = same-bar: the signal is filled at the very close it was computed from, which no order could have achieved. Allowed only when written explicitly; the run then logs an `EXECUTION TIMING … SAME-BAR` warning, `quantbox validate` reports a warning, and the manifest records `same_bar: true`. |
| `venue.allow_shorts` | bool | — | `false`: negative **target** weights are clipped to `0` *before* tranching and the leverage cap. **The long side is not re-normalised** — the book carries less gross; it is never re-levered to refill it. `true`: shorts pass through. Must not contradict an explicit `risk.allow_short` (the run refuses). |

Unknown keys, non-integers, booleans and negative lags are **refused**
(`ConfigValidationError` from the runner, `ValueError` from the pipeline, before
any data is loaded) — a typo never falls back to a default. `execution` and
`venue` are run-level: a variant that tries to override either is refused.

**Strategies must not lag their own output.** A strategy emits the weights it
wants given data through `t`; internal lags inside a *signal* or an *estimator*
(e.g. `weights.shift(1) * returns` to estimate realised vol causally) are fine
and unaffected. A strategy that already shifts the weights it returns would be
lagged twice — remove that shift rather than setting `lag_bars: 0`.

**Where it is recorded.** `run_manifest.json` carries
`execution: {lag_bars, fill: "close", same_bar, description}` and
`venue: {declared, allow_shorts}`; `metrics.json` carries `execution_lag_bars`;
`summary.md` has an **Execution timing** line, the HTML report states it in the
masthead and the reproducibility appendix, and the CLI prints `EXECUTION: …`
under `METRICS:`. Sweep grids carry a `lag_bars` column.

**NaN weight rows — one saved book per engine, and the engines disagree.** A
NaN weight cell mid-series means "the strategy said nothing for this bar". The
engines have always answered that differently: **vectorbt forward-fills** (holds
the last target; leading NaN → 0) while **rsims treats NaN as 0** (goes flat).
This pipeline does not change either engine's numbers; it materialises the
policy the chosen engine already applies into the frame it hands over
(`quantbox.execution.materialise_nan_policy`), so `traded_weights` and the
`traded_*` metrics describe the book that engine actually traded — and never
contain NaN. The disagreement itself is a **known issue**: the same config with
mid-series NaN weights gives different books on the two engines. Emit explicit
weights for every bar to avoid depending on it. (The sweep path hands vectorbt
the raw frame, so it holds through NaN rows like any vectorbt run; it saves no
weights.)

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
> number produced by `quantbox run -c` was same-bar.** To reproduce an old
> number, set `execution.lag_bars: 0`. What that reproduces, precisely: the
> **historical metric keys** (`total_return`, `cagr`, `sharpe`, … — the 12 keys
> `metrics.json` carried before) and the returns / equity series, pinned by the
> frozen goldens in `cookbook/canonical/expected_same_bar/`. It does **not** make
> the run directory byte-identical to an old one: `metrics.json` gains the
> `execution_lag_bars` / `traded_*` / `target_*` keys, a `traded_weights`
> artifact appears, and the report is built from the traded weights. `sweep`'s
> `shift_signal` (Python kwarg and `backtest.shift_signal` in sweep YAML) still
> works as a deprecated alias of `execution.lag_bars`; sweep numbers are unchanged.
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
  target book handed to the engine; with `rebalancing_freq` ≠ 1 or a `threshold`
  the engine trades a subset of those bars and positions drift in between.
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
