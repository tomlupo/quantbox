# Report blocks — reusable units of the research report

The HTML research report (`backtest.pipeline.v1`) is composed of *blocks*:
named, self-describing chart units that render into a fixed editorial
template. A block has metadata (title, description, payload contract, tags)
and a builder function that produces a Plotly figure dict.

Strategies opt in to specific blocks by emitting matching payloads; generic
blocks fire automatically for every backtest from pipeline outputs alone.

## Why blocks

- **Composable** — strategies pick which diagnostic blocks they want. Add a
  block once, every strategy gets it for free.
- **Reusable** — the same chart type works across strategies (e.g. a
  drawdown-timeline block doesn't care whether the underlying strategy is
  crypto trend, equity carry, or stat-arb).
- **Versionable** — block payload schemas are documented; future block
  versions can coexist.
- **Domain-agnostic** — generic blocks compute from standard pipeline outputs
  only (`returns`, `portfolio_daily`, `weights_history`, `bt_prices`). No
  strategy code is required.

## Block kinds

| Section | Purpose | Trigger |
|---|---|---|
| `framework` | Generic — works for any strategy | Auto-render when required pipeline outputs are present |
| `diagnostics` | Strategy-specific deep-dive charts | Strategy emits `details["diagnostics"][block_name] = {...payload}` |
| `comparison` | Cross-variant overlays | Auto-render when ≥2 variants in the run |
| `appendix` | Reproducibility, params, repro JSON | Always |

## The current catalog

### Generic blocks (framework section, auto-on)

| Name | Title | Required inputs |
|---|---|---|
| `return_distribution` | Daily return histogram + Normal overlay + stat badge | `returns` |
| `rolling_metrics` | Rolling annualised Sharpe + rolling vol (two-panel) | `returns` |
| `turnover_timeline` | Daily portfolio turnover + cumulative (dual-axis) | `weights_history` |

### Diagnostic blocks (strategy opt-in)

| Name | Title | Tags | Payload schema |
|---|---|---|---|
| `regime_overlay` | Regime & trend overlay | trend, regime | `{ref_ticker, fast_window, slow_window}` |
| `signal_count` | Active signals over time | trend, signal | `{series, cap?, label?}` |
| `donchian_overlay` | Donchian channel + trailing stop | trend, donchian, crypto | `{ref_ticker, window, price, high, low, mid, trailing_stop, signal}` |
| `per_window_signal` | Per-window Donchian signal density | trend, ensemble | `{ref_ticker, signals: {window: Series}}` |
| `vol_overview` | Realized volatility + scalers | vol-target | `{realized_vol, scalers, vol_lookback}` |

### Framework charts (always emitted by the pipeline)

These are hard-wired in `_report.py` rather than living in the block
registry. Migrating them is future work.

- `portfolio` — equity curve + drawdown + rolling Sharpe (manual 3-panel)
- `monthly` — monthly returns heatmap with YTD column
- `contrib` — per-ticker contribution bars
- `weights` — top-20 weight heatmap (weekly average)
- `position_stack` — stacked-membership area chart
- `universe_size` — daily count of tickers with non-NaN price

### Cross-variant blocks

- `equity_overlay` — multi-variant equity curves on a log y-axis
- `per_variant_table` — per-cell coolwarm gradient table of headline metrics
- `per_variant_stats_heatmap` — row-normalised heatmap (one row per metric)
- `gross_exposure_overlay` — gross exposure by variant
- `btc_weight_overlay` — single-symbol weight time series across variants

## Adding a new diagnostic block

A strategy block lives in three places: a **builder function**, a **registry
entry**, and a **strategy emit**.

### 1. Write the builder

```python
# my_module/_charts.py
import plotly.graph_objects as go


def build_factor_loadings_chart(payload, **ctx):
    """Heatmap of strategy weight × factor exposures.

    Payload:
      {
          "factors": list[str],
          "tickers": list[str],
          "loadings": pd.DataFrame  # rows=tickers, cols=factors
      }
    """
    loadings = payload.get("loadings")
    if loadings is None or loadings.empty:
        return None
    fig = go.Figure(
        data=go.Heatmap(
            z=loadings.values,
            x=loadings.columns.tolist(),
            y=loadings.index.tolist(),
            colorscale="RdBu",
            zmid=0,
        )
    )
    fig.update_layout(title="<b>Factor loadings</b>", height=400)
    return fig.to_dict()
```

### 2. Register the block

```python
from quantbox.plugins.pipeline.blocks import register_block, ReportBlock

register_block(ReportBlock(
    name="factor_loadings",
    title="Factor loadings",
    description="Heatmap of strategy weights × factor exposures.",
    builder=build_factor_loadings_chart,
    section="diagnostics",
    payload_schema='{"factors": list[str], "tickers": list[str], "loadings": DataFrame}',
    tags=("factor", "exposure"),
))
```

For one-off lab work, do this at import time in your research module.
For shipped strategies, register inside your strategy's plugin entry-point.

### 3. Emit the payload from your strategy

```python
def run(self, data, params=None):
    # ... strategy logic ...
    return {
        "weights": weights,
        "details": {
            "diagnostics": {
                "factor_loadings": {
                    "factors": ["market", "value", "momentum"],
                    "tickers": list(weights.columns),
                    "loadings": loadings_df,
                },
            },
        },
    }
```

Next backtest run, the block renders automatically. Blocks with no payload
auto-hide.

## Adding a new generic block

Generic blocks don't take a payload — they derive everything from the
shared `ctx`:

```python
ctx = {
    "returns": pd.Series,        # strategy daily returns
    "portfolio_daily": pd.DataFrame,  # value column + index
    "weights_history": pd.DataFrame,  # per-day per-ticker weights
    "bt_prices": pd.DataFrame,        # backtest universe prices
    "variant_results": dict,          # only in multi-variant mode
}
```

Declare which keys you need in `requires`; the dispatcher skips the block
if any required input is missing or empty:

```python
register_block(ReportBlock(
    name="drawdown_timeline",
    title="Drawdown timeline",
    description="Underwater chart with peak-to-trough annotations.",
    builder=build_drawdown_timeline,
    section="framework",
    payload_schema="generic — derives from ctx['portfolio_daily']",
    tags=("generic",),
    requires=("portfolio_daily",),
))
```

## Where strategy-specific block libraries live

Engine-side (quantbox) ships only **truly generic** blocks. Domain-specific
blocks (Donchian channels for trend-following, factor decompositions for
stat-arb, etc.) belong in the **consuming repo** — typically
`quantbox-lab/research/<topic>/blocks.py` — so the engine stays agnostic.

Lab blocks register at import time:

```python
# quantbox-lab/research/my_strategy/blocks.py
from quantbox.plugins.pipeline.blocks import register_block, ReportBlock

register_block(ReportBlock(name="my_strategy.factor_loadings", ...))
```

Then any config that imports this module before running the pipeline gets
the block. The `name` should be namespaced (`my_strategy.foo`) to avoid
collisions with other lab modules.

## Conventions

- **Naming** — `snake_case` for the block name. Prefix with the strategy
  shortname for strategy-specific blocks (`donchian.overlay`,
  `factor.loadings`).
- **Payload Series/DataFrames** — pass `pandas` objects directly. The reporter
  serialises to Plotly JSON; don't pre-convert to lists.
- **Return `None`** — when the block can't render with the given inputs
  (empty Series, missing column). The template auto-hides the card.
- **Editorial layout** — the template applies a common Plotly layout
  (fonts, palette, legend position) AFTER the builder runs, so don't fight
  it. Builders should focus on data + structural axes; let the editorial
  override handle typography and palette.

## See also

- `src/quantbox/plugins/pipeline/blocks.py` — registry implementation +
  generic block builders.
- `src/quantbox/plugins/pipeline/_report.py` — diagnostic dispatch +
  legacy chart builders (will be migrated to the registry over time).
- `src/quantbox/plugins/pipeline/templates/research_report.html` — the
  editorial template.
- `tests/test_report_blocks.py` — registry + generic block tests.
