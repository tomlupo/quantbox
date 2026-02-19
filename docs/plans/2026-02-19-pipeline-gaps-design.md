# Design: Close Quant Pipeline Gaps

**Date:** 2026-02-19
**Scope:** High + Medium priority gaps from pipeline coverage analysis
**Repos:** quantbox (core), quantbox-datasets, quantbox-lab

---

## Problem

QuantBox covers backtesting (stage 5) and execution (stage 8) well, but has significant gaps in statistical validation (stage 6), feature engineering (stage 3), data breadth (stage 2), risk depth (stage 7), and monitoring (stage 9). These gaps make it easy to ship overfit strategies to production.

## Approach

Layered plugins across existing repos:
- **quantbox-core** — new plugin protocols + builtin implementations
- **quantbox-datasets** — expanded curated data (equities, macro, fundamentals)
- **quantbox-lab** — professional config-driven research pipelines

---

## 1. New Plugin Protocols (quantbox-core)

### 1.1 FeaturePlugin

Reusable feature engineering layer. Extracts logic from `ml_strategy.py` into standalone plugins that any strategy can consume.

```python
class FeaturePlugin(Protocol):
    meta: PluginMeta

    def compute(
        self,
        data: dict[str, pd.DataFrame],  # prices, volume, market_cap, etc.
        params: dict[str, Any],
    ) -> pd.DataFrame:  # MultiIndex: (date, symbol) x feature_name
        ...
```

**Builtin implementations:**

| Plugin | Features |
|--------|----------|
| `features.technical.v1` | RSI, MACD, Bollinger Bands, ATR, SMA ratios, momentum, volatility (extracted from `_FeatureEngineer`) |
| `features.cross_sectional.v1` | Z-score, percentile rank, sector-relative normalization across universe at each date |
| `features.macro.v1` | Macro regime indicators (yield curve slope, VIX percentile, credit spreads) when macro data available |

**Output format:** Stacked DataFrame with `(date, symbol)` MultiIndex and one column per feature. Strategies call `feature_plugin.compute(data, params)` to get features without reimplementing them.

**Integration with ML strategy:** `ml_prediction.v1` gains an optional `features` config key. When set, it delegates to FeaturePlugin(s) instead of its internal `_FeatureEngineer`. Internal feature code remains as fallback for backward compatibility.

### 1.2 ValidationPlugin

Post-backtest statistical rigor. Runs after backtest pipeline produces results.

```python
class ValidationPlugin(Protocol):
    meta: PluginMeta

    def validate(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        benchmark: pd.DataFrame | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:  # {"findings": [...], "metrics": {...}, "passed": bool}
        ...
```

**Builtin implementations:**

| Plugin | Purpose | Key params |
|--------|---------|------------|
| `validation.walk_forward.v1` | Rolling OOS splits, IS vs OOS Sharpe degradation | `n_splits`, `train_ratio` |
| `validation.statistical.v1` | Deflated Sharpe Ratio, multiple hypothesis correction (Bonferroni, FDR), bootstrap CIs | `n_trials`, `confidence`, `n_strategies_tested` |
| `validation.turnover.v1` | Turnover analysis, cost-adjusted returns, capacity estimation | `cost_bps`, `aum_usd` |
| `validation.regime.v1` | Conditional performance by market regime (trending/mean-reverting/crisis) | `regime_method` (vol-based or return-based) |
| `validation.benchmark.v1` | Alpha/beta decomposition, factor attribution, information ratio | `benchmark` (symbol or DataFrame) |

**Pipeline integration:** The backtest pipeline gains an optional `validation` config section. When present, validation plugins run after backtesting and their findings are included in `run_manifest.json` and the metrics artifact.

### 1.3 MonitorPlugin

Runtime monitoring for live/paper trading, integrated with the trading pipeline.

```python
class MonitorPlugin(Protocol):
    meta: PluginMeta

    def check(
        self,
        result: RunResult,
        history: list[RunResult] | None,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:  # alerts: [{level, rule, detail, action}]
        ...
```

**Builtin implementations:**

| Plugin | Purpose | Key params |
|--------|---------|------------|
| `monitor.drawdown.v1` | Max drawdown alerts, kill-switch (halt trading if threshold breached) | `max_drawdown`, `action` (warn/halt) |
| `monitor.signal_decay.v1` | Rolling Sharpe / hit rate tracking, alerts on degradation | `window`, `min_sharpe`, `min_hit_rate` |
| `monitor.pnl_attribution.v1` | Decomposes PnL into alpha, factor, transaction costs | `benchmark`, `factor_model` |

**Kill-switch mechanism:** When `monitor.drawdown.v1` returns an alert with `action: "halt"`, the trading pipeline writes a halt file to `artifacts/halt.json`. Subsequent runs check for this file and skip execution until manually cleared. This is additive — existing approval gate remains separate.

**Telegram integration:** Enhanced `telegram.publisher.v1` gains awareness of monitor alerts. When alerts exist in `RunResult.notes`, the publisher formats and sends them with appropriate severity (normal, warning, critical).

### 1.4 Enhanced Risk Plugins

New RiskPlugin implementations (same protocol, no protocol change):

| Plugin | Purpose |
|--------|---------|
| `risk.factor_exposure.v1` | Check portfolio beta, sector, market-cap factor exposures against limits |
| `risk.correlation.v1` | Alert when portfolio correlation to existing positions or other strategies exceeds threshold |
| `risk.drawdown_control.v1` | Scale positions or halt on portfolio-level drawdown (works with monitor.drawdown.v1) |

---

## 2. Data Expansion (quantbox-datasets)

### 2.1 New Dataset Modules

```
quantbox-datasets/
├── src/quantbox_datasets/
│   ├── crypto/              # existing
│   ├── equities/            # NEW
│   │   ├── us_stocks.py     # S&P 500, Russell 2000 via yfinance
│   │   └── etf_universe.py  # SPY, QQQ, sector/bond/commodity ETFs
│   ├── macro/               # NEW
│   │   ├── fred.py          # FRED: GDP, CPI, unemployment, yields, VIX
│   │   └── indicators.py    # derived: yield curve slope, real rates, credit spreads
│   ├── fundamental/         # NEW
│   │   └── yfinance_fundamentals.py  # P/E, P/B, EPS, revenue quarterly
│   └── registry.py          # CLI: quantbox-datasets fetch --source equities --universe sp500
```

### 2.2 Output Format

All datasets produce Parquet in quantbox's wide-format convention:
- `prices.parquet` — date x symbol (close)
- `volume.parquet` — date x symbol
- `market_cap.parquet` — date x symbol
- `fundamentals.parquet` — date x symbol x metric (stacked)
- `macro.parquet` — date x indicator (wide)

### 2.3 Data Plugin

`dataset.curated.v2` data plugin in quantbox-core wraps quantbox-datasets:

```yaml
data:
  name: "dataset.curated.v2"
  params:
    source: equities        # or: crypto, macro, multi
    universe: sp500          # or: top100, etf_core, custom
    lookback_days: 1260
    include_fundamentals: true
    include_macro: true
```

Plugin calls quantbox-datasets, caches to `data/curated/`, returns standard wide DataFrames.

---

## 3. Research Pipelines (quantbox-lab)

### 3.1 Directory Structure

```
quantbox-lab/
├── configs/
│   ├── research/
│   │   ├── crypto_trend_validation.yaml
│   │   ├── ml_prediction_walkforward.yaml
│   │   ├── cross_asset_momentum_study.yaml
│   │   ├── strategy_comparison.yaml
│   │   └── regime_analysis.yaml
│   └── data/
│       ├── fetch_crypto.yaml
│       ├── fetch_equities.yaml
│       └── fetch_macro.yaml
├── scripts/
│   ├── run_research.py          # main: config -> backtest + validation + report
│   ├── run_comparison.py        # compare N strategies side-by-side
│   ├── generate_report.py       # markdown + charts from artifacts
│   └── fetch_data.py            # data download orchestrator
├── notebooks/
│   └── visualize_results.ipynb  # interactive result exploration
├── reports/                     # generated reports
├── artifacts/                   # run outputs
└── data/                        # cached datasets
```

### 3.2 Research Config Schema

```yaml
research:
  name: "crypto_trend_full_validation"

  backtest:
    pipeline: "backtest.pipeline.v1"
    params:
      engine: vectorbt
      fees: 0.001
      rebalancing_freq: 1

  strategies:
    - name: "strategy.crypto_trend.v1"
      params:
        lookback_days: 365

  features:
    - name: "features.technical.v1"
    - name: "features.cross_sectional.v1"

  validation:
    - name: "validation.walk_forward.v1"
      params:
        n_splits: 5
        train_ratio: 0.7
    - name: "validation.statistical.v1"
      params:
        n_trials: 100
        confidence: 0.95
    - name: "validation.turnover.v1"
      params:
        cost_bps: 10
    - name: "validation.benchmark.v1"
      params:
        benchmark: "BTC"
    - name: "validation.regime.v1"

  data:
    name: "dataset.curated.v2"
    params:
      source: crypto
      universe: top100
      lookback_days: 1095
```

### 3.3 Report Output

`generate_report.py` produces professional markdown:

```markdown
# Research Report: crypto_trend_full_validation
## 2026-02-19

### Performance Summary
| Metric | In-Sample | Out-of-Sample | Degradation |
|--------|-----------|---------------|-------------|
| Sharpe | 1.42 | 0.89 | -37% |

### Statistical Validation
- Deflated Sharpe Ratio: 0.71 (p < 0.05)
- Haircut Sharpe (FDR-adjusted): 0.64
- Bootstrap 95% CI: [0.45, 1.12]

### Turnover Analysis
- Annual turnover: 12.4x
- Cost-adjusted Sharpe: 0.78

### Regime Breakdown
| Regime | Return | Sharpe | % Time |

### Factor Attribution
- Alpha: 18% ann. | Beta: 7% | Costs: -3%
```

### 3.4 Research Scripts

**`run_research.py`** — main entry point:
1. Load research config
2. Fetch/cache data via quantbox-datasets
3. Run backtest pipeline via quantbox CLI
4. Run each validation plugin on results
5. Save validation artifacts
6. Generate report

**`run_comparison.py`** — strategy comparison:
1. Load comparison config (list of strategies + shared data/validation)
2. Run each strategy through backtest + validation
3. Produce side-by-side comparison report

---

## 4. Integration Points

### Pipeline Config Changes

The backtest pipeline config gains optional sections:

```yaml
plugins:
  # ... existing sections ...
  features:           # NEW: optional
    - name: "features.technical.v1"
  validation:         # NEW: optional
    - name: "validation.walk_forward.v1"
  monitors:           # NEW: optional (trading pipeline only)
    - name: "monitor.drawdown.v1"
```

### Manifest Changes

`plugins/manifest.yaml` gains new plugin lists:

```yaml
plugins:
  builtins:
    features:
      - features.technical.v1
      - features.cross_sectional.v1
      - features.macro.v1
    validation:
      - validation.walk_forward.v1
      - validation.statistical.v1
      - validation.turnover.v1
      - validation.regime.v1
      - validation.benchmark.v1
    monitors:
      - monitor.drawdown.v1
      - monitor.signal_decay.v1
      - monitor.pnl_attribution.v1
```

### contracts.py Changes

Add `FeaturePlugin`, `ValidationPlugin`, `MonitorPlugin` protocols. Add `"feature"`, `"validation"`, `"monitor"` to `PluginKind` literal. Backward compatible — existing code sees no changes.

---

## 5. Dependencies

### quantbox-core
- No new required deps (numpy, pandas, scipy already available)
- `features.macro.v1` needs macro data available (optional graceful degradation)

### quantbox-datasets
- `yfinance` (already optional dep)
- `pandas-datareader` (for FRED)
- `fredapi` (optional, for direct FRED API access)

### quantbox-lab
- `matplotlib` / `plotly` (for report charts)
- Depends on quantbox @ dev + quantbox-datasets

---

## 6. Non-Goals

- No web UI or dashboard (CLI + artifacts + Telegram)
- No HFT or tick-level features
- No deep learning in this iteration (sklearn sufficient)
- No custom exchange gateways
- No multi-agent orchestration
