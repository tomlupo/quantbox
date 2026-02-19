# Pipeline Gaps Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close high and medium priority gaps in quantbox's quant research pipeline — add Feature, Validation, and Monitor plugin types, expand data sources, enhance risk, and build professional research pipelines in quantbox-lab.

**Architecture:** New protocols added to `contracts.py`, builtin plugins in `plugins/`, registry expanded in `builtins.py` and `registry.py`. Backtest pipeline gains optional `validation` and `features` config sections. quantbox-datasets gets equities/macro/fundamental modules. quantbox-lab gets config-driven research scripts.

**Tech Stack:** Python 3.12, pandas, numpy, scipy, sklearn, yfinance, pandas-datareader, vectorbt, DuckDB, Parquet

---

## Phase 1: Foundation — Protocols & Registry

### Task 1: Add FeaturePlugin, ValidationPlugin, MonitorPlugin protocols to contracts.py

**Files:**
- Modify: `packages/quantbox-core/src/quantbox/contracts.py`
- Test: `tests/test_contracts_new_protocols.py`

**Step 1: Write failing test**

```python
# tests/test_contracts_new_protocols.py
"""Tests for new plugin protocols: Feature, Validation, Monitor."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pandas as pd
from quantbox.contracts import (
    FeaturePlugin,
    MonitorPlugin,
    PluginMeta,
    RunResult,
    ValidationPlugin,
)


@dataclass
class _StubFeature:
    meta = PluginMeta(name="test.feature", kind="feature", version="0.1.0", core_compat=">=0.1,<1")

    def compute(self, data: dict[str, pd.DataFrame], params: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame({"feat_a": [1.0]}, index=pd.MultiIndex.from_tuples([("2026-01-01", "BTC")], names=["date", "symbol"]))


@dataclass
class _StubValidation:
    meta = PluginMeta(name="test.validation", kind="validation", version="0.1.0", core_compat=">=0.1,<1")

    def validate(self, returns: pd.DataFrame, weights: pd.DataFrame, benchmark: pd.DataFrame | None, params: dict[str, Any]) -> dict[str, Any]:
        return {"findings": [], "metrics": {"sharpe": 1.0}, "passed": True}


@dataclass
class _StubMonitor:
    meta = PluginMeta(name="test.monitor", kind="monitor", version="0.1.0", core_compat=">=0.1,<1")

    def check(self, result: RunResult, history: list[RunResult] | None, params: dict[str, Any]) -> list[dict[str, Any]]:
        return []


def test_feature_plugin_protocol():
    plugin: FeaturePlugin = _StubFeature()
    result = plugin.compute({"prices": pd.DataFrame()}, {})
    assert isinstance(result, pd.DataFrame)
    assert result.index.names == ["date", "symbol"]


def test_validation_plugin_protocol():
    plugin: ValidationPlugin = _StubValidation()
    result = plugin.validate(pd.DataFrame(), pd.DataFrame(), None, {})
    assert result["passed"] is True
    assert "metrics" in result


def test_monitor_plugin_protocol():
    plugin: MonitorPlugin = _StubMonitor()
    result_obj = RunResult(run_id="test", pipeline_name="test", mode="backtest", asof="2026-01-01", artifacts={}, metrics={}, notes={})
    alerts = plugin.check(result_obj, None, {})
    assert alerts == []
```

**Step 2: Run test to verify it fails**

Run: `cd /home/tom/workspace/projects/quantbox && uv run pytest tests/test_contracts_new_protocols.py -v`
Expected: FAIL — `ImportError: cannot import name 'FeaturePlugin'`

**Step 3: Implement protocols in contracts.py**

Add to `contracts.py` after the existing `PluginKind` literal and protocols:

```python
# Update PluginKind to include new types
PluginKind = Literal["pipeline", "broker", "data", "publisher", "risk", "strategy", "rebalancing", "feature", "validation", "monitor"]


class FeaturePlugin(Protocol):
    """Computes reusable features from market data.

    Returns a stacked DataFrame with (date, symbol) MultiIndex and one
    column per feature. Strategies can consume features instead of
    reimplementing their own feature engineering.
    """

    meta: PluginMeta

    def compute(
        self,
        data: dict[str, pd.DataFrame],
        params: dict[str, Any],
    ) -> pd.DataFrame: ...


class ValidationPlugin(Protocol):
    """Post-backtest statistical validation.

    Runs after backtest pipeline produces results. Returns findings,
    metrics, and a pass/fail verdict.
    """

    meta: PluginMeta

    def validate(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        benchmark: pd.DataFrame | None,
        params: dict[str, Any],
    ) -> dict[str, Any]: ...


class MonitorPlugin(Protocol):
    """Runtime monitoring for live/paper trading.

    Checks run results and history for anomalies, returns alerts.
    """

    meta: PluginMeta

    def check(
        self,
        result: RunResult,
        history: list[RunResult] | None,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]: ...
```

**Step 4: Run test to verify it passes**

Run: `cd /home/tom/workspace/projects/quantbox && uv run pytest tests/test_contracts_new_protocols.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add packages/quantbox-core/src/quantbox/contracts.py tests/test_contracts_new_protocols.py
git commit -m "feat: add FeaturePlugin, ValidationPlugin, MonitorPlugin protocols"
```

### Task 2: Extend registry to support new plugin types

**Files:**
- Modify: `packages/quantbox-core/src/quantbox/registry.py`
- Test: `tests/test_registry_new_types.py`

**Step 1: Write failing test**

```python
# tests/test_registry_new_types.py
"""Test that PluginRegistry discovers feature, validation, and monitor plugins."""
from quantbox.registry import PluginRegistry


def test_registry_has_feature_field():
    reg = PluginRegistry.discover()
    assert hasattr(reg, "features")
    assert isinstance(reg.features, dict)


def test_registry_has_validation_field():
    reg = PluginRegistry.discover()
    assert hasattr(reg, "validations")
    assert isinstance(reg.validations, dict)


def test_registry_has_monitor_field():
    reg = PluginRegistry.discover()
    assert hasattr(reg, "monitors")
    assert isinstance(reg.monitors, dict)
```

**Step 2: Run test to verify it fails**

Run: `cd /home/tom/workspace/projects/quantbox && uv run pytest tests/test_registry_new_types.py -v`
Expected: FAIL — `AttributeError: 'PluginRegistry' object has no attribute 'features'`

**Step 3: Update registry.py**

Add to `ENTRYPOINT_GROUPS`:
```python
ENTRYPOINT_GROUPS = {
    # ... existing ...
    "feature": "quantbox.features",
    "validation": "quantbox.validations",
    "monitor": "quantbox.monitors",
}
```

Add fields to `PluginRegistry`:
```python
from .contracts import (
    # ... existing imports ...
    FeaturePlugin,
    MonitorPlugin,
    ValidationPlugin,
)

@dataclass
class PluginRegistry:
    # ... existing fields ...
    features: dict[str, type[FeaturePlugin]] = field(default_factory=dict)
    validations: dict[str, type[ValidationPlugin]] = field(default_factory=dict)
    monitors: dict[str, type[MonitorPlugin]] = field(default_factory=dict)

    @staticmethod
    def discover() -> PluginRegistry:
        builtins = builtin_plugins()
        return PluginRegistry(
            # ... existing ...
            features={**builtins.get("feature", {}), **_load_group(ENTRYPOINT_GROUPS["feature"])},
            validations={**builtins.get("validation", {}), **_load_group(ENTRYPOINT_GROUPS["validation"])},
            monitors={**builtins.get("monitor", {}), **_load_group(ENTRYPOINT_GROUPS["monitor"])},
        )
```

**Step 4: Run test to verify it passes**

Run: `cd /home/tom/workspace/projects/quantbox && uv run pytest tests/test_registry_new_types.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/quantbox-core/src/quantbox/registry.py tests/test_registry_new_types.py
git commit -m "feat: extend PluginRegistry with feature, validation, monitor types"
```

---

## Phase 2: Feature Plugins

### Task 3: Create features.technical.v1 plugin

**Files:**
- Create: `packages/quantbox-core/src/quantbox/plugins/features/__init__.py`
- Create: `packages/quantbox-core/src/quantbox/plugins/features/technical.py`
- Test: `tests/test_feature_technical.py`

**Step 1: Write failing test**

```python
# tests/test_feature_technical.py
"""Tests for features.technical.v1 plugin."""
import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.features.technical import TechnicalFeatures


@pytest.fixture
def sample_prices():
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    rng = np.random.default_rng(42)
    btc = 40000 * np.cumprod(1 + rng.normal(0.001, 0.03, 100))
    eth = 2500 * np.cumprod(1 + rng.normal(0.001, 0.04, 100))
    return pd.DataFrame({"BTC": btc, "ETH": eth}, index=dates)


@pytest.fixture
def sample_volume(sample_prices):
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {c: rng.uniform(1e6, 1e8, len(sample_prices)) for c in sample_prices.columns},
        index=sample_prices.index,
    )


def test_technical_features_shape(sample_prices):
    plugin = TechnicalFeatures()
    result = plugin.compute({"prices": sample_prices}, {})
    assert isinstance(result, pd.DataFrame)
    assert result.index.names == ["date", "symbol"]
    # Should have features for both BTC and ETH
    symbols = result.index.get_level_values("symbol").unique()
    assert set(symbols) == {"BTC", "ETH"}


def test_technical_features_contains_expected(sample_prices):
    plugin = TechnicalFeatures()
    result = plugin.compute({"prices": sample_prices}, {})
    expected_features = ["rsi_14", "macd", "bb_position_20d", "return_5d", "volatility_20d"]
    for feat in expected_features:
        assert feat in result.columns, f"Missing feature: {feat}"


def test_technical_features_with_volume(sample_prices, sample_volume):
    plugin = TechnicalFeatures()
    result = plugin.compute({"prices": sample_prices, "volume": sample_volume}, {})
    assert "volume_ratio_20d" in result.columns


def test_technical_features_no_nans_in_tail(sample_prices):
    plugin = TechnicalFeatures()
    result = plugin.compute({"prices": sample_prices}, {})
    tail = result.tail(20)
    # Tail should have very few NaNs (warm-up period excluded)
    nan_ratio = tail.isna().sum().sum() / (tail.shape[0] * tail.shape[1])
    assert nan_ratio < 0.1


def test_technical_features_meta():
    plugin = TechnicalFeatures()
    assert plugin.meta.name == "features.technical.v1"
    assert plugin.meta.kind == "feature"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/tom/workspace/projects/quantbox && uv run pytest tests/test_feature_technical.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement**

Create `packages/quantbox-core/src/quantbox/plugins/features/__init__.py`:
```python
from .technical import TechnicalFeatures

__all__ = ["TechnicalFeatures"]
```

Create `packages/quantbox-core/src/quantbox/plugins/features/technical.py`:

Extract and refactor the `_FeatureEngineer` from `ml_strategy.py` into a standalone FeaturePlugin. The plugin takes wide-format DataFrames (`prices`, optional `volume`) and returns a stacked `(date, symbol)` MultiIndex DataFrame.

Key features to compute per symbol:
- Returns at multiple horizons (5d, 10d, 20d, 60d)
- Volatility at multiple horizons
- Momentum ratios
- SMA ratios and slopes
- RSI (14, 28)
- Bollinger Band position (20d)
- MACD, signal, histogram (normalized)
- ATR proxy (14d, 28d)
- Volume ratio and trend (when available)
- Day-of-week cyclical encoding

The implementation should use the exact same math as `_FeatureEngineer` in `ml_strategy.py` (lines 48-116) but wrapped in the `FeaturePlugin` protocol. Accept `lookback_periods` from params (default `[5, 10, 20, 60]`).

**Step 4: Run test to verify it passes**

Run: `cd /home/tom/workspace/projects/quantbox && uv run pytest tests/test_feature_technical.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add packages/quantbox-core/src/quantbox/plugins/features/ tests/test_feature_technical.py
git commit -m "feat: add features.technical.v1 plugin (extracted from ml_strategy)"
```

### Task 4: Create features.cross_sectional.v1 plugin

**Files:**
- Create: `packages/quantbox-core/src/quantbox/plugins/features/cross_sectional.py`
- Modify: `packages/quantbox-core/src/quantbox/plugins/features/__init__.py`
- Test: `tests/test_feature_cross_sectional.py`

**Step 1: Write failing test**

```python
# tests/test_feature_cross_sectional.py
"""Tests for features.cross_sectional.v1 plugin."""
import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.features.cross_sectional import CrossSectionalFeatures


@pytest.fixture
def sample_prices():
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    rng = np.random.default_rng(42)
    data = {}
    for sym in ["BTC", "ETH", "SOL", "ADA", "DOT"]:
        data[sym] = 100 * np.cumprod(1 + rng.normal(0.001, 0.03, 100))
    return pd.DataFrame(data, index=dates)


def test_cross_sectional_z_scores(sample_prices):
    plugin = CrossSectionalFeatures()
    result = plugin.compute({"prices": sample_prices}, {"methods": ["zscore"]})
    assert isinstance(result, pd.DataFrame)
    assert result.index.names == ["date", "symbol"]
    assert "return_20d_zscore" in result.columns


def test_cross_sectional_percentile_rank(sample_prices):
    plugin = CrossSectionalFeatures()
    result = plugin.compute({"prices": sample_prices}, {"methods": ["percentile"]})
    assert "return_20d_percentile" in result.columns
    # Percentile values should be between 0 and 1
    pct_col = result["return_20d_percentile"].dropna()
    assert pct_col.min() >= 0.0
    assert pct_col.max() <= 1.0


def test_cross_sectional_both_methods(sample_prices):
    plugin = CrossSectionalFeatures()
    result = plugin.compute({"prices": sample_prices}, {"methods": ["zscore", "percentile"]})
    assert "return_20d_zscore" in result.columns
    assert "return_20d_percentile" in result.columns


def test_cross_sectional_meta():
    plugin = CrossSectionalFeatures()
    assert plugin.meta.name == "features.cross_sectional.v1"
    assert plugin.meta.kind == "feature"
```

**Step 2: Run test, verify fails**

Run: `cd /home/tom/workspace/projects/quantbox && uv run pytest tests/test_feature_cross_sectional.py -v`

**Step 3: Implement**

`cross_sectional.py` computes cross-sectional features at each date across the universe:
- For each horizon in `[5, 10, 20, 60]`, compute returns
- At each date, rank returns cross-sectionally:
  - `zscore`: `(x - mean) / std` across symbols at that date
  - `percentile`: `rank / count` across symbols at that date
- Output: stacked `(date, symbol)` DataFrame with columns like `return_20d_zscore`, `return_20d_percentile`

Params: `methods` (list of `"zscore"`, `"percentile"`), `horizons` (list of ints, default `[5, 10, 20, 60]`)

**Step 4: Run test, verify passes**

**Step 5: Commit**

```bash
git add packages/quantbox-core/src/quantbox/plugins/features/ tests/test_feature_cross_sectional.py
git commit -m "feat: add features.cross_sectional.v1 plugin"
```

### Task 5: Register feature plugins in builtins.py

**Files:**
- Modify: `packages/quantbox-core/src/quantbox/plugins/builtins.py`
- Test: `tests/test_feature_discovery.py`

**Step 1: Write failing test**

```python
# tests/test_feature_discovery.py
"""Test that feature plugins are discoverable via registry."""
from quantbox.registry import PluginRegistry


def test_technical_features_in_registry():
    reg = PluginRegistry.discover()
    assert "features.technical.v1" in reg.features


def test_cross_sectional_features_in_registry():
    reg = PluginRegistry.discover()
    assert "features.cross_sectional.v1" in reg.features
```

**Step 2: Run test, verify fails**

**Step 3: Add to builtins.py**

Add imports and `"feature": _map(TechnicalFeatures, CrossSectionalFeatures)` to the `builtins()` dict.

**Step 4: Run test, verify passes**

**Step 5: Commit**

```bash
git add packages/quantbox-core/src/quantbox/plugins/builtins.py tests/test_feature_discovery.py
git commit -m "feat: register feature plugins in builtins"
```

---

## Phase 3: Validation Plugins

### Task 6: Create validation.walk_forward.v1 plugin

**Files:**
- Create: `packages/quantbox-core/src/quantbox/plugins/validation/__init__.py`
- Create: `packages/quantbox-core/src/quantbox/plugins/validation/walk_forward.py`
- Test: `tests/test_validation_walk_forward.py`

**Step 1: Write failing test**

```python
# tests/test_validation_walk_forward.py
"""Tests for validation.walk_forward.v1 plugin."""
import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.validation.walk_forward import WalkForwardValidation


@pytest.fixture
def sample_returns():
    dates = pd.date_range("2024-01-01", periods=500, freq="D")
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.001, 0.02, 500), index=dates, name="returns")


@pytest.fixture
def sample_weights():
    dates = pd.date_range("2024-01-01", periods=500, freq="D")
    return pd.DataFrame({"BTC": 0.5, "ETH": 0.5}, index=dates)


def test_walk_forward_returns_findings(sample_returns, sample_weights):
    plugin = WalkForwardValidation()
    result = plugin.validate(sample_returns.to_frame(), sample_weights, None, {"n_splits": 3, "train_ratio": 0.7})
    assert "findings" in result
    assert "metrics" in result
    assert "passed" in result


def test_walk_forward_metrics_contain_is_oos(sample_returns, sample_weights):
    plugin = WalkForwardValidation()
    result = plugin.validate(sample_returns.to_frame(), sample_weights, None, {"n_splits": 3, "train_ratio": 0.7})
    metrics = result["metrics"]
    assert "is_sharpe_mean" in metrics
    assert "oos_sharpe_mean" in metrics
    assert "sharpe_degradation" in metrics


def test_walk_forward_detects_overfit(sample_weights):
    """A strategy with decaying alpha should show IS >> OOS."""
    dates = pd.date_range("2024-01-01", periods=500, freq="D")
    rng = np.random.default_rng(42)
    # IS period has positive returns, OOS has noise
    returns = pd.Series(rng.normal(-0.001, 0.02, 500), index=dates)
    plugin = WalkForwardValidation()
    result = plugin.validate(returns.to_frame(), sample_weights, None, {"n_splits": 3, "train_ratio": 0.7})
    assert isinstance(result["metrics"]["sharpe_degradation"], float)


def test_walk_forward_meta():
    plugin = WalkForwardValidation()
    assert plugin.meta.name == "validation.walk_forward.v1"
    assert plugin.meta.kind == "validation"
```

**Step 2: Run test, verify fails**

**Step 3: Implement**

`walk_forward.py` implements expanding/rolling walk-forward OOS testing:
- Split the return series into `n_splits` sequential folds
- For each fold: `train_ratio` of data = IS, remainder = OOS
- Compute Sharpe ratio on IS and OOS portions separately
- Report: `is_sharpe_mean`, `oos_sharpe_mean`, `sharpe_degradation` (% drop)
- Finding if OOS Sharpe < 0 or degradation > 50%

Params: `n_splits` (int, default 5), `train_ratio` (float, default 0.7), `trading_days` (int, default 365)

**Step 4: Run test, verify passes**

**Step 5: Commit**

```bash
git add packages/quantbox-core/src/quantbox/plugins/validation/ tests/test_validation_walk_forward.py
git commit -m "feat: add validation.walk_forward.v1 plugin"
```

### Task 7: Create validation.statistical.v1 plugin

**Files:**
- Create: `packages/quantbox-core/src/quantbox/plugins/validation/statistical.py`
- Modify: `packages/quantbox-core/src/quantbox/plugins/validation/__init__.py`
- Test: `tests/test_validation_statistical.py`

**Step 1: Write failing test**

```python
# tests/test_validation_statistical.py
"""Tests for validation.statistical.v1 plugin."""
import numpy as np
import pandas as pd

from quantbox.plugins.validation.statistical import StatisticalValidation


def _make_returns(n=500, seed=42):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.001, 0.02, n), index=dates)


def _make_weights(n=500):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"BTC": 0.5, "ETH": 0.5}, index=dates)


def test_statistical_deflated_sharpe():
    plugin = StatisticalValidation()
    returns = _make_returns()
    result = plugin.validate(returns.to_frame(), _make_weights(), None, {"n_trials": 50, "confidence": 0.95})
    assert "deflated_sharpe" in result["metrics"]
    assert isinstance(result["metrics"]["deflated_sharpe"], float)


def test_statistical_bootstrap_ci():
    plugin = StatisticalValidation()
    returns = _make_returns()
    result = plugin.validate(returns.to_frame(), _make_weights(), None, {"n_bootstrap": 500, "confidence": 0.95})
    assert "sharpe_ci_lower" in result["metrics"]
    assert "sharpe_ci_upper" in result["metrics"]
    assert result["metrics"]["sharpe_ci_lower"] <= result["metrics"]["sharpe_ci_upper"]


def test_statistical_haircut_sharpe():
    plugin = StatisticalValidation()
    returns = _make_returns()
    result = plugin.validate(returns.to_frame(), _make_weights(), None, {"n_strategies_tested": 10})
    assert "haircut_sharpe" in result["metrics"]
    # Haircut should be <= original
    assert result["metrics"]["haircut_sharpe"] <= result["metrics"].get("observed_sharpe", float("inf"))


def test_statistical_meta():
    plugin = StatisticalValidation()
    assert plugin.meta.name == "validation.statistical.v1"
```

**Step 2: Run test, verify fails**

**Step 3: Implement**

`statistical.py` computes:
- **Deflated Sharpe Ratio** (Bailey & Lopez de Prado): adjusts observed Sharpe for the number of trials/strategies tested using the DSR formula. Uses scipy.stats.norm.
- **Bootstrap confidence intervals**: resample returns with replacement N times, compute Sharpe each time, report percentile CI
- **Haircut Sharpe** (Harvey, Liu & Zhu): FDR-adjusted Sharpe = observed_sharpe * (1 - haircut_factor), where haircut depends on `n_strategies_tested`

Params: `n_trials` (int, default 100), `n_bootstrap` (int, default 1000), `confidence` (float, default 0.95), `n_strategies_tested` (int, default 1), `trading_days` (int, default 365)

**Step 4: Run test, verify passes**

**Step 5: Commit**

```bash
git add packages/quantbox-core/src/quantbox/plugins/validation/ tests/test_validation_statistical.py
git commit -m "feat: add validation.statistical.v1 plugin (Deflated Sharpe, bootstrap CI)"
```

### Task 8: Create validation.turnover.v1 plugin

**Files:**
- Create: `packages/quantbox-core/src/quantbox/plugins/validation/turnover.py`
- Modify: `packages/quantbox-core/src/quantbox/plugins/validation/__init__.py`
- Test: `tests/test_validation_turnover.py`

**Step 1: Write failing test**

```python
# tests/test_validation_turnover.py
"""Tests for validation.turnover.v1 plugin."""
import numpy as np
import pandas as pd

from quantbox.plugins.validation.turnover import TurnoverValidation


def _make_data(n=252):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.001, 0.02, n), index=dates)
    # Weights that change daily = high turnover
    weights = pd.DataFrame(
        {"BTC": rng.uniform(0.3, 0.7, n), "ETH": rng.uniform(0.3, 0.7, n)},
        index=dates,
    )
    weights = weights.div(weights.sum(axis=1), axis=0)
    return returns, weights


def test_turnover_metrics():
    returns, weights = _make_data()
    plugin = TurnoverValidation()
    result = plugin.validate(returns.to_frame(), weights, None, {"cost_bps": 10})
    assert "annual_turnover" in result["metrics"]
    assert "cost_adjusted_sharpe" in result["metrics"]
    assert "breakeven_cost_bps" in result["metrics"]


def test_turnover_high_vs_low():
    dates = pd.date_range("2024-01-01", periods=252, freq="D")
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.001, 0.02, 252), index=dates)

    # Static weights = zero turnover
    static = pd.DataFrame({"BTC": 0.5, "ETH": 0.5}, index=dates)
    plugin = TurnoverValidation()
    result_static = plugin.validate(returns.to_frame(), static, None, {"cost_bps": 10})
    assert result_static["metrics"]["annual_turnover"] < 0.01


def test_turnover_meta():
    assert TurnoverValidation().meta.name == "validation.turnover.v1"
```

**Step 2: Run test, verify fails**

**Step 3: Implement**

`turnover.py` computes:
- **Annual turnover**: `sum(abs(weight_changes)) / 2` summed daily, annualized
- **Cost-adjusted returns**: `returns - daily_turnover * cost_bps / 10000`
- **Cost-adjusted Sharpe**: Sharpe of cost-adjusted returns
- **Break-even cost**: max cost in bps where Sharpe stays > 0

Params: `cost_bps` (float, default 10), `trading_days` (int, default 365)

**Step 4: Run test, verify passes**

**Step 5: Commit**

```bash
git add packages/quantbox-core/src/quantbox/plugins/validation/ tests/test_validation_turnover.py
git commit -m "feat: add validation.turnover.v1 plugin"
```

### Task 9: Create validation.regime.v1 plugin

**Files:**
- Create: `packages/quantbox-core/src/quantbox/plugins/validation/regime.py`
- Test: `tests/test_validation_regime.py`

**Step 1: Write failing test**

```python
# tests/test_validation_regime.py
"""Tests for validation.regime.v1 plugin."""
import numpy as np
import pandas as pd

from quantbox.plugins.validation.regime import RegimeValidation


def _make_data(n=500):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.0005, 0.02, n), index=dates)
    weights = pd.DataFrame({"BTC": 0.5, "ETH": 0.5}, index=dates)
    return returns, weights


def test_regime_breakdown():
    returns, weights = _make_data()
    plugin = RegimeValidation()
    result = plugin.validate(returns.to_frame(), weights, None, {})
    assert "regime_breakdown" in result["metrics"]
    breakdown = result["metrics"]["regime_breakdown"]
    assert isinstance(breakdown, list)
    assert len(breakdown) > 0
    assert "regime" in breakdown[0]
    assert "sharpe" in breakdown[0]
    assert "pct_time" in breakdown[0]


def test_regime_labels():
    returns, weights = _make_data()
    plugin = RegimeValidation()
    result = plugin.validate(returns.to_frame(), weights, None, {})
    regimes = [r["regime"] for r in result["metrics"]["regime_breakdown"]]
    assert set(regimes) <= {"trending_up", "trending_down", "low_vol", "high_vol"}


def test_regime_meta():
    assert RegimeValidation().meta.name == "validation.regime.v1"
```

**Step 2: Run test, verify fails**

**Step 3: Implement**

`regime.py` classifies each day into a regime based on rolling statistics:
- Compute 60d rolling return and 60d rolling volatility
- Classify into regimes:
  - `trending_up`: rolling return > +1 std
  - `trending_down`: rolling return < -1 std
  - `high_vol`: rolling vol > median vol (and not trending)
  - `low_vol`: everything else
- Compute Sharpe, return, vol, and % time in each regime

Params: `window` (int, default 60), `regime_method` (str, default "vol_return")

**Step 4: Run test, verify passes**

**Step 5: Commit**

```bash
git add packages/quantbox-core/src/quantbox/plugins/validation/ tests/test_validation_regime.py
git commit -m "feat: add validation.regime.v1 plugin"
```

### Task 10: Create validation.benchmark.v1 plugin

**Files:**
- Create: `packages/quantbox-core/src/quantbox/plugins/validation/benchmark.py`
- Test: `tests/test_validation_benchmark.py`

**Step 1: Write failing test**

```python
# tests/test_validation_benchmark.py
"""Tests for validation.benchmark.v1 plugin."""
import numpy as np
import pandas as pd

from quantbox.plugins.validation.benchmark import BenchmarkValidation


def _make_data(n=500):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.001, 0.02, n), index=dates)
    weights = pd.DataFrame({"BTC": 0.5, "ETH": 0.5}, index=dates)
    benchmark = pd.Series(rng.normal(0.0005, 0.015, n), index=dates)
    return returns, weights, benchmark


def test_benchmark_alpha_beta():
    returns, weights, benchmark = _make_data()
    plugin = BenchmarkValidation()
    result = plugin.validate(returns.to_frame(), weights, benchmark.to_frame(), {})
    assert "alpha" in result["metrics"]
    assert "beta" in result["metrics"]
    assert "information_ratio" in result["metrics"]
    assert "tracking_error" in result["metrics"]


def test_benchmark_no_benchmark_graceful():
    returns, weights, _ = _make_data()
    plugin = BenchmarkValidation()
    result = plugin.validate(returns.to_frame(), weights, None, {})
    assert result["passed"] is True
    assert "alpha" not in result["metrics"]  # can't compute without benchmark


def test_benchmark_meta():
    assert BenchmarkValidation().meta.name == "validation.benchmark.v1"
```

**Step 2: Run test, verify fails**

**Step 3: Implement**

`benchmark.py` computes:
- **Alpha/Beta**: OLS regression of strategy returns on benchmark returns. Alpha annualized.
- **Information ratio**: annualized excess return / tracking error
- **Tracking error**: annualized std of excess returns
- **R-squared**: how much variance explained by benchmark

Uses numpy for regression (no sklearn needed): `beta = cov(r, b) / var(b)`, `alpha = mean(r) - beta * mean(b)`

Params: `trading_days` (int, default 365)

**Step 4: Run test, verify passes**

**Step 5: Commit**

```bash
git add packages/quantbox-core/src/quantbox/plugins/validation/ tests/test_validation_benchmark.py
git commit -m "feat: add validation.benchmark.v1 plugin"
```

### Task 11: Register all validation plugins in builtins.py

**Files:**
- Modify: `packages/quantbox-core/src/quantbox/plugins/validation/__init__.py`
- Modify: `packages/quantbox-core/src/quantbox/plugins/builtins.py`
- Test: `tests/test_validation_discovery.py`

**Step 1: Write failing test**

```python
# tests/test_validation_discovery.py
from quantbox.registry import PluginRegistry


def test_all_validation_plugins_registered():
    reg = PluginRegistry.discover()
    expected = [
        "validation.walk_forward.v1",
        "validation.statistical.v1",
        "validation.turnover.v1",
        "validation.regime.v1",
        "validation.benchmark.v1",
    ]
    for name in expected:
        assert name in reg.validations, f"Missing: {name}"
```

**Step 2: Run test, verify fails**

**Step 3: Update `__init__.py` and `builtins.py`**

Update `validation/__init__.py` with all imports.
Add `"validation": _map(...)` to `builtins()` dict.

**Step 4: Run test, verify passes**

**Step 5: Commit**

```bash
git add packages/quantbox-core/src/quantbox/plugins/validation/__init__.py packages/quantbox-core/src/quantbox/plugins/builtins.py tests/test_validation_discovery.py
git commit -m "feat: register all validation plugins in builtins"
```

---

## Phase 4: Monitor Plugins

### Task 12: Create monitor.drawdown.v1 plugin

**Files:**
- Create: `packages/quantbox-core/src/quantbox/plugins/monitor/__init__.py`
- Create: `packages/quantbox-core/src/quantbox/plugins/monitor/drawdown.py`
- Test: `tests/test_monitor_drawdown.py`

**Step 1: Write failing test**

```python
# tests/test_monitor_drawdown.py
"""Tests for monitor.drawdown.v1 plugin."""
from quantbox.contracts import RunResult
from quantbox.plugins.monitor.drawdown import DrawdownMonitor


def _make_result(total_return=-0.15, max_drawdown=-0.25):
    return RunResult(
        run_id="test", pipeline_name="trade.full_pipeline.v1",
        mode="live", asof="2026-02-19",
        artifacts={}, metrics={"total_return": total_return, "max_drawdown": max_drawdown},
        notes={},
    )


def test_drawdown_alert_triggered():
    plugin = DrawdownMonitor()
    result = _make_result(max_drawdown=-0.25)
    alerts = plugin.check(result, None, {"max_drawdown": -0.20, "action": "warn"})
    assert len(alerts) == 1
    assert alerts[0]["level"] == "warn"
    assert alerts[0]["rule"] == "max_drawdown_exceeded"


def test_drawdown_halt_action():
    plugin = DrawdownMonitor()
    result = _make_result(max_drawdown=-0.25)
    alerts = plugin.check(result, None, {"max_drawdown": -0.20, "action": "halt"})
    assert alerts[0]["action"] == "halt"


def test_drawdown_no_alert():
    plugin = DrawdownMonitor()
    result = _make_result(max_drawdown=-0.05)
    alerts = plugin.check(result, None, {"max_drawdown": -0.20})
    assert len(alerts) == 0


def test_drawdown_meta():
    assert DrawdownMonitor().meta.name == "monitor.drawdown.v1"
```

**Step 2: Run test, verify fails**

**Step 3: Implement**

`drawdown.py`: Checks `result.metrics["max_drawdown"]` against `params["max_drawdown"]` threshold. If breached, returns alert with `action` from params (default "warn"). Also checks cumulative return vs `max_loss` if specified.

Params: `max_drawdown` (float, default -0.20), `max_loss` (float, default -0.30), `action` (str, default "warn" — one of "warn", "halt")

**Step 4: Run test, verify passes**

**Step 5: Commit**

```bash
git add packages/quantbox-core/src/quantbox/plugins/monitor/ tests/test_monitor_drawdown.py
git commit -m "feat: add monitor.drawdown.v1 plugin"
```

### Task 13: Create monitor.signal_decay.v1 plugin

**Files:**
- Create: `packages/quantbox-core/src/quantbox/plugins/monitor/signal_decay.py`
- Test: `tests/test_monitor_signal_decay.py`

**Step 1: Write failing test**

```python
# tests/test_monitor_signal_decay.py
"""Tests for monitor.signal_decay.v1 plugin."""
from quantbox.contracts import RunResult
from quantbox.plugins.monitor.signal_decay import SignalDecayMonitor


def _make_results_history(sharpes):
    """Create a list of RunResults with given Sharpe values."""
    return [
        RunResult(
            run_id=f"run_{i}", pipeline_name="test", mode="live",
            asof=f"2026-02-{i+1:02d}", artifacts={},
            metrics={"sharpe": s, "win_rate": 0.5}, notes={},
        )
        for i, s in enumerate(sharpes)
    ]


def test_signal_decay_detected():
    plugin = SignalDecayMonitor()
    history = _make_results_history([1.5, 1.2, 0.8, 0.4, 0.1])
    current = history[-1]
    alerts = plugin.check(current, history, {"min_sharpe": 0.5, "window": 3})
    assert len(alerts) >= 1
    assert alerts[0]["rule"] == "signal_decay"


def test_signal_healthy():
    plugin = SignalDecayMonitor()
    history = _make_results_history([1.5, 1.4, 1.6, 1.5, 1.7])
    current = history[-1]
    alerts = plugin.check(current, history, {"min_sharpe": 0.5, "window": 3})
    assert len(alerts) == 0


def test_signal_decay_meta():
    assert SignalDecayMonitor().meta.name == "monitor.signal_decay.v1"
```

**Step 2: Run test, verify fails**

**Step 3: Implement**

`signal_decay.py`: Looks at rolling window of recent run Sharpe ratios from history. Computes mean Sharpe over window. Alert if below `min_sharpe`.

**Step 4: Run test, verify passes**

**Step 5: Commit**

```bash
git add packages/quantbox-core/src/quantbox/plugins/monitor/ tests/test_monitor_signal_decay.py
git commit -m "feat: add monitor.signal_decay.v1 plugin"
```

### Task 14: Register monitor plugins in builtins.py

**Files:**
- Modify: `packages/quantbox-core/src/quantbox/plugins/monitor/__init__.py`
- Modify: `packages/quantbox-core/src/quantbox/plugins/builtins.py`
- Test: `tests/test_monitor_discovery.py`

Similar pattern as Task 11. Register `DrawdownMonitor` and `SignalDecayMonitor`.

**Step 1-5:** Same TDD cycle.

```bash
git commit -m "feat: register monitor plugins in builtins"
```

---

## Phase 5: Enhanced Risk Plugins

### Task 15: Create risk.factor_exposure.v1 plugin

**Files:**
- Create: `packages/quantbox-core/src/quantbox/plugins/risk/factor_exposure.py`
- Test: `tests/test_risk_factor_exposure.py`

**Step 1: Write failing test**

```python
# tests/test_risk_factor_exposure.py
"""Tests for risk.factor_exposure.v1 plugin."""
import pandas as pd

from quantbox.plugins.risk.factor_exposure import FactorExposureRiskManager


def test_beta_check():
    plugin = FactorExposureRiskManager()
    targets = pd.DataFrame({"symbol": ["BTC", "ETH"], "weight": [0.7, 0.3]})
    findings = plugin.check_targets(targets, {"max_single_weight": 0.5})
    # BTC at 0.7 exceeds 0.5
    assert any(f["rule"] == "single_weight_exceeded" for f in findings)


def test_sector_concentration():
    plugin = FactorExposureRiskManager()
    targets = pd.DataFrame({
        "symbol": ["BTC", "ETH", "SOL", "AAPL"],
        "weight": [0.3, 0.3, 0.3, 0.1],
    })
    findings = plugin.check_targets(targets, {"max_sector_weight": 0.5, "sectors": {"crypto": ["BTC", "ETH", "SOL"]}})
    # Crypto sector at 0.9 exceeds 0.5
    assert any(f["rule"] == "sector_concentration_exceeded" for f in findings)


def test_no_findings_clean():
    plugin = FactorExposureRiskManager()
    targets = pd.DataFrame({"symbol": ["BTC", "ETH"], "weight": [0.5, 0.5]})
    findings = plugin.check_targets(targets, {"max_single_weight": 0.6})
    assert len(findings) == 0


def test_factor_exposure_meta():
    assert FactorExposureRiskManager().meta.name == "risk.factor_exposure.v1"
```

**Step 2: Run test, verify fails**

**Step 3: Implement**

`factor_exposure.py` implements the `RiskPlugin` protocol (same as existing risk plugins):
- `check_targets`: validates single-position weight, sector concentration, gross beta (when benchmark returns provided in params)
- `check_orders`: pass-through (delegates to trading_basic for order-level)

Params: `max_single_weight`, `max_sector_weight`, `sectors` (dict mapping sector name to symbol list)

**Step 4: Run test, verify passes**

**Step 5: Commit**

```bash
git add packages/quantbox-core/src/quantbox/plugins/risk/ tests/test_risk_factor_exposure.py
git commit -m "feat: add risk.factor_exposure.v1 plugin"
```

### Task 16: Create risk.drawdown_control.v1 plugin

**Files:**
- Create: `packages/quantbox-core/src/quantbox/plugins/risk/drawdown_control.py`
- Test: `tests/test_risk_drawdown_control.py`

**Step 1: Write failing test**

```python
# tests/test_risk_drawdown_control.py
"""Tests for risk.drawdown_control.v1 plugin."""
import pandas as pd

from quantbox.plugins.risk.drawdown_control import DrawdownControlRiskManager


def test_halt_on_drawdown():
    plugin = DrawdownControlRiskManager()
    targets = pd.DataFrame({"symbol": ["BTC", "ETH"], "weight": [0.5, 0.5]})
    findings = plugin.check_targets(targets, {
        "max_drawdown": -0.20,
        "current_drawdown": -0.25,
        "action": "halt",
    })
    assert any(f["level"] == "error" and f["rule"] == "drawdown_halt" for f in findings)


def test_scale_on_drawdown():
    plugin = DrawdownControlRiskManager()
    targets = pd.DataFrame({"symbol": ["BTC", "ETH"], "weight": [0.5, 0.5]})
    findings = plugin.check_targets(targets, {
        "max_drawdown": -0.20,
        "current_drawdown": -0.15,
        "scale_threshold": -0.10,
        "scale_factor": 0.5,
    })
    assert any(f["rule"] == "drawdown_scale" for f in findings)


def test_no_drawdown_ok():
    plugin = DrawdownControlRiskManager()
    targets = pd.DataFrame({"symbol": ["BTC", "ETH"], "weight": [0.5, 0.5]})
    findings = plugin.check_targets(targets, {"max_drawdown": -0.20, "current_drawdown": -0.05})
    assert len(findings) == 0


def test_drawdown_control_meta():
    assert DrawdownControlRiskManager().meta.name == "risk.drawdown_control.v1"
```

**Step 2-5:** Same TDD cycle.

```bash
git commit -m "feat: add risk.drawdown_control.v1 plugin"
```

### Task 17: Register new risk plugins and update manifest

**Files:**
- Modify: `packages/quantbox-core/src/quantbox/plugins/risk/__init__.py`
- Modify: `packages/quantbox-core/src/quantbox/plugins/builtins.py`
- Modify: `plugins/manifest.yaml`

Register `FactorExposureRiskManager` and `DrawdownControlRiskManager` in builtins and add all new plugins to the manifest.

```bash
git commit -m "feat: register new risk plugins, update manifest with all new plugin types"
```

---

## Phase 6: Pipeline Integration

### Task 18: Add validation support to backtest pipeline

**Files:**
- Modify: `packages/quantbox-core/src/quantbox/runner.py`
- Modify: `packages/quantbox-core/src/quantbox/plugins/pipeline/backtest_pipeline.py`
- Test: `tests/test_backtest_with_validation.py`

**Step 1: Write failing test**

```python
# tests/test_backtest_with_validation.py
"""Test that backtest pipeline runs validation plugins when configured."""
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta, RunResult


def test_runner_resolves_validation_plugins():
    """Validation plugins from config should be resolved by runner."""
    from quantbox.registry import PluginRegistry

    reg = PluginRegistry.discover()
    # If validation plugins are registered, runner should find them
    assert len(reg.validations) > 0
```

**Step 2: Run test, verify fails or passes (depending on Phase 3 completion)**

**Step 3: Modify runner.py**

In `run_from_config()`, after the main pipeline run, resolve and execute validation plugins:

```python
# After result = pipeline.run(...)
# --- Validation plugins (post-backtest) ---
validation_cfg = cfg["plugins"].get("validation", []) or []
if validation_cfg and mode == "backtest":
    validation_results = []
    for v_cfg in validation_cfg:
        v_cls = registry.validations[v_cfg["name"]]
        v_plugin = v_cls(**v_cfg.get("params_init", {}))
        # Load returns and weights from artifacts
        returns_df = pd.read_parquet(result.artifacts.get("returns", ""))
        weights_df = pd.read_parquet(result.artifacts.get("weights_history", ""))
        benchmark_df = None  # TODO: load from config if specified
        v_result = v_plugin.validate(returns_df, weights_df, benchmark_df, v_cfg.get("params", {}))
        validation_results.append({"plugin": v_cfg["name"], **v_result})
    store.put_json("validation", validation_results)
    result.artifacts["validation"] = store.get_path("validation")
    result.notes["validation"] = validation_results
```

**Step 4: Run test, verify passes**

**Step 5: Commit**

```bash
git add packages/quantbox-core/src/quantbox/runner.py tests/test_backtest_with_validation.py
git commit -m "feat: add validation plugin support to runner and backtest pipeline"
```

### Task 19: Add monitor support to trading pipeline

**Files:**
- Modify: `packages/quantbox-core/src/quantbox/runner.py`
- Test: `tests/test_trading_with_monitors.py`

Similar to Task 18 but for `mode in ("paper", "live")`. After pipeline run, resolve and execute monitor plugins. If any alert has `action: "halt"`, write `halt.json` to artifacts.

```bash
git commit -m "feat: add monitor plugin support to runner for paper/live modes"
```

---

## Phase 7: Data Expansion (quantbox-datasets)

### Task 20: Add equities data module to quantbox-datasets

**Files:**
- Create: `src/quantbox_datasets/equities/__init__.py`
- Create: `src/quantbox_datasets/equities/us_stocks.py`
- Create: `src/quantbox_datasets/equities/etf_universe.py`
- Test: `tests/test_equities.py`

**Working directory:** `/home/tom/workspace/projects/quantbox-datasets`

**Step 1: Write failing test**

```python
# tests/test_equities.py
"""Tests for equities data module."""
import pandas as pd
import pytest

from quantbox_datasets.equities.us_stocks import fetch_sp500_prices
from quantbox_datasets.equities.etf_universe import ETF_CORE, fetch_etf_prices


def test_sp500_returns_dataframe():
    # Fetch a small lookback to keep test fast
    df = fetch_sp500_prices(lookback_days=30, top_n=5)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] >= 1  # at least 1 stock
    assert isinstance(df.index, pd.DatetimeIndex)


def test_etf_universe_defined():
    assert "SPY" in ETF_CORE
    assert "QQQ" in ETF_CORE
    assert len(ETF_CORE) >= 5


def test_etf_prices():
    df = fetch_etf_prices(symbols=["SPY", "QQQ"], lookback_days=30)
    assert isinstance(df, pd.DataFrame)
    assert "SPY" in df.columns
```

Note: These tests require network access. Mark with `@pytest.mark.network` if CI should skip them.

**Step 2: Run test, verify fails**

**Step 3: Implement**

`us_stocks.py`:
- `fetch_sp500_prices(lookback_days, top_n)`: uses yfinance to download S&P 500 constituent prices. Gets constituent list from Wikipedia or hardcoded top-N. Returns wide DataFrame.
- Caches to `data/curated/equities/sp500_prices.parquet`

`etf_universe.py`:
- `ETF_CORE`: list of core ETFs (SPY, QQQ, IWM, EFA, EEM, TLT, GLD, USO, XLF, XLK, XLV, XLE)
- `fetch_etf_prices(symbols, lookback_days)`: downloads via yfinance, returns wide DataFrame
- `fetch_etf_volume(symbols, lookback_days)`: same for volume

**Step 4: Run test, verify passes**

**Step 5: Commit**

```bash
cd /home/tom/workspace/projects/quantbox-datasets
git add src/quantbox_datasets/equities/ tests/test_equities.py
git commit -m "feat: add equities data module (S&P 500, ETF universe)"
```

### Task 21: Add macro data module to quantbox-datasets

**Files:**
- Create: `src/quantbox_datasets/macro/__init__.py`
- Create: `src/quantbox_datasets/macro/fred.py`
- Create: `src/quantbox_datasets/macro/indicators.py`
- Test: `tests/test_macro.py`

**Step 1: Write failing test**

```python
# tests/test_macro.py
import pandas as pd
from quantbox_datasets.macro.indicators import compute_yield_curve_slope, MACRO_SERIES


def test_macro_series_defined():
    assert "GDP" in MACRO_SERIES
    assert "CPI" in MACRO_SERIES
    assert "UNRATE" in MACRO_SERIES


def test_yield_curve_slope():
    # Synthetic test with mock data
    tens = pd.Series([3.5, 3.6, 3.4], index=pd.date_range("2025-01-01", periods=3))
    twos = pd.Series([4.0, 3.9, 3.8], index=pd.date_range("2025-01-01", periods=3))
    slope = compute_yield_curve_slope(tens, twos)
    assert isinstance(slope, pd.Series)
    assert slope.iloc[0] == pytest.approx(-0.5)
```

**Step 2-5:** TDD cycle.

`fred.py`: Fetches FRED series via `pandas_datareader.data.DataReader(series, 'fred')`. Key series: GDP, CPI, UNRATE, DGS10, DGS2, VIXCLS, BAMLH0A0HYM2.

`indicators.py`: Derived indicators from raw FRED data:
- `compute_yield_curve_slope(tens, twos)`: DGS10 - DGS2
- `compute_real_rate(nominal, cpi)`: nominal - YoY CPI
- `compute_credit_spread(hy_oas)`: high yield OAS

```bash
git commit -m "feat: add macro data module (FRED, derived indicators)"
```

### Task 22: Add fundamental data module to quantbox-datasets

**Files:**
- Create: `src/quantbox_datasets/fundamental/__init__.py`
- Create: `src/quantbox_datasets/fundamental/yfinance_fundamentals.py`
- Test: `tests/test_fundamentals.py`

`yfinance_fundamentals.py`: Fetches quarterly fundamentals from yfinance for a list of symbols. Returns DataFrame with columns: `date`, `symbol`, `pe_ratio`, `pb_ratio`, `eps`, `revenue`, `market_cap`.

```bash
git commit -m "feat: add fundamental data module (yfinance quarterly)"
```

---

## Phase 8: Research Lab (quantbox-lab)

### Task 23: Create research runner script

**Files:**
- Create: `scripts/run_research.py`

**Working directory:** `/home/tom/workspace/projects/quantbox-lab`

**Implementation:**

```python
#!/usr/bin/env python
"""Run a full research pipeline: backtest + validation + report.

Usage:
    python scripts/run_research.py -c configs/research/crypto_trend_validation.yaml
"""
```

The script:
1. Loads research YAML config
2. Extracts the `backtest` section → runs `quantbox run` via subprocess or direct API
3. Reads backtest artifacts (returns, weights)
4. Instantiates each validation plugin from config
5. Runs validation, collects results
6. Calls `generate_report.py` to produce markdown report
7. Saves everything to `artifacts/<run_id>/`

```bash
git commit -m "feat: add research runner script"
```

### Task 24: Create report generator

**Files:**
- Create: `scripts/generate_report.py`

Reads artifacts directory and produces a professional markdown report with:
- Performance summary table (IS vs OOS if available)
- Statistical validation results
- Turnover analysis
- Regime breakdown table
- Factor attribution (if benchmark available)
- Config used

Output: `reports/<run_id>_report.md`

```bash
git commit -m "feat: add research report generator"
```

### Task 25: Create strategy comparison script

**Files:**
- Create: `scripts/run_comparison.py`

Runs N strategies through the same backtest + validation pipeline, then produces a side-by-side comparison report.

```bash
git commit -m "feat: add strategy comparison script"
```

### Task 26: Create research YAML configs

**Files:**
- Create: `configs/research/crypto_trend_validation.yaml`
- Create: `configs/research/ml_prediction_walkforward.yaml`
- Create: `configs/research/cross_asset_momentum_study.yaml`
- Create: `configs/research/strategy_comparison.yaml`
- Create: `configs/data/fetch_crypto.yaml`
- Create: `configs/data/fetch_equities.yaml`

Each config uses the new validation plugins. Example `crypto_trend_validation.yaml`:

```yaml
research:
  name: "crypto_trend_full_validation"

  backtest:
    config:
      run:
        mode: backtest
        asof: "2026-02-19"
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
              lookback_days: 1095
        strategies:
          - name: "strategy.crypto_trend.v1"
            weight: 1.0
            params:
              lookback_days: 365
        data:
          name: "binance.live_data.v1"
          params_init:
            quote_asset: USDT

  validation:
    - name: "validation.walk_forward.v1"
      params:
        n_splits: 5
        train_ratio: 0.7
    - name: "validation.statistical.v1"
      params:
        n_trials: 100
        confidence: 0.95
        n_strategies_tested: 1
    - name: "validation.turnover.v1"
      params:
        cost_bps: 10
    - name: "validation.regime.v1"
      params:
        window: 60
    - name: "validation.benchmark.v1"
      params: {}
```

```bash
git commit -m "feat: add research YAML configs for all pipeline types"
```

### Task 27: Create data fetch script and configs

**Files:**
- Create: `scripts/fetch_data.py`
- Create: `configs/data/fetch_macro.yaml`

`fetch_data.py` reads a data config and calls quantbox-datasets to fetch and cache data locally.

```bash
git commit -m "feat: add data fetch orchestrator script"
```

### Task 28: Update quantbox-lab README and pyproject.toml

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml`

Add quantbox-datasets as dependency. Update README with new research workflow.

```bash
git commit -m "docs: update README with full research pipeline workflow"
```

---

## Phase 9: Final Integration

### Task 29: Run full test suite and fix any issues

Run: `cd /home/tom/workspace/projects/quantbox && uv run pytest -q`

Fix any integration issues, import errors, or test failures.

```bash
git commit -m "fix: resolve integration issues from full test suite"
```

### Task 30: Run a full research pipeline end-to-end in quantbox-lab

```bash
cd /home/tom/workspace/projects/quantbox-lab
python scripts/run_research.py -c configs/research/crypto_trend_validation.yaml
```

Verify:
- Backtest runs successfully
- All 5 validation plugins produce results
- Report is generated in `reports/`
- No errors

```bash
git commit -m "test: verify end-to-end research pipeline"
```

---

## Dependency Order

```
Task 1 (protocols) → Task 2 (registry)
Task 2 → Tasks 3-5 (features) [parallel]
Task 2 → Tasks 6-11 (validation) [parallel with features]
Task 2 → Tasks 12-14 (monitors) [parallel with above]
Task 2 → Tasks 15-17 (risk) [parallel with above]
Tasks 5, 11, 14, 17 → Task 18-19 (pipeline integration)
Tasks 18-19 → Tasks 20-22 (datasets) [can start earlier, no core dep]
Tasks 18-22 → Tasks 23-28 (lab)
All → Task 29-30 (integration)
```

Tasks 3-17 can be parallelized across subagents since they create independent plugin files.
