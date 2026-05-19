"""Tests for the report-block registry (``plugins/pipeline/blocks.py``).

Covers:
- registry shape (every default block has the required metadata)
- registration roundtrip (``register_block``)
- generic-block builders return plausible figure dicts from realistic inputs
- generic-block builders return None / are skipped when required inputs are missing
- the legacy ``_DIAGNOSTIC_BUILDERS`` alias remains a non-empty dict-of-callables
  (back-compat for any external code that imports it)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.pipeline.blocks import (
    BLOCKS,
    ReportBlock,
    diagnostic_block_names,
    get_block,
    register_block,
)


@pytest.fixture
def sample_returns() -> pd.Series:
    rng = np.random.RandomState(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="D")
    return pd.Series(rng.normal(0.0005, 0.02, 500), index=idx, name="ret")


@pytest.fixture
def sample_weights() -> pd.DataFrame:
    rng = np.random.RandomState(7)
    idx = pd.date_range("2024-01-01", periods=500, freq="D")
    # A weights history with daily churn so turnover is non-zero
    raw = rng.dirichlet(np.ones(5), size=500)
    return pd.DataFrame(raw, index=idx, columns=["BTC", "ETH", "SOL", "BNB", "XRP"])


class TestRegistryShape:
    def test_default_blocks_have_required_metadata(self) -> None:
        for name, block in BLOCKS.items():
            assert isinstance(block, ReportBlock)
            assert block.name == name, f"{name} disagrees with registry key"
            assert block.title, f"{name} missing title"
            assert block.description, f"{name} missing description"
            assert block.builder is not None, f"{name} missing builder"
            assert block.section in {"framework", "diagnostics", "comparison", "appendix"}

    def test_diagnostic_names_only_returns_diagnostic_section(self) -> None:
        names = diagnostic_block_names()
        for n in names:
            assert BLOCKS[n].section == "diagnostics"
        # Generic blocks (section='framework') should NOT appear.
        assert "return_distribution" not in names
        assert "rolling_metrics" not in names
        assert "turnover_timeline" not in names

    def test_get_block_returns_none_for_unknown(self) -> None:
        assert get_block("does_not_exist") is None
        assert get_block("regime_overlay") is not None


class TestRegistration:
    def test_register_and_lookup(self) -> None:
        def _builder(_payload, **_ctx):
            return {"data": [], "layout": {}}

        b = ReportBlock(
            name="__test_block__",
            title="Test block",
            description="ephemeral test entry",
            builder=_builder,
            section="diagnostics",
            tags=("test",),
        )
        register_block(b)
        try:
            assert get_block("__test_block__") is b
            assert "__test_block__" in diagnostic_block_names()
        finally:
            BLOCKS.pop("__test_block__", None)


class TestReturnDistributionBlock:
    def test_renders_valid_figure(self, sample_returns: pd.Series) -> None:
        block = BLOCKS["return_distribution"]
        fig = block.builder(None, returns=sample_returns)
        assert fig is not None
        assert "data" in fig and "layout" in fig
        # Histogram + Normal overlay = 2 traces
        assert len(fig["data"]) >= 1
        # Stat badge appears in title text
        title_text = fig["layout"]["title"]["text"]
        for fragment in ("mean=", "std=", "skew=", "exc-kurt="):
            assert fragment in title_text

    def test_empty_returns_yields_none(self) -> None:
        block = BLOCKS["return_distribution"]
        assert block.builder(None, returns=pd.Series([], dtype=float)) is None


class TestRollingMetricsBlock:
    def test_renders_with_enough_data(self, sample_returns: pd.Series) -> None:
        block = BLOCKS["rolling_metrics"]
        fig = block.builder(None, returns=sample_returns)
        assert fig is not None
        assert "data" in fig
        # Two-panel: Sharpe + vol traces
        assert len(fig["data"]) == 2

    def test_short_returns_skips(self) -> None:
        block = BLOCKS["rolling_metrics"]
        short = pd.Series(np.zeros(30), index=pd.date_range("2024-01-01", periods=30))
        assert block.builder(None, returns=short) is None


class TestTurnoverTimelineBlock:
    def test_renders_with_dynamic_weights(self, sample_weights: pd.DataFrame) -> None:
        block = BLOCKS["turnover_timeline"]
        fig = block.builder(None, weights_history=sample_weights)
        assert fig is not None
        # Daily + cumulative = 2 traces
        assert len(fig["data"]) == 2

    def test_constant_weights_skips(self) -> None:
        block = BLOCKS["turnover_timeline"]
        idx = pd.date_range("2024-01-01", periods=100, freq="D")
        constant = pd.DataFrame(0.5, index=idx, columns=["A", "B"])
        # No |Δ| changes → turnover sum is zero → block returns None
        assert block.builder(None, weights_history=constant) is None


class TestLegacyAlias:
    def test_diagnostic_builders_dict_still_exists(self) -> None:
        # External callers may still import the legacy name.
        from quantbox.plugins.pipeline._report import _DIAGNOSTIC_BUILDERS

        assert isinstance(_DIAGNOSTIC_BUILDERS, dict)
        assert len(_DIAGNOSTIC_BUILDERS) > 0
        for name, builder in _DIAGNOSTIC_BUILDERS.items():
            assert callable(builder), f"{name} builder is not callable"
            assert name in BLOCKS, f"legacy alias has stray entry: {name}"
