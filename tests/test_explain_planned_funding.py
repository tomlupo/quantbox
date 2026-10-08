"""explain sees a perp dataset with no funding file when its data plugin plans no file (TOM-1627).

``dataset.curated.v1`` plans no files (no ``planned_paths``), so ``quantbox config explain``
could not see a perp dataset without a funding series on rsims: only the run refused it,
after load. A data plugin may now answer ``planned_funding()`` — does the dataset it will
serve carry a funding series? — before any data is read. ``check_planned_data`` reads it
(:func:`quantbox.funding_guard.planned_funding`), so explain, validate and the run refuse
the same thing at the same point. How the prices file is checked and hashed does not change.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from quantbox.contracts import PluginMeta
from quantbox.exceptions import ConfigValidationError
from quantbox.explain import explain_config
from quantbox.plugins.pipeline.backtest_pipeline import BacktestPipeline
from quantbox.registry import PluginRegistry
from quantbox.store import FileArtifactStore

MISSING = "funding_missing"
IGNORE = {"ignore": True, "reason": "perp book studied without funding: the carry is measured in TOM-0000"}


def _prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=60, freq="D", name="date")
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {s: 100.0 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))) for s in ("AAA", "BBB")},
        index=idx,
    )


@dataclass
class _Curated:
    """A data plugin that plans no file (like dataset.curated.v1) and answers its market and funding."""

    meta = PluginMeta(
        name="fake.curated_planned_funding.v1",
        kind="data",
        version="0.0.1",
        core_compat=">=0.1.0",
        description="Plans no file; answers planned_market() and planned_funding().",
        params_schema={
            "type": "object",
            "properties": {
                "market": {"description": "What planned_market() answers."},
                "with_funding": {"description": "What planned_funding() answers, and whether a series comes back."},
            },
        },
    )
    market: str = "futures"
    with_funding: bool | None = False
    loaded: bool = False

    def planned_market(self) -> str | None:
        return self.market

    def planned_funding(self) -> bool | None:
        return self.with_funding

    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame({"symbol": ["AAA", "BBB"]})

    def load_market_data(self, universe: Any, asof: str, params: dict[str, Any]) -> dict[str, pd.DataFrame]:
        self.loaded = True
        out = {"prices": _prices()}
        if self.with_funding:
            out["funding_rates"] = _prices() * 0 + 0.0001
        return out

    def load_fx(self, asof: str, params: dict[str, Any]) -> None:
        return None


def _explain(tmp_path, engine: str, *, with_funding: bool | None, ignore: bool = False) -> dict:
    registry = PluginRegistry.discover()
    registry.data[_Curated.meta.name] = _Curated
    answer = "null" if with_funding is None else str(with_funding).lower()
    cfg = yaml.safe_load(f"""
run: {{mode: backtest, asof: "2024-02-20", pipeline: backtest.pipeline.v1}}
artifacts: {{root: "{tmp_path / "artifacts"}"}}
plugins:
  pipeline:
    name: backtest.pipeline.v1
    params: {{engine: {engine}, fees: 0.0}}
  strategies:
    - name: strategy.static_weights.v1
      weight: 1.0
      params_init: {{weights: {{AAA: 1.0}}}}
  data:
    name: {_Curated.meta.name}
    params_init: {{market: futures, with_funding: {answer}}}
""")
    if ignore:
        cfg["plugins"]["pipeline"]["params"]["funding"] = dict(IGNORE)
    return explain_config(copy.deepcopy(cfg), registry)


def test_explain_refuses_a_perp_dataset_with_no_funding_on_rsims_when_the_plugin_plans_no_file(tmp_path):
    doc = _explain(tmp_path, "rsims", with_funding=False)
    assert doc["ok"] is False
    assert any(MISSING in e for e in doc["errors"]), doc["errors"]


def test_explain_plans_a_perp_dataset_with_funding_on_rsims_when_the_plugin_plans_no_file(tmp_path):
    doc = _explain(tmp_path, "rsims", with_funding=True)
    assert doc["ok"] is True, doc["errors"]
    assert doc["funding"]["modelled"] is True


def test_explain_takes_the_declared_ignore_for_a_perp_dataset_with_no_funding_on_rsims(tmp_path):
    doc = _explain(tmp_path, "rsims", with_funding=False, ignore=True)
    assert doc["ok"] is True, doc["errors"]


def test_explain_refuses_an_ignore_when_the_plugin_plans_funding_on_rsims(tmp_path):
    doc = _explain(tmp_path, "rsims", with_funding=True, ignore=True)
    assert doc["ok"] is False
    assert any("charges funding" in e for e in doc["errors"]), doc["errors"]


def test_a_plugin_that_cannot_say_still_defers_to_the_frames(tmp_path):
    """planned_funding() None (a live source): explain cannot know, so it does not refuse; the run does."""
    doc = _explain(tmp_path, "rsims", with_funding=None)
    assert doc["ok"] is True, doc["errors"]


def test_the_run_refuses_a_planned_perp_dataset_with_no_funding_before_any_data_is_read(tmp_path):
    data = _Curated(with_funding=False)
    with pytest.raises(ConfigValidationError, match=MISSING):
        BacktestPipeline().run(
            mode="backtest",
            asof="2024-02-20",
            params={"engine": "rsims", "fees": 0.0, "strategies": [{"name": "s", "weight": 1.0}]},
            data=data,
            store=FileArtifactStore(str(tmp_path), "run"),
            broker=None,
            risk=[],
            strategies=[],
        )
    assert data.loaded is False
