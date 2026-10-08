"""The funding guard's two gaps (TOM-1619): a perp market with no funding series, and the helper doors.

TOM-1609 refused a funding SERIES on an engine that does not charge it. Two holes stayed:

1. A perp dataset whose data plugin hands back no funding series (or plans no file):
   the guard saw nothing. Now the dataset's ``market`` (its manifest, through the data
   plugin's ``planned_market()``) reaches the run, explain and validate. A perp market
   on an engine that does not charge funding is refused; a perp market with no funding
   series is refused on an engine that does (it would charge zero). The one escape is
   the same ``funding: {ignore: true, reason}``, recorded in run@1.
2. ``backtest()``, ``optimize()`` and the sweep called the seam without the guard. The
   check now lives in the seam (:func:`quantbox.engine.simulate`); each door accepts the
   funding series and the escape, and forwards both.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from test_dataset_resolve import _build, fake_datasets, root  # noqa: F401 — fixtures

from quantbox.contracts import PluginMeta
from quantbox.engine import simulate
from quantbox.exceptions import ConfigValidationError
from quantbox.execution import resolve_execution
from quantbox.explain import explain_config, validate_explain
from quantbox.plugins.backtesting import backtest, optimize
from quantbox.plugins.pipeline.backtest_pipeline import BacktestPipeline
from quantbox.registry import PluginRegistry
from quantbox.run_manifest import validate_run_manifest
from quantbox.runner import run_from_config
from quantbox.store import FileArtifactStore
from quantbox.sweep import sweep
from quantbox.validate import validate_config

NOT_CHARGED = "funding_not_charged"
MISSING = "funding_missing"
REASON = "perp book studied without funding: the carry is measured separately in TOM-0000"
IGNORE = {"ignore": True, "reason": REASON}


# =========================================================================== 1. the market


def _perp_config(tmp_path: Path, ds_root: Path, engine: str, *, market: str = "perp", funding: bool = False):
    """A by-name dataset (local_file_data) whose manifest declares *market*; no funding file by default."""
    sha = _build(ds_root, "perp-daily", market=market, funding=funding)
    lab = tmp_path / "lab"
    lab.mkdir()
    (lab / "datasets.lock").write_text(yaml.safe_dump({"perp-daily": sha}))
    cfg = yaml.safe_load(f"""
run: {{mode: backtest, asof: "2024-02-20", pipeline: backtest.pipeline.v1}}
artifacts: {{root: "{tmp_path / "artifacts"}"}}
plugins:
  pipeline:
    name: backtest.pipeline.v1
    params:
      engine: {engine}
      fees: 0.0
      universe: {{symbols: [AAA, BBB]}}
  strategies:
    - name: strategy.static_weights.v1
      weight: 1.0
      params_init: {{weights: {{AAA: 0.5, BBB: 0.5}}}}
  data:
    name: local_file_data
    params_init: {{dataset: perp-daily}}
""")
    config_path = lab / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return cfg, config_path


def _ignore(cfg: dict) -> dict:
    cfg["plugins"]["pipeline"]["params"]["funding"] = dict(IGNORE)
    return cfg


def _errors(findings) -> list[str]:
    return [f.message for f in findings if f.level == "error"]


@pytest.mark.parametrize("market", ["perp", "futures"])
def test_a_perp_market_on_vectorbt_is_refused_by_explain(tmp_path, root, fake_datasets, market):  # noqa: F811
    cfg, config_path = _perp_config(tmp_path, root, "vectorbt", market=market)
    doc = explain_config(cfg, PluginRegistry.discover(), config_path=config_path)

    assert doc["ok"] is False
    assert any(NOT_CHARGED in e and f"market: {market}" in e for e in doc["errors"]), doc["errors"]
    assert validate_explain(doc) == []


def test_a_perp_market_on_vectorbt_is_refused_by_validate(tmp_path, root, fake_datasets):  # noqa: F811
    cfg, config_path = _perp_config(tmp_path, root, "vectorbt")
    errors = _errors(validate_config(cfg, config_path=config_path))
    assert any(NOT_CHARGED in m for m in errors), errors


def test_a_perp_market_on_vectorbt_is_refused_by_the_run_before_any_data_is_read(tmp_path, root, fake_datasets):  # noqa: F811
    cfg, config_path = _perp_config(tmp_path, root, "vectorbt")
    with pytest.raises(ConfigValidationError, match=NOT_CHARGED) as exc:
        run_from_config(cfg, PluginRegistry.discover(), config_path=config_path)
    assert [f.code for f in exc.value.findings] == [NOT_CHARGED]
    assert fake_datasets == []  # refused before the dataset was loaded


def test_a_perp_market_on_vectorbt_runs_with_the_declared_ignore_and_records_it(tmp_path, root, fake_datasets):  # noqa: F811
    cfg, config_path = _perp_config(tmp_path, root, "vectorbt")
    _ignore(cfg)
    registry = PluginRegistry.discover()

    planned = explain_config(copy.deepcopy(cfg), registry, config_path=config_path)
    assert planned["ok"] is True, planned["errors"]
    assert planned["dataset"]["market"] == "perp"
    assert planned["funding"]["ignored_reason"] == REASON
    assert not _errors(validate_config(copy.deepcopy(cfg), config_path=config_path))

    result = run_from_config(copy.deepcopy(cfg), registry, config_path=config_path)
    manifest = json.loads((tmp_path / "artifacts" / result.run_id / "run_manifest.json").read_text())
    assert manifest["funding"]["modelled"] is False
    assert manifest["funding"]["ignored_reason"] == REASON
    assert validate_run_manifest(manifest) == []


def test_a_perp_market_with_no_funding_file_is_refused_on_rsims(tmp_path, root, fake_datasets):  # noqa: F811
    cfg, config_path = _perp_config(tmp_path, root, "rsims")
    doc = explain_config(copy.deepcopy(cfg), PluginRegistry.discover(), config_path=config_path)
    assert doc["ok"] is False
    assert any(MISSING in e for e in doc["errors"]), doc["errors"]
    assert any(MISSING in m for m in _errors(validate_config(copy.deepcopy(cfg), config_path=config_path)))
    with pytest.raises(ConfigValidationError, match=MISSING):
        run_from_config(copy.deepcopy(cfg), PluginRegistry.discover(), config_path=config_path)


def test_a_perp_market_with_no_funding_file_runs_on_rsims_with_the_declared_ignore(tmp_path, root, fake_datasets):  # noqa: F811
    cfg, config_path = _perp_config(tmp_path, root, "rsims")
    _ignore(cfg)
    result = run_from_config(cfg, PluginRegistry.discover(), config_path=config_path)
    manifest = json.loads((tmp_path / "artifacts" / result.run_id / "run_manifest.json").read_text())
    assert manifest["funding"]["modelled"] is False
    assert manifest["funding"]["ignored_reason"] == REASON


def test_a_perp_market_with_funding_on_rsims_runs_and_refuses_an_ignore(tmp_path, root, fake_datasets):  # noqa: F811
    cfg, config_path = _perp_config(tmp_path, root, "rsims", funding=True)
    result = run_from_config(copy.deepcopy(cfg), PluginRegistry.discover(), config_path=config_path)
    manifest = json.loads((tmp_path / "artifacts" / result.run_id / "run_manifest.json").read_text())
    assert manifest["funding"]["modelled"] is True

    errors = _errors(validate_config(_ignore(copy.deepcopy(cfg)), config_path=config_path))
    assert any("charges funding" in m for m in errors), errors


@pytest.mark.parametrize("engine", ["vectorbt", "rsims"])
def test_a_spot_market_without_funding_is_planned_on_both_engines(tmp_path, root, fake_datasets, engine):  # noqa: F811
    cfg, config_path = _perp_config(tmp_path, root, engine, market="spot")
    doc = explain_config(cfg, PluginRegistry.discover(), config_path=config_path)
    assert doc["ok"] is True, doc["errors"]
    assert doc["dataset"]["market"] == "spot"


# A data plugin that plans no file but names its market (dataset.curated.v1, quantbox-datasets):
# the market is read before any data; the funding series once the frames come back.


def _prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {s: 100.0 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))) for s in ("AAA", "BBB")},
        index=pd.DatetimeIndex(idx.values),
    )


@dataclass
class _CuratedLike:
    meta = PluginMeta(
        name="fake.curated_perp.v1",
        kind="data",
        version="0.0.1",
        core_compat=">=0.1.0",
        description="A data plugin with no planned files that names its market (like dataset.curated.v1).",
        params_schema={
            "type": "object",
            "properties": {
                "market": {"description": "The market planned_market() answers."},
                "with_funding": {"description": "Hand back a funding series."},
            },
        },
    )
    market: str = "futures"
    with_funding: bool = False
    loaded: bool = False

    def planned_market(self) -> str | None:
        return self.market

    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame({"symbol": ["AAA", "BBB"]})

    def load_market_data(self, universe: Any, asof: str, params: dict[str, Any]) -> dict[str, pd.DataFrame]:
        self.loaded = True
        prices = _prices()
        out = {"prices": prices}
        if self.with_funding:
            out["funding_rates"] = prices * 0 + 0.0001
        return out

    def load_fx(self, asof: str, params: dict[str, Any]) -> None:
        return None


class _Hold:
    meta = type("M", (), {"name": "strategy.hold.v1"})()
    ran = False

    def run(self, data: Any, params: Any = None) -> dict[str, Any]:
        _Hold.ran = True
        return {"weights": pd.DataFrame({"AAA": 0.5, "BBB": 0.5}, index=_prices().index)}


def _run_curated(tmp_path, data, **params):
    _Hold.ran = False
    return BacktestPipeline().run(
        mode="backtest",
        asof="2024-02-09",
        params={"fees": 0.0, "strategies": [{"name": "strategy.hold.v1", "weight": 1.0}], **params},
        data=data,
        store=FileArtifactStore(str(tmp_path), "run"),
        broker=None,
        risk=[],
        strategies=[_Hold()],
    )


def test_a_plugin_named_perp_market_on_vectorbt_is_refused_before_any_data_is_read(tmp_path):
    data = _CuratedLike()
    with pytest.raises(ConfigValidationError, match=NOT_CHARGED):
        _run_curated(tmp_path, data, engine="vectorbt")
    assert data.loaded is False


def test_a_plugin_named_perp_market_without_funding_is_refused_on_rsims_once_the_frames_come_back(tmp_path):
    with pytest.raises(ConfigValidationError, match=MISSING):
        _run_curated(tmp_path, _CuratedLike(), engine="rsims")
    assert _Hold.ran is False  # refused before any strategy ran


def test_a_plugin_named_perp_market_with_funding_runs_on_rsims(tmp_path):
    result = _run_curated(tmp_path, _CuratedLike(with_funding=True), engine="rsims")
    assert result.notes["funding"]["modelled"] is True


def test_explain_refuses_a_plugin_named_perp_market_on_vectorbt(tmp_path):
    registry = PluginRegistry.discover()
    registry.data[_CuratedLike.meta.name] = _CuratedLike
    cfg = yaml.safe_load(f"""
run: {{mode: backtest, asof: "2024-02-09", pipeline: backtest.pipeline.v1}}
artifacts: {{root: "{tmp_path / "artifacts"}"}}
plugins:
  pipeline:
    name: backtest.pipeline.v1
    params: {{engine: vectorbt, fees: 0.0}}
  strategies:
    - name: strategy.static_weights.v1
      weight: 1.0
      params_init: {{weights: {{AAA: 1.0}}}}
  data:
    name: {_CuratedLike.meta.name}
    params_init: {{market: futures}}
""")
    doc = explain_config(copy.deepcopy(cfg), registry)
    assert doc["ok"] is False
    assert any(NOT_CHARGED in e for e in doc["errors"]), doc["errors"]

    cfg["plugins"]["pipeline"]["params"]["funding"] = dict(IGNORE)
    doc = explain_config(cfg, registry)
    assert doc["ok"] is True, doc["errors"]
    assert doc["dataset"]["market"] == "futures"


# =========================================================================== 2. the doors


def _weights() -> pd.DataFrame:
    return pd.DataFrame({"AAA": 0.5, "BBB": 0.5}, index=_prices().index)


def _funding() -> pd.DataFrame:
    return _prices() * 0 + 0.0001


def test_the_seam_refuses_funding_on_vectorbt():
    with pytest.raises(ConfigValidationError, match=NOT_CHARGED):
        simulate(_prices(), _weights(), engine="vectorbt", timing=resolve_execution(None), funding=_funding())


def test_the_seam_takes_the_declared_ignore_and_charges_funding_on_rsims():
    book = simulate(
        _prices(),
        _weights(),
        engine="vectorbt",
        timing=resolve_execution(None),
        funding=_funding(),
        funding_ignore=IGNORE,
    )
    assert book.funding_modelled is False
    book = simulate(_prices(), _weights(), engine="rsims", timing=resolve_execution(None), funding=_funding())
    assert book.funding_modelled is True


def test_the_seam_refuses_an_ignore_with_funding_on_rsims():
    with pytest.raises(ConfigValidationError, match="charges funding"):
        simulate(
            _prices(),
            _weights(),
            engine="rsims",
            timing=resolve_execution(None),
            funding=_funding(),
            funding_ignore=IGNORE,
        )


def test_backtest_refuses_funding_on_vectorbt():
    with pytest.raises(ConfigValidationError, match=NOT_CHARGED):
        backtest(_prices(), _weights(), engine="vectorbt", funding_rates=_funding())


def test_backtest_forwards_funding_and_the_ignore():
    assert backtest(_prices(), _weights(), engine="vectorbt", funding_rates=_funding(), funding=IGNORE)["metrics"]
    assert backtest(_prices(), _weights(), engine="rsims", funding_rates=_funding())["book"].funding_modelled is True


def _weights_fn(prices: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(params["w"], index=prices.index, columns=prices.columns)


@pytest.mark.parametrize("method", ["grid", "walk_forward"])
def test_optimize_refuses_funding_on_vectorbt(method):
    # Inside the search a failing combination is skipped; the guard must refuse the CALL, not each combination.
    with pytest.raises(ConfigValidationError, match=NOT_CHARGED):
        optimize(
            _prices(),
            _weights_fn,
            {"w": [0.5]},
            method=method,
            engine="vectorbt",
            funding_rates=_funding(),
            train_size=20,
            test_size=10,
        )


def test_optimize_forwards_funding_and_the_ignore():
    out = optimize(_prices(), _weights_fn, {"w": [0.5]}, engine="vectorbt", funding_rates=_funding(), funding=IGNORE)
    assert not out["all_results"].empty
    out = optimize(_prices(), _weights_fn, {"w": [0.5]}, engine="rsims", funding_rates=_funding())
    assert not out["all_results"].empty


class _SweepHold:
    def __init__(self, **params: Any) -> None:
        self.params = params

    def run(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        prices = data["prices"]
        return {"weights": pd.DataFrame(self.params["w"], index=prices.index, columns=prices.columns)}


def test_the_sweep_refuses_funding_on_vectorbt():
    data = {"prices": _prices(), "funding_rates": _funding()}
    with pytest.raises(ConfigValidationError, match=NOT_CHARGED):
        sweep(_SweepHold, {}, {"w": [0.5]}, data, backtest_kwargs={"engine": "vectorbt", "fees": 0.0})


def test_the_sweep_forwards_funding_and_the_ignore():
    data = {"prices": _prices(), "funding_rates": _funding()}
    grid = sweep(
        _SweepHold, {}, {"w": [0.5]}, data, backtest_kwargs={"engine": "vectorbt", "fees": 0.0}, funding=IGNORE
    )
    assert len(grid) == 1
    grid = sweep(_SweepHold, {}, {"w": [0.5]}, data, backtest_kwargs={"engine": "rsims", "fees": 0.0})
    assert len(grid) == 1
