"""Arms as data (TOM-1363): one base config + named overrides or a grid.

The contract under test:

- an arms file reproduces the separate runs it replaces, metric for metric;
- arms run in parallel within ``max_workers`` and a memory budget;
- a failing arm fails the batch WITH ITS NAME, and the other arms' results stay;
- the batch summary records n_trials and links every arm's run@1 manifest;
- a ``source: file.py:Class`` strategy runs as an arm, in a sweep and in a variant;
- arms and sweep share ONE timing setting (``execution: {lag_bars}``), recorded
  in every manifest.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from quantbox.arms import ARMS_SCHEMA_ID, expand_arms, load_arms, plan_workers, run_arms
from quantbox.cli import app
from quantbox.registry import PluginRegistry
from quantbox.run_manifest import json_safe
from quantbox.runner import run_from_config

# A local-source strategy: holds `frac` of A while A's last return was positive.
# Different `frac` values give different books, so arms are distinguishable.
_SOURCE_STRATEGY = """
from dataclasses import dataclass

import pandas as pd

from quantbox.contracts import PluginMeta


@dataclass
class FracMomentum:
    meta = PluginMeta(
        name="lab.strategy.frac_momentum.v1",
        kind="strategy",
        version="0.0.1",
        core_compat=">=0.1",
        status="research",
        description="Test-only local-source strategy.",
    )

    frac: float = 1.0

    def run(self, data, params=None):
        for key, value in (params or {}).items():
            setattr(self, key, value)
        if not isinstance(self.frac, (int, float)):
            raise TypeError(f"frac must be a number, got {self.frac!r}")
        prices = data["prices"]
        up = (prices["A"].pct_change() > 0).astype(float)
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        weights["A"] = up * float(self.frac)
        return {"weights": weights}
"""

FRACS = (0.2, 0.4, 0.6, 0.8, 1.0)


def _prices(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(11)
    a = 100.0 * np.cumprod(1 + rng.normal(0, 0.02, n))
    return pd.DataFrame({"A": a, "USD": 100.0}, index=idx)


def _write_base(tmp_path: Path) -> Path:
    prices = _prices()
    long = prices.rename_axis("date").reset_index().melt("date", var_name="symbol", value_name="close")
    prices_path = tmp_path / "prices.parquet"
    long.to_parquet(prices_path, index=False)
    strategy_path = tmp_path / "frac_strategy.py"
    strategy_path.write_text(_SOURCE_STRATEGY)
    base = {
        "run": {"mode": "backtest", "asof": "2024-02-29", "pipeline": "backtest.pipeline.v1"},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "plugins": {
            "pipeline": {
                "name": "backtest.pipeline.v1",
                "params": {
                    "engine": "vectorbt",
                    "fees": 0.001,
                    "venue": {"allow_shorts": False},
                    "universe": {"symbols": ["A", "USD"]},
                },
            },
            "strategies": [
                {"source": f"{strategy_path}:FracMomentum", "weight": 1.0, "params": {"frac": 1.0}},
            ],
            "data": {"name": "local_file_data", "params_init": {"prices_path": str(prices_path)}},
        },
    }
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False))
    return path


def _write_arms(tmp_path: Path, body: dict) -> Path:
    path = tmp_path / "arms.yaml"
    path.write_text(yaml.safe_dump({"base": "base.yaml", **body}, sort_keys=False))
    return path


def _manifest(summary_path: Path, arm: dict) -> dict:
    return json.loads((summary_path.parent / arm["manifest"]).read_text())


# ----------------------------------------------------------------------
# Expansion: overrides and grid
# ----------------------------------------------------------------------


def test_overrides_apply_dotted_paths_through_lists(tmp_path):
    _write_base(tmp_path)
    spec = load_arms(_write_arms(tmp_path, {"overrides": {"low": {"plugins.strategies.0.params.frac": 0.2}}}))
    [(name, cfg)] = expand_arms(spec)
    assert name == "low"
    assert cfg["plugins"]["strategies"][0]["params"]["frac"] == 0.2
    # The base is not mutated by an arm.
    assert spec.base["plugins"]["strategies"][0]["params"]["frac"] == 1.0


def test_grid_is_the_cartesian_product_with_readable_names(tmp_path):
    _write_base(tmp_path)
    spec = load_arms(
        _write_arms(
            tmp_path,
            {"grid": {"plugins.strategies.0.params.frac": [0.2, 0.4], "plugins.pipeline.params.fees": [0.0, 0.001]}},
        )
    )
    names = [name for name, _ in expand_arms(spec)]
    assert names == ["frac=0.2,fees=0.0", "frac=0.2,fees=0.001", "frac=0.4,fees=0.0", "frac=0.4,fees=0.001"]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({}, "overrides.*grid"),
        ({"overrides": {"a": {"x": 1}}, "grid": {"x": [1]}}, "overrides.*grid"),
        ({"overrides": {"a": {"plugins.strategies.7.params.frac": 1}}}, "index 7"),
        ({"overrides": {"a": {"plugins.nope.params": 1}}}, "nope"),
        ({"overrides": {"a/b": {"plugins.pipeline.params.fees": 0.0}}}, "arm name"),
        ({"overrides": {"a": {"plugins.pipeline.params.execution.lag_bars": 2}}}, "execution"),
        ({"overrides": {"a": {"run.n_trials": 3}}}, "n_trials"),
        ({"overrides": {"a": {"plugins.pipeline.params.fees": 0.0}}, "n_trials": 0}, "n_trials"),
        ({"overrides": {"a": {"x": 1}, "b": {"x": 2}}, "n_trials": 1}, "n_trials"),
        ({"overrides": {"a": {"plugins.pipeline.params.fees": 0.0}}, "parallel": {"max_wrkers": 2}}, "max_wrkers"),
        ({"overrides": {"a": {"plugins.pipeline.params.fees": 0.0}}, "execution": {"lag_bar": 0}}, "unknown key"),
    ],
)
def test_a_malformed_arms_file_is_refused_before_any_run(tmp_path, body, message):
    _write_base(tmp_path)
    with pytest.raises(ValueError, match=message):
        expand_arms(load_arms(_write_arms(tmp_path, body)))


# ----------------------------------------------------------------------
# Parallelism within a memory budget
# ----------------------------------------------------------------------


def test_workers_are_bounded_by_max_workers_arms_and_memory():
    assert plan_workers(5, max_workers=8, memory_budget_gb=2.5, arm_memory_gb=1.0) == 2
    assert plan_workers(5, max_workers=1, memory_budget_gb=64, arm_memory_gb=1.0) == 1
    assert plan_workers(3, max_workers=8, memory_budget_gb=64, arm_memory_gb=1.0) == 3


def test_a_budget_smaller_than_one_arm_is_refused():
    with pytest.raises(ValueError, match="memory budget"):
        plan_workers(5, max_workers=4, memory_budget_gb=0.5, arm_memory_gb=1.0)


# ----------------------------------------------------------------------
# End to end: one arms file == five separate runs
# ----------------------------------------------------------------------


def test_an_arms_file_reproduces_the_separate_runs_metric_for_metric(tmp_path):
    base_path = _write_base(tmp_path)
    arms_path = _write_arms(
        tmp_path,
        {
            "overrides": {f"frac-{f}": {"plugins.strategies.0.params.frac": f} for f in FRACS},
            "parallel": {"max_workers": 2, "memory_budget_gb": 4, "arm_memory_gb": 1.0},
        },
    )
    summary = run_arms(load_arms(arms_path))
    summary_path = Path(summary["path"])

    assert summary["schema"] == ARMS_SCHEMA_ID
    assert summary["status"] == "ok", [a["error"] for a in summary["arms"]]
    assert summary["parallel"]["workers"] == 2
    assert [a["name"] for a in summary["arms"]] == [f"frac-{f}" for f in FRACS]

    # The same five configs, run one by one the old way.
    base = yaml.safe_load(base_path.read_text())
    for frac, arm in zip(FRACS, summary["arms"], strict=True):
        cfg = json.loads(json.dumps(base))
        cfg["plugins"]["strategies"][0]["params"]["frac"] = frac
        cfg["artifacts"]["root"] = str(tmp_path / "separate" / str(frac))
        separate = run_from_config(cfg, PluginRegistry.discover(), config_path=base_path)
        manifest = _manifest(summary_path, arm)
        assert manifest["metrics"] == json_safe(separate.metrics), arm["name"]
        assert arm["metrics"] == manifest["metrics"]

    # The arms are actually different books.
    returns = {a["metrics"]["total_return"] for a in summary["arms"]}
    assert len(returns) == len(FRACS)


def test_the_summary_records_n_trials_and_links_every_manifest(tmp_path):
    _write_base(tmp_path)
    arms_path = _write_arms(
        tmp_path,
        {"grid": {"plugins.strategies.0.params.frac": [0.5, 1.0]}, "n_trials": 9, "parallel": {"max_workers": 1}},
    )
    summary = run_arms(load_arms(arms_path))
    summary_path = Path(summary["path"])
    on_disk = json.loads(summary_path.read_text())
    assert on_disk == json.loads(json.dumps(summary))
    assert on_disk["n_trials"] == 9
    assert len(on_disk["arms"]) == 2
    for arm in on_disk["arms"]:
        manifest = _manifest(summary_path, arm)
        assert manifest["schema"] == "quantbox/run@1"
        assert manifest["run_id"] == arm["run_id"]
        assert manifest["n_trials"] == 9


def test_n_trials_defaults_to_the_number_of_arms(tmp_path):
    _write_base(tmp_path)
    # frac above 1 is a levered book the base (vectorbt, no venue.financing) refuses (ADR-0007).
    arms_path = _write_arms(tmp_path, {"grid": {"plugins.strategies.0.params.frac": [0.25, 0.5, 1.0]}})
    summary = run_arms(load_arms(arms_path), max_workers=1)
    assert summary["n_trials"] == 3
    assert {_manifest(Path(summary["path"]), a)["n_trials"] for a in summary["arms"]} == {3}


def test_a_failing_arm_fails_the_batch_by_name_and_keeps_the_others(tmp_path):
    _write_base(tmp_path)
    arms_path = _write_arms(
        tmp_path,
        {
            "overrides": {
                "good-1": {"plugins.strategies.0.params.frac": 0.5},
                "broken": {"plugins.strategies.0.params.frac": "not-a-number"},
                "good-2": {"plugins.strategies.0.params.frac": 1.0},
            },
            "parallel": {"max_workers": 2, "memory_budget_gb": 4},
        },
    )
    summary = run_arms(load_arms(arms_path))
    assert summary["status"] == "failed"
    assert summary["failed"] == ["broken"]
    by_name = {a["name"]: a for a in summary["arms"]}
    assert by_name["broken"]["status"] == "failed"
    assert "frac must be a number" in by_name["broken"]["error"]
    for name in ("good-1", "good-2"):
        assert by_name[name]["status"] == "ok"
        assert _manifest(Path(summary["path"]), by_name[name])["run_id"] == by_name[name]["run_id"]

    # The CLI turns it into a non-zero exit that names the arm.
    result = CliRunner().invoke(app, ["arms", "-c", str(arms_path), "--max-workers", "1"])
    assert result.exit_code == 1
    assert "broken" in result.output


def test_cli_json_prints_the_summary(tmp_path):
    _write_base(tmp_path)
    arms_path = _write_arms(tmp_path, {"overrides": {"only": {"plugins.strategies.0.params.frac": 0.5}}})
    result = CliRunner().invoke(app, ["arms", "-c", str(arms_path), "--json", "--max-workers", "1"])
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["schema"] == ARMS_SCHEMA_ID
    assert summary["parallel"]["workers"] == 1


# ----------------------------------------------------------------------
# One timing setting for arms and sweep
# ----------------------------------------------------------------------


def test_the_arms_execution_block_is_the_timing_of_every_arm(tmp_path):
    _write_base(tmp_path)
    arms_path = _write_arms(
        tmp_path,
        {"execution": {"lag_bars": 2}, "grid": {"plugins.strategies.0.params.frac": [0.5, 1.0]}},
    )
    summary = run_arms(load_arms(arms_path), max_workers=1)
    assert summary["execution"]["lag_bars"] == 2
    for arm in summary["arms"]:
        assert _manifest(Path(summary["path"]), arm)["execution"]["lag_bars"] == 2


def test_the_arms_execution_block_refuses_same_bar(tmp_path):
    _write_base(tmp_path)
    arms_path = _write_arms(
        tmp_path, {"execution": {"lag_bars": 0}, "overrides": {"a": {"plugins.pipeline.params.fees": 0.0}}}
    )
    with pytest.raises(ValueError, match="lag_bars must be >= 1"):
        load_arms(arms_path)


def test_the_arms_execution_block_must_agree_with_the_base(tmp_path):
    base_path = _write_base(tmp_path)
    base = yaml.safe_load(base_path.read_text())
    base["plugins"]["pipeline"]["params"]["execution"] = {"lag_bars": 2}
    base_path.write_text(yaml.safe_dump(base))
    arms_path = _write_arms(
        tmp_path, {"execution": {"lag_bars": 3}, "overrides": {"a": {"plugins.pipeline.params.fees": 0.0}}}
    )
    with pytest.raises(ValueError, match="contradicts"):
        load_arms(arms_path)


def test_without_an_execution_block_the_default_timing_is_recorded(tmp_path):
    _write_base(tmp_path)
    arms_path = _write_arms(tmp_path, {"overrides": {"a": {"plugins.strategies.0.params.frac": 0.5}}})
    summary = run_arms(load_arms(arms_path), max_workers=1)
    assert summary["execution"]["lag_bars"] == 1
    assert _manifest(Path(summary["path"]), summary["arms"][0])["execution"]["lag_bars"] == 1


# ----------------------------------------------------------------------
# `source:` everywhere: sweep and variants
# ----------------------------------------------------------------------


@pytest.fixture
def fake_quantbox_datasets(monkeypatch):
    """`quantbox sweep` loads its data through quantbox_datasets.lock; stand it in."""
    prices = _prices()

    class _Dataset:
        def __init__(self):
            self.prices = prices
            self.volume = prices * 0 + 1.0
            self.market_cap = prices * 0 + 1.0

    pkg = types.ModuleType("quantbox_datasets")
    lock = types.ModuleType("quantbox_datasets.lock")
    lock.find_lock = lambda start: None
    lock.load = lambda name, lock=None: _Dataset()
    pkg.lock = lock
    monkeypatch.setitem(sys.modules, "quantbox_datasets", pkg)
    monkeypatch.setitem(sys.modules, "quantbox_datasets.lock", lock)


@pytest.mark.parametrize("lag", [1, 2])
def test_a_source_strategy_runs_in_a_sweep_and_its_manifest_records_the_timing(tmp_path, fake_quantbox_datasets, lag):
    (tmp_path / "frac_strategy.py").write_text(_SOURCE_STRATEGY)
    cfg = {
        # Relative to the sweep config, like every other sweep path.
        "strategy": {"source": "frac_strategy.py:FracMomentum"},
        "data": {"dataset": "toy", "frames": ["prices"]},
        "base_params": {},
        "sweep_params": {"frac": [0.5, 1.0]},
        "bands": [0.0],
        "backtest": {"fees": 0.0},
        "execution": {"lag_bars": lag},
        "output_dir": "out",
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(cfg))
    result = CliRunner().invoke(app, ["sweep", "-c", str(path)])
    assert result.exit_code == 0, (result.output, result.exception)

    grid = pd.read_parquet(tmp_path / "out" / "grid.parquet")
    assert len(grid) == 2
    manifest = json.loads((tmp_path / "out" / "sweep_manifest.json").read_text())
    assert manifest["schema"] == "quantbox/sweep@1"
    assert manifest["execution"]["lag_bars"] == lag
    assert manifest["n_trials"] == 2
    assert manifest["strategy"] == {"source": "frac_strategy.py:FracMomentum"}
    assert manifest["grid"] == "grid.parquet"


def test_a_registered_strategy_name_still_works_in_a_sweep(tmp_path, fake_quantbox_datasets):
    cfg = {
        "strategy": "strategy.static_weights.v1",
        "data": {"dataset": "toy", "frames": ["prices"]},
        "sweep_params": {"weights": [{"A": 1.0}]},
        "bands": [0.0],
        "output_dir": "out",
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(cfg))
    result = CliRunner().invoke(app, ["sweep", "-c", str(path)])
    assert result.exit_code == 0, (result.output, result.exception)
    manifest = json.loads((tmp_path / "out" / "sweep_manifest.json").read_text())
    assert manifest["execution"]["lag_bars"] == 1
    assert manifest["strategy"] == {"name": "strategy.static_weights.v1"}


def test_a_source_strategy_runs_as_a_variant(tmp_path):
    base_path = _write_base(tmp_path)
    cfg = yaml.safe_load(base_path.read_text())
    source = cfg["plugins"]["strategies"][0]["source"]
    cfg["plugins"]["pipeline"]["params"]["variants"] = [
        {"name": "half", "strategy": {"source": source, "params": {"frac": 0.5}}},
        {"name": "full", "strategy": {"source": source, "params": {"frac": 1.0}}},
    ]
    result = run_from_config(cfg, PluginRegistry.discover(), config_path=base_path)
    assert result.notes["variants"] == ["half", "full"]


# --- the worker pool honours requires-python >=3.10 (PR #222 review BLOCKER) ---


@pytest.mark.parametrize(
    ("version", "recycles"),
    [((3, 10, 20), False), ((3, 11, 0), True), ((3, 12, 3), True)],
)
def test_the_arm_pool_only_asks_for_child_recycling_where_python_has_it(monkeypatch, version, recycles):
    """``max_tasks_per_child`` is 3.11+; on 3.10 passing it is a TypeError that
    kills every parallel batch after its directory already exists."""
    import inspect
    from concurrent.futures import ProcessPoolExecutor

    from quantbox import arms

    # Unpatched: every kwarg handed over is one THIS interpreter's executor accepts.
    accepted = inspect.signature(ProcessPoolExecutor.__init__).parameters
    assert set(arms._pool_kwargs(2)) <= set(accepted)

    monkeypatch.setattr(arms.sys, "version_info", version)
    kwargs = arms._pool_kwargs(2)
    assert kwargs["max_workers"] == 2
    assert kwargs["mp_context"].get_start_method() == "spawn"
    assert ("max_tasks_per_child" in kwargs) is recycles
