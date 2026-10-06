"""Every run config in ``cookbook/configs/`` runs (TOM-1526).

Two configs were broken on ``dev`` and no test said so: ``run_stress_test.yaml``
stopped with ``KeyError: 'pipeline'`` and ``run_synthetic_backtest.yaml`` with
``'list' object has no attribute 'to_parquet'``. ``quantbox validate`` passed
both, and ``run --dry-run`` passed both too: neither builds the run.

Two tests per config:

- **resolve** — every config: :func:`quantbox.runner.prepare_config` and
  :func:`quantbox.runner.resolve_run`, the step that builds every plugin the
  run names. No data, no network, no broker instance.
- **run** — :func:`quantbox.runner.run_from_config`, end to end, as far as the
  config's TIER allows. A config this test cannot run is skipped BY NAME with
  its reason, in :data:`TIERS`. A run config missing from :data:`TIERS` fails
  :func:`test_every_run_config_has_a_tier`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from quantbox.dataset_lock import DatasetResolveError, lock_for_config, require_match, resolve_dataset
from quantbox.registry import PluginRegistry
from quantbox.runner import prepare_config, resolve_run, run_from_config

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "cookbook" / "configs"


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


#: A run config is a file with a ``run:`` block (``instruments.yaml`` is a mapping a config reads).
RUN_CONFIGS = sorted(p for p in CONFIG_DIR.glob("*.yaml") if "run" in _load(p))

#: Runs end to end, offline, on every box.
FULL = "full"
#: Runs end to end when its pinned dataset (``plugins.data.params_init.dataset``) resolves on this
#: box ($QUANTBOX_DATASETS_ROOT, at the sha in datasets.lock); a named skip otherwise.
DATASET = "dataset"
#: Resolved only; the reason says why a full run is not possible in a test.
RESOLVE = "resolve"

_NETWORK = "its data plugin ({}) fetches market data over the network"
_USER_FILE = "it reads files the user supplies ({})"

#: Every run config, by name: how far this test runs it, and why not further.
TIERS: dict[str, tuple[str, str]] = {
    "run_synthetic_backtest.yaml": (FULL, "synthetic data, generated in process"),
    "run_backtest_overlay_reversal_derisk.yaml": (FULL, "the committed canonical fixture"),
    "run_stress_test.yaml": (DATASET, "pinned dataset"),
    "run_backtest_beglobal.yaml": (DATASET, "pinned dataset; money_market sleeve on SHY (TOM-1528)"),
    "run_backtest_portfolio_optimizer.yaml": (DATASET, "pinned dataset"),
    "example_minimal.yaml": (DATASET, "pinned dataset"),
    "run_backtest_crypto_trend.yaml": (RESOLVE, _NETWORK.format("binance.live_data.v1")),
    "run_backtest_ml_prediction.yaml": (RESOLVE, _NETWORK.format("binance.live_data.v1")),
    "run_crypto_regime.yaml": (RESOLVE, _NETWORK.format("binance.live_data.v1")),
    "run_spot_paper_crypto_trend.yaml": (RESOLVE, _NETWORK.format("binance.live_data.v1")),
    "run_trading_multi_strategy.yaml": (RESOLVE, _NETWORK.format("binance.live_data.v1")),
    "run_xsmom.yaml": (RESOLVE, _NETWORK.format("binance.live_data.v1")),
    "run_trading_full.yaml": (RESOLVE, _NETWORK.format("binance.live_data.v1") + "; broker binance.live.v1 needs keys"),
    "run_futures_paper.yaml": (RESOLVE, _NETWORK.format("binance.futures_data.v1")),
    "run_futures_paper_carver_trend.yaml": (RESOLVE, _NETWORK.format("binance.futures_data.v1")),
    "run_futures_paper_crypto_trend.yaml": (RESOLVE, _NETWORK.format("binance.futures_data.v1")),
    "run_futures_paper_momentum_ls.yaml": (RESOLVE, _NETWORK.format("binance.futures_data.v1")),
    "run_fund_selection.yaml": (RESOLVE, _USER_FILE.format("./data/curated/prices.parquet")),
    "run_trade_from_allocations.yaml": (
        RESOLVE,
        _USER_FILE.format("./data/curated/prices.parquet and a fund_selection run's allocations"),
    ),
    "run_trade_from_allocations_ibkr_paper.yaml": (
        RESOLVE,
        _USER_FILE.format("./data/curated/prices.parquet and <FUND_RUN_ID>/allocations.parquet"),
    ),
    "run_trade_from_allocations_binance_live.yaml": (
        RESOLVE,
        "mode live with broker binance.live.v1: a test never places a real order",
    ),
}

IDS = [p.name for p in RUN_CONFIGS]


@pytest.fixture(scope="module")
def registry() -> PluginRegistry:
    return PluginRegistry.discover()


@pytest.mark.pipeline_smoke
def test_every_run_config_has_a_tier():
    """A new config is run, or skipped with a reason, from the day it lands: no silent gap."""
    assert len(RUN_CONFIGS) >= 20, IDS  # a glob that matches nothing would make every test below vacuous
    assert sorted(TIERS) == sorted(IDS), {
        "no tier": sorted(set(IDS) - set(TIERS)),
        "tier for a missing config": sorted(set(TIERS) - set(IDS)),
    }


@pytest.mark.pipeline_smoke
@pytest.mark.parametrize("path", RUN_CONFIGS, ids=IDS)
def test_cookbook_config_resolves(path: Path, registry: PluginRegistry, monkeypatch):
    """Every config builds every plugin it names (``run --dry-run`` stops before this)."""
    monkeypatch.chdir(REPO)
    cfg = prepare_config(_load(path))
    resolved = resolve_run(cfg, registry, config_path=path)
    assert resolved.pipe_name == cfg["plugins"]["pipeline"]["name"]


def _dataset_or_skip(path: Path, cfg: dict[str, Any]) -> None:
    name = ((cfg["plugins"].get("data") or {}).get("params_init") or {}).get("dataset")
    assert name, f"{path.name}: tier {DATASET!r} but plugins.data.params_init.dataset is not set"
    try:
        require_match(resolve_dataset(name, lock=lock_for_config(path)))
    except DatasetResolveError as exc:
        pytest.skip(f"{path.name}: pinned dataset {name!r} is not available on this box: {exc}")


@pytest.mark.pipeline_smoke
@pytest.mark.slow
@pytest.mark.parametrize("path", RUN_CONFIGS, ids=IDS)
def test_cookbook_config_runs(path: Path, registry: PluginRegistry, tmp_path, monkeypatch):
    tier, reason = TIERS[path.name]
    if tier == RESOLVE:
        pytest.skip(f"{path.name}: resolved only — {reason}")
    cfg = _load(path)
    if tier == DATASET:
        _dataset_or_skip(path, cfg)
    monkeypatch.chdir(REPO)  # configs name repo-relative paths (./cookbook/canonical/...)
    cfg["artifacts"]["root"] = str(tmp_path)
    result = run_from_config(cfg, registry, config_path=path)
    manifest = json.loads((tmp_path / result.run_id / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["plugins"]["pipeline"] == cfg["plugins"]["pipeline"]["name"]
    assert result.metrics, f"{path.name}: the run reported no metrics"
