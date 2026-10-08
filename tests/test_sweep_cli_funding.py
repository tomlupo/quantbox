"""``quantbox sweep`` reads the dataset's market for the funding guard (TOM-1627).

The CLI sweep loads a dataset by name, but never read its ``market``: a sweep on a perp
dataset whose ``data.frames`` leave out ``funding_rates`` ran on vectorbt with no
funding at all (the seam saw no series, so it had nothing to refuse). The CLI now reads
the market from the dataset manifest and hands it, with the frames it loaded, to the
same :func:`quantbox.funding_guard.check_funding`.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from quantbox.cli import app
from quantbox.exceptions import ConfigValidationError

NOT_CHARGED = "funding_not_charged"
MISSING = "funding_missing"
IGNORE = {"ignore": True, "reason": "perp book studied without funding: the carry is measured in TOM-0000"}


def _prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=60, freq="D", name="date")
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {s: 100.0 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))) for s in ("AAA", "BBB")},
        index=idx,
    )


# =========================================================================== 1. the CLI sweep


class _Dataset:
    """Stand-in for quantbox_datasets.Dataset: its parsed ``manifest`` and its wide frames."""

    def __init__(self, market: str, funding: bool) -> None:
        self.manifest = {"name": "perp-daily", "market": market}
        self._frames = {"prices": _prices()}
        if funding:
            self._frames["funding_rates"] = _prices() * 0 + 0.0001

    def _read(self, key: str) -> pd.DataFrame:
        return self._frames.get(key, pd.DataFrame())

    prices = property(lambda self: self._read("prices"))
    volume = property(lambda self: self._read("volume"))
    market_cap = property(lambda self: self._read("market_cap"))
    funding_rates = property(lambda self: self._read("funding_rates"))


@pytest.fixture
def lock_serving(monkeypatch):
    """quantbox_datasets.lock serving one dataset; returns a setter for its market and funding."""
    served: dict[str, Any] = {}
    module = types.ModuleType("quantbox_datasets.lock")
    module.find_lock = lambda start: None
    module.load = lambda name, **kwargs: _Dataset(served["market"], served["funding"])
    monkeypatch.setitem(sys.modules, "quantbox_datasets", types.ModuleType("quantbox_datasets"))
    monkeypatch.setitem(sys.modules, "quantbox_datasets.lock", module)

    def serve(market: str, *, funding: bool = False) -> None:
        served.update(market=market, funding=funding)

    return serve


_STRATEGY = """
import pandas as pd

from quantbox.contracts import PluginMeta


class Hold:
    meta = PluginMeta(name="strategy.hold_sweep.v1", kind="strategy", version="0.0.1", core_compat=">=0.1.0")

    def __init__(self, **params):
        self.params = params

    def run(self, data):
        prices = data["prices"]
        return {"weights": pd.DataFrame(self.params["w"], index=prices.index, columns=prices.columns)}
"""


def _sweep(tmp_path: Path, engine: str, *, frames: list[str], funding: dict | None = None):
    (tmp_path / "hold.py").write_text(_STRATEGY)
    backtest: dict[str, Any] = {"engine": engine, "fees": 0.0}
    if funding is not None:
        backtest["funding"] = funding
    cfg = {
        "strategy": {"source": "hold.py:Hold"},
        "data": {"dataset": "perp-daily", "frames": frames},
        "base_params": {},
        "sweep_params": {"w": [0.5]},
        "bands": [0.0],
        "heatmap": {"metrics": ["total_return"]},
        "backtest": backtest,
        "output_dir": "out",
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return CliRunner().invoke(app, ["sweep", "-c", str(path)])


@pytest.mark.parametrize("market", ["perp", "futures"])
def test_a_cli_sweep_on_a_perp_dataset_without_funding_on_vectorbt_is_refused(tmp_path, lock_serving, market):
    lock_serving(market)
    result = _sweep(tmp_path, "vectorbt", frames=["prices"])
    assert isinstance(result.exception, ConfigValidationError), (repr(result.exception), result.output)
    assert [f.code for f in result.exception.findings] == [NOT_CHARGED]
    assert f"market: {market}" in str(result.exception)
    assert not (tmp_path / "out" / "grid.parquet").exists()


def test_a_cli_sweep_on_a_perp_dataset_whose_funding_is_left_out_of_frames_on_vectorbt_is_refused(
    tmp_path, lock_serving
):
    lock_serving("futures", funding=True)  # the dataset carries funding; data.frames leaves it out
    result = _sweep(tmp_path, "vectorbt", frames=["prices"])
    assert isinstance(result.exception, ConfigValidationError), (repr(result.exception), result.output)
    assert NOT_CHARGED in str(result.exception)


def test_a_cli_sweep_on_a_perp_dataset_without_funding_on_rsims_is_refused(tmp_path, lock_serving):
    lock_serving("perp")
    result = _sweep(tmp_path, "rsims", frames=["prices"])
    assert isinstance(result.exception, ConfigValidationError), (repr(result.exception), result.output)
    assert [f.code for f in result.exception.findings] == [MISSING]


def test_a_cli_sweep_on_a_perp_dataset_runs_with_the_declared_ignore(tmp_path, lock_serving):
    lock_serving("perp")
    result = _sweep(tmp_path, "vectorbt", frames=["prices"], funding=dict(IGNORE))
    assert result.exit_code == 0, (repr(result.exception), result.output)
    assert (tmp_path / "out" / "grid.parquet").exists()


def test_a_cli_sweep_on_a_perp_dataset_with_funding_runs_on_rsims(tmp_path, lock_serving):
    lock_serving("perp", funding=True)
    result = _sweep(tmp_path, "rsims", frames=["prices", "funding_rates"])
    assert result.exit_code == 0, (repr(result.exception), result.output)


@pytest.mark.parametrize("engine", ["vectorbt", "rsims"])
def test_a_cli_sweep_on_a_spot_dataset_runs_on_both_engines(tmp_path, lock_serving, engine):
    lock_serving("spot")
    result = _sweep(tmp_path, engine, frames=["prices"])
    assert result.exit_code == 0, (repr(result.exception), result.output)
