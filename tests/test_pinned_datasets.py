"""quantbox reads quantbox-datasets by name, pinned by datasets.lock — never by sibling path (TOM-993)."""

from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from quantbox.plugins.datasources.local_file_data import LocalFileDataPlugin
from quantbox.validate import validate_config

REPO = Path(__file__).resolve().parents[1]


class _FakeDataset:
    def __init__(self) -> None:
        idx = pd.date_range("2026-01-01", periods=5, freq="D")  # tz-naive, as quantbox-datasets serves it
        self.prices = pd.DataFrame({"AAA": range(5), "BBB": range(5), "CCC": range(5)}, index=idx, dtype=float)
        self.volume = self.prices * 10
        self.market_cap = pd.DataFrame()
        self.funding_rates = pd.DataFrame()
        self.universe = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"]})

    def _read(self, key: str) -> pd.DataFrame:
        return getattr(self, key)


@pytest.fixture
def fake_lock(monkeypatch):
    """Stand in for quantbox_datasets.lock; record every load() call."""
    calls: list[tuple[str, dict]] = []
    module = types.ModuleType("quantbox_datasets.lock")

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return _FakeDataset()

    module.load = load

    # Same semantics as the real find_lock: the nearest EXISTING lock, else None.
    def find_lock(start=None):
        here = Path(start or Path.cwd()).resolve()
        for directory in (here, *here.parents):
            if (directory / "datasets.lock").is_file():
                return directory / "datasets.lock"
        return None

    module.find_lock = find_lock
    monkeypatch.setitem(sys.modules, "quantbox_datasets", types.ModuleType("quantbox_datasets"))
    monkeypatch.setitem(sys.modules, "quantbox_datasets.lock", module)
    return calls


def test_local_file_data_loads_dataset_by_name(fake_lock):
    plugin = LocalFileDataPlugin(dataset="etf-daily")
    universe = plugin.load_universe({})
    data = plugin.load_market_data(universe[universe["symbol"] != "CCC"], "2026-01-03", {})

    assert [name for name, _ in fake_lock] == ["etf-daily"]  # loaded once, pinned (no pinned=False)
    assert fake_lock[0][1] == {}
    assert list(universe["symbol"]) == ["AAA", "BBB", "CCC"]
    assert list(data["prices"].columns) == ["AAA", "BBB"]
    assert data["prices"].index.max() == pd.Timestamp("2026-01-03", tz="UTC")  # asof cut, UTC like file reads
    assert data["volume"].shape == (3, 2)
    assert data["market_cap"].empty and data["funding_rates"].empty


def test_dataset_name_is_not_flagged_legacy():
    cfg = {"plugins": {"data": {"name": "local_file_data", "params_init": {"dataset": "etf-daily"}}}}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_config(cfg)
    assert not [x for x in w if issubclass(x.category, DeprecationWarning)]


def test_sweep_loads_dataset_with_the_lock_next_to_the_config(fake_lock, monkeypatch, tmp_path):
    import quantbox.analysis
    from quantbox.cli import app

    seen = {}

    def fake_run_grid(**kwargs):
        seen.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(quantbox.analysis, "run_grid", fake_run_grid)
    (tmp_path / "datasets.lock").write_text("crypto-spot-daily: abc123\n")
    config = tmp_path / "sweep.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "strategy": "strategy.crypto_regime_trend.v1",
                "data": {"dataset": "crypto-spot-daily", "frames": ["prices", "volume"]},
                "sweep_params": {},
            }
        )
    )

    result = CliRunner().invoke(app, ["sweep", "-c", str(config)])

    assert result.exit_code == 0, result.output
    assert fake_lock == [("crypto-spot-daily", {"lock": tmp_path / "datasets.lock"})]
    assert set(seen["market_data"]) == {"prices", "volume"}


def test_repo_lock_pins_every_dataset_the_cookbook_reads():
    pins = yaml.safe_load((REPO / "datasets.lock").read_text())
    used = set()
    for path in (REPO / "cookbook").rglob("*.yaml"):
        text = path.read_text()
        assert "quantbox-datasets/datasets" not in text, f"{path} reads quantbox-datasets through a sibling path"
        data = ((yaml.safe_load(text) or {}).get("plugins") or {}).get("data") or {}
        if (data.get("params_init") or {}).get("dataset"):
            used.add(data["params_init"]["dataset"])
    assert used, "expected at least one cookbook config to read a dataset by name"
    assert used <= {name for name, sha in pins.items() if sha}
