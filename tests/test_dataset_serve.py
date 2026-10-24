"""A monthly or returns dataset is served by name from its manifest's ``serve`` block (TOM-2232).

Guards whose job is to fail (a gap, an unknown kind, a frequency the config contradicts,
an altered table), the pure logic of the total-return index, and the one whole path: a
line-local frozen monthly returns dataset run through the backtest, its hash and its
bars_per_year recorded in run_manifest.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from quantbox.cli import app
from quantbox.dataset import total_return_index
from quantbox.dataset_lock import DatasetServeError, resolve_dataset
from quantbox.exceptions import ConfigValidationError
from quantbox.frequency import Frequency
from quantbox.registry import PluginRegistry
from quantbox.runner import resolve_run, run_from_config


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _month_ends(n: int, start: str = "2000-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME", name="date")


def _returns(n: int = 48, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.006, 0.04, size=(n, 3)), index=_month_ends(n), columns=["EQ", "BD", "CASH"])


def _freeze(where: Path, name: str, table: pd.DataFrame, serve: dict | None, *, pin_file: str = "monthly") -> str:
    """A frozen dataset as a line's data plugin writes it (D5): the table, its manifest; returns the pin."""
    ds = where / name
    ds.mkdir(parents=True)
    table.to_parquet(ds / f"{pin_file}.parquet")
    sha = _sha(ds / f"{pin_file}.parquet")
    manifest = {
        "name": name,
        "refresh": "frozen",
        "source": "test",
        "pin_file": f"{pin_file}.parquet",
        "content_hashes": {f"{pin_file}.parquet": {"sha256": sha}},
    }
    if serve is not None:
        manifest["serve"] = serve
    (ds / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    return sha


@pytest.fixture
def line(tmp_path, monkeypatch):
    """A research line with no datasets root at all: only its own frozen copy."""
    monkeypatch.delenv("QUANTBOX_DATASETS_ROOT", raising=False)
    monkeypatch.setattr("quantbox.dataset_lock.datasets_root", _no_root)
    line = tmp_path / "line"
    line.mkdir()
    return line


def _no_root():
    from quantbox.dataset_lock import DatasetResolveError

    raise DatasetResolveError("no datasets root")


MONTHLY_RETURNS = {"table": "monthly", "kind": "returns", "frequency": "monthly"}


# -- the total-return index (pure logic) ---------------------------------------


def test_the_index_gives_back_every_return_the_first_included():
    returns = _returns()
    index = total_return_index(returns, "1M")
    assert index.index[0] == pd.Timestamp("1999-12-31")  # one month-end before the first return
    assert (index.iloc[0] == 1.0).all()
    pd.testing.assert_frame_equal(index.pct_change().iloc[1:], returns, check_freq=False, atol=1e-12, rtol=0)


def test_a_series_that_starts_late_is_based_at_1_one_bar_before_its_first_return():
    returns = _returns()
    returns.iloc[:10, 1] = np.nan
    index = total_return_index(returns, "1M")
    assert index["BD"].iloc[:10].isna().all()
    assert index["BD"].iloc[10] == 1.0
    np.testing.assert_allclose(index["BD"].pct_change().iloc[11:], returns["BD"].iloc[10:], atol=1e-12)


def test_a_gap_inside_a_series_is_refused_not_compounded_as_zero():
    returns = _returns()
    returns.iloc[20:23, 0] = np.nan
    with pytest.raises(DatasetServeError, match="'EQ' has a gap"):
        total_return_index(returns, "1M")


def test_a_missing_month_is_refused():
    returns = _returns().drop(_month_ends(48)[30])
    with pytest.raises(DatasetServeError, match="monthly returns table has a gap"):
        total_return_index(returns, "1M")


def test_a_return_at_or_below_minus_100_percent_is_refused():
    returns = _returns()
    returns.iloc[5, 2] = -1.0
    with pytest.raises(DatasetServeError, match="-100 %"):
        total_return_index(returns, "1M")


@pytest.mark.parametrize("calendar", ["24/7", "NYSE"])
@pytest.mark.parametrize(("spec", "expected"), [("1M", 12.0), ("monthly", 12.0), ("3M", 4.0)])
def test_a_month_bar_is_12_per_year_on_any_calendar(spec, expected, calendar):
    assert Frequency.parse({"bar_size": spec, "calendar": calendar}).bars_per_year() == expected


# -- resolution --------------------------------------------------------------------


def test_a_frozen_dataset_resolves_with_the_sha256_of_its_pin_file(line):
    sha = _freeze(line / "datasets", "sbbi-like", _returns(), None, pin_file="annual")
    (line / "datasets.lock").write_text(f"sbbi-like: {sha}\n")
    out = resolve_dataset("sbbi-like", lock=line / "datasets.lock")
    assert out["actual_sha256"] == sha and out["matches"] is True
    assert out["pin_file"] == "annual.parquet"
    assert "serve" not in out  # no serve block: served exactly as before


@pytest.mark.parametrize(
    ("serve", "match"),
    [
        ({"kind": "excess_returns", "table": "monthly"}, "serve.kind 'excess_returns'"),
        ({"kind": "returns", "table": "monthly", "frequency": "fortnightly"}, "serve.frequency"),
        ({"kind": "returns", "table": "quarterly"}, "does not exist"),
        ({"kind": "returns", "table": "monthly", "weights": "x"}, "unknown keys"),
    ],
)
def test_a_serve_block_that_cannot_be_served_is_refused_by_name(line, serve, match):
    sha = _freeze(line / "datasets", "kf", _returns(), serve)
    (line / "datasets.lock").write_text(f"kf: {sha}\n")
    with pytest.raises(DatasetServeError, match=match) as err:
        resolve_dataset("kf", lock=line / "datasets.lock")
    assert isinstance(err.value, ConfigValidationError)


def test_a_served_table_that_is_not_the_bytes_its_manifest_vouches_for_is_refused(line):
    sha = _freeze(line / "datasets", "sbbi-like", _returns(), MONTHLY_RETURNS, pin_file="annual")
    _returns(seed=8).to_parquet(line / "datasets" / "sbbi-like" / "monthly.parquet")  # not in content_hashes
    (line / "datasets.lock").write_text(f"sbbi-like: {sha}\n")
    with pytest.raises(DatasetServeError, match="must be vouched for"):
        resolve_dataset("sbbi-like", lock=line / "datasets.lock")


def test_the_root_wins_over_a_line_copy_and_the_pin_decides_its_bytes(line, tmp_path, monkeypatch):
    """Promoted: the dataset now sits under the root too. The root's copy is read, held to the line's pin."""
    root = tmp_path / "root"
    _freeze(root, "kf", _returns(seed=8), MONTHLY_RETURNS)
    sha = _freeze(line / "datasets", "kf", _returns(), MONTHLY_RETURNS)
    monkeypatch.setattr("quantbox.dataset_lock.datasets_root", lambda: root)
    (line / "datasets.lock").write_text(f"kf: {sha}\n")
    result = CliRunner().invoke(app, ["dataset", "resolve", "kf", "--lock", str(line / "datasets.lock"), "--json"])
    assert result.exit_code == 1  # the root's bytes are not the pinned build
    assert str(root / "kf") in result.output


# -- the whole path ----------------------------------------------------------------


def _config(line: Path, params: dict | None = None) -> Path:
    cfg = {
        "run": {"mode": "backtest", "asof": "2003-12-31", "pipeline": "backtest.pipeline.v1"},
        "artifacts": {"root": str(line / "artifacts")},
        "plugins": {
            "pipeline": {"name": "backtest.pipeline.v1", "params": {"fees": 0.0, **(params or {})}},
            "strategies": [
                {
                    "name": "strategy.cross_asset_momentum.v1",
                    "weight": 1.0,
                    "params": {"windows": [3, 6], "long_only": True, "position_size": 0.5},
                }
            ],
            "data": {"name": "local_file_data", "params_init": {"dataset": "kf"}},
        },
    }
    path = line / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_a_line_local_monthly_returns_dataset_runs_and_records_its_hash_and_bars_per_year(line):
    returns = _returns()
    sha = _freeze(line / "datasets", "kf", returns, MONTHLY_RETURNS)
    (line / "datasets.lock").write_text(f"kf: {sha}\n")
    config = _config(line)

    cfg = yaml.safe_load(config.read_text())
    result = run_from_config(cfg, PluginRegistry.discover(), config_path=config)
    manifest = json.loads((line / "artifacts" / result.run_id / "run_manifest.json").read_text())

    assert manifest["dataset"]["sha256"] == sha
    serve = manifest["dataset"]["resolved"]["serve"]
    assert serve["kind"] == "returns" and "cumulative product" in serve["construction"]
    assert serve["bars_per_year"] == 12.0 and serve["frequency_source"] == "dataset"
    served = pd.read_parquet(line / "artifacts" / result.run_id / "universe.parquet")
    assert sorted(served["symbol"]) == ["BD", "CASH", "EQ"]


def test_a_config_declaring_another_frequency_than_its_dataset_is_refused(line):
    sha = _freeze(line / "datasets", "kf", _returns(), MONTHLY_RETURNS)
    (line / "datasets.lock").write_text(f"kf: {sha}\n")
    config = _config(line, {"frequency": "1d"})
    with pytest.raises(DatasetServeError, match="is 1M bars"):
        resolve_run(yaml.safe_load(config.read_text()), PluginRegistry.discover(), config_path=config)


def test_a_returns_dataset_with_a_gap_is_refused_before_any_strategy_runs(line):
    returns = _returns()
    returns.iloc[20, 0] = np.nan
    sha = _freeze(line / "datasets", "kf", returns, MONTHLY_RETURNS)
    (line / "datasets.lock").write_text(f"kf: {sha}\n")
    config = _config(line)
    with pytest.raises(DatasetServeError, match="has a gap"):
        run_from_config(yaml.safe_load(config.read_text()), PluginRegistry.discover(), config_path=config)
