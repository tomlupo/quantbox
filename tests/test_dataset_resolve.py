"""A config names a dataset; the runner resolves it from datasets.lock + $QUANTBOX_DATASETS_ROOT (TOM-1349)."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import types
import warnings
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from quantbox.cli import app
from quantbox.registry import PluginRegistry
from quantbox.runner import run_from_config


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(root: Path, name: str, *, market: str = "spot", funding: bool = False) -> str:
    """Write a tiny dataset under *root*; return the sha256 of its prices.parquet."""
    ds = root / name
    ds.mkdir(parents=True)
    idx = pd.date_range("2024-01-01", periods=60, freq="D", name="date")
    prices = pd.DataFrame({"AAA": [100.0 + i for i in range(60)], "BBB": [50.0 + i % 7 for i in range(60)]}, index=idx)
    prices.to_parquet(ds / "prices.parquet")
    (ds / "manifest.yaml").write_text(yaml.safe_dump({"name": name, "market": market}))
    if funding:
        (prices * 0 + 0.0001).to_parquet(ds / "funding_rates.parquet")
    return _sha(ds / "prices.parquet")


class _ParquetDataset:
    """Stand-in for quantbox_datasets.Dataset: wide frames read from one directory."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _read(self, key: str) -> pd.DataFrame:
        f = self.path / f"{key}.parquet"
        return pd.read_parquet(f) if f.exists() else pd.DataFrame()

    prices = property(lambda self: self._read("prices"))
    volume = property(lambda self: self._read("volume"))
    market_cap = property(lambda self: self._read("market_cap"))
    funding_rates = property(lambda self: self._read("funding_rates"))
    universe = property(lambda self: pd.DataFrame({"symbol": list(self.prices.columns)}))


@pytest.fixture
def fake_datasets(monkeypatch):
    """quantbox_datasets.lock.load serving <root>/<name>; records every call."""
    calls: list[tuple[str, dict]] = []
    module = types.ModuleType("quantbox_datasets.lock")

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return _ParquetDataset(Path(kwargs["root"]) / name)

    module.load = load
    monkeypatch.setitem(sys.modules, "quantbox_datasets", types.ModuleType("quantbox_datasets"))
    monkeypatch.setitem(sys.modules, "quantbox_datasets.lock", module)
    return calls


@pytest.fixture
def root(tmp_path, monkeypatch):
    root = tmp_path / "datasets"
    monkeypatch.setenv("QUANTBOX_DATASETS_ROOT", str(root))
    return root


def _resolve_json(*args: str) -> tuple[int, dict]:
    result = CliRunner().invoke(app, ["dataset", "resolve", *args, "--json"])
    try:
        return result.exit_code, json.loads(result.stdout)
    except json.JSONDecodeError:
        raise AssertionError(result.output) from None


def test_resolve_json_reports_path_pin_market_funding_and_match(root, tmp_path, monkeypatch):
    sha = _build(root, "crypto-futures-daily", market="futures", funding=True)
    line = tmp_path / "line"
    line.mkdir()
    (line / "datasets.lock").write_text(f"crypto-futures-daily: {sha}\n")
    monkeypatch.chdir(line)

    code, out = _resolve_json("crypto-futures-daily")

    assert code == 0
    assert out["name"] == "crypto-futures-daily"
    assert out["path"] == str(root / "crypto-futures-daily")
    assert out["lock"] == str(line / "datasets.lock")
    assert out["sha256"] == sha
    assert out["actual_sha256"] == sha
    assert out["matches"] is True
    assert out["market"] == "futures"
    assert out["funding_rates"] == str(root / "crypto-futures-daily" / "funding_rates.parquet")


def test_resolve_json_without_a_pin_reports_no_match_verdict(root, tmp_path, monkeypatch):
    sha = _build(root, "etf-daily")
    monkeypatch.chdir(tmp_path)

    code, out = _resolve_json("etf-daily")

    assert code == 0
    assert out["sha256"] is None and out["lock"] is None
    assert out["actual_sha256"] == sha
    assert out["matches"] is None
    assert out["funding_rates"] is None


def test_resolve_json_mismatch_exits_nonzero_and_names_both_shas(root, tmp_path, monkeypatch):
    actual = _build(root, "etf-daily")
    pinned = "a" * 64
    (tmp_path / "datasets.lock").write_text(f"etf-daily: {pinned}\n")
    monkeypatch.chdir(tmp_path)

    code, out = _resolve_json("etf-daily")

    assert code == 1
    assert out["matches"] is False
    assert pinned in out["error"] and actual in out["error"]


def _fake_restore(monkeypatch, restore):
    module = types.ModuleType("quantbox_datasets.dataset")
    module.resolve_pinned_dataset = restore
    monkeypatch.setitem(sys.modules, "quantbox_datasets", types.ModuleType("quantbox_datasets"))
    monkeypatch.setitem(sys.modules, "quantbox_datasets.dataset", module)


def test_a_pin_restored_from_git_history_resolves_to_the_restored_build(root, tmp_path, monkeypatch):
    """quantbox-datasets ADR-0004: a moved-on checkout still serves the pinned build."""
    pinned = _build(tmp_path / "history", "etf-daily")  # the build the lock names
    (root / "etf-daily").mkdir(parents=True)
    pd.DataFrame({"AAA": [1.0]}, index=pd.date_range("2025-01-01", periods=1)).to_parquet(
        root / "etf-daily" / "prices.parquet"
    )  # the checkout has moved on
    (tmp_path / "datasets.lock").write_text(f"etf-daily: {pinned}\n")
    _fake_restore(monkeypatch, lambda path, sha: tmp_path / "history" / "etf-daily")
    monkeypatch.chdir(tmp_path)

    code, out = _resolve_json("etf-daily")

    assert code == 0
    assert out["matches"] is True and out["restored"] is True
    assert out["path"] == str(tmp_path / "history" / "etf-daily")
    assert out["actual_sha256"] == pinned


def test_a_failed_restore_carries_its_reason_into_the_mismatch(root, tmp_path, monkeypatch):
    actual = _build(root, "etf-daily")
    pinned = "b" * 64
    (tmp_path / "datasets.lock").write_text(f"etf-daily: {pinned}\n")

    def restore(path, sha):
        raise RuntimeError("no commit records that hash")

    _fake_restore(monkeypatch, restore)
    monkeypatch.chdir(tmp_path)

    code, out = _resolve_json("etf-daily")

    assert code == 1
    assert pinned in out["error"] and actual in out["error"]
    assert "no commit records that hash" in out["error"]


def test_a_pin_yaml_reads_as_a_number_is_refused_not_treated_as_unpinned(root, tmp_path, monkeypatch):
    _build(root, "etf-daily")
    (tmp_path / "datasets.lock").write_text(f"etf-daily: {'0' * 64}\n")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["dataset", "resolve", "etf-daily", "--json"])

    assert result.exit_code == 1
    assert "not a mapping of dataset name to sha256" in result.output


def test_resolve_without_a_root_says_so(tmp_path, monkeypatch):
    monkeypatch.delenv("QUANTBOX_DATASETS_ROOT", raising=False)
    monkeypatch.setitem(sys.modules, "quantbox_datasets.lock", None)  # not installed
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["dataset", "resolve", "etf-daily", "--json"])

    assert result.exit_code == 1
    assert "QUANTBOX_DATASETS_ROOT" in result.output


def _config(artifacts: Path) -> dict:
    return yaml.safe_load(f"""
run: {{mode: backtest, asof: "2024-02-28", pipeline: backtest.pipeline.v1}}
artifacts: {{root: "{artifacts}"}}
plugins:
  pipeline:
    name: backtest.pipeline.v1
    params:
      fees: 0.0
      venue: {{allow_shorts: false}}
      universe: {{symbols: [AAA, BBB]}}
  strategies:
    - name: strategy.cross_asset_momentum.v1
      weight: 1.0
      params: {{windows: [5, 10], long_only: true, position_size: 0.5}}
  data:
    name: local_file_data
    params_init: {{dataset: etf-daily}}
""")


def _run(config_path: Path, artifacts: Path) -> dict:
    cfg = _config(artifacts)
    config_path.write_text(yaml.safe_dump(cfg))
    result = run_from_config(cfg, PluginRegistry.discover(), config_path=config_path)
    return json.loads((artifacts / result.run_id / "run_manifest.json").read_text())


def test_same_config_from_lab_root_and_worktree_resolves_the_same_sha(root, tmp_path, monkeypatch, fake_datasets):
    sha = _build(root, "etf-daily")
    lab = tmp_path / "lab"
    worktree = lab / ".claude" / "worktrees" / "wt"
    for checkout in (lab, worktree):
        (checkout / "lines" / "x").mkdir(parents=True)
        (checkout / "datasets.lock").write_text(f"etf-daily: {sha}\n")

    monkeypatch.chdir(lab)
    from_lab = _run(lab / "lines" / "x" / "config.yaml", tmp_path / "a1")
    monkeypatch.chdir(worktree)
    from_worktree = _run(worktree / "lines" / "x" / "config.yaml", tmp_path / "a2")

    assert from_lab["dataset"]["resolved"]["sha256"] == sha
    assert from_worktree["dataset"]["resolved"]["sha256"] == sha
    assert from_lab["dataset"]["resolved"]["path"] == from_worktree["dataset"]["resolved"]["path"]
    # The worktree's own lock pinned it, not the lab root's.
    assert from_worktree["dataset"]["resolved"]["lock"] == str(worktree / "datasets.lock")
    assert [kw["root"] for _, kw in fake_datasets] == [str(root), str(root)]


def test_resolve_json_matches_the_manifest_dataset_block(root, tmp_path, monkeypatch, fake_datasets):
    sha = _build(root, "etf-daily")
    (tmp_path / "datasets.lock").write_text(f"etf-daily: {sha}\n")
    monkeypatch.chdir(tmp_path)

    manifest = _run(tmp_path / "config.yaml", tmp_path / "artifacts")
    code, out = _resolve_json("etf-daily")

    assert code == 0
    assert manifest["dataset"]["tier"] == "lock"
    assert manifest["dataset"]["resolved"] == out


def test_run_fails_loudly_on_a_sha_mismatch_naming_both(root, tmp_path, monkeypatch, fake_datasets):
    actual = _build(root, "etf-daily")
    pinned = "f" * 64
    (tmp_path / "datasets.lock").write_text(f"etf-daily: {pinned}\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(Exception) as exc:
        _run(tmp_path / "config.yaml", tmp_path / "artifacts")

    assert pinned in str(exc.value) and actual in str(exc.value)
    assert fake_datasets == []  # never read the unpinned bytes


def test_runner_resolves_the_lock_next_to_the_config_not_the_cwd(root, tmp_path, monkeypatch, fake_datasets):
    sha = _build(root, "etf-daily")
    line = tmp_path / "line"
    line.mkdir()
    (line / "datasets.lock").write_text(f"etf-daily: {sha}\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "datasets.lock").write_text(f"etf-daily: {'e' * 64}\n")
    monkeypatch.chdir(elsewhere)

    manifest = _run(line / "config.yaml", tmp_path / "artifacts")

    assert manifest["dataset"]["resolved"]["lock"] == str(line / "datasets.lock")
    assert manifest["dataset"]["resolved"]["matches"] is True


def test_an_unpinned_dataset_is_recorded_raw_and_strict_mode_is_unchanged(root, tmp_path, monkeypatch, fake_datasets):
    _build(root, "etf-daily")
    monkeypatch.chdir(tmp_path)  # no datasets.lock anywhere above

    manifest = _run(tmp_path / "config.yaml", tmp_path / "artifacts")
    assert manifest["dataset"]["tier"] == "raw"
    assert manifest["dataset"]["resolved"]["sha256"] is None

    # Strict mode rejected every by-name dataset before TOM-1349; whether a lock pin is
    # enough evidence for it is a separate decision, so a pinned one is still rejected.
    (tmp_path / "datasets.lock").write_text(f"etf-daily: {_sha(root / 'etf-daily' / 'prices.parquet')}\n")
    cfg = _config(tmp_path / "strict")
    cfg["run"]["strict"] = True
    with pytest.raises(RuntimeError, match="strict mode"):
        run_from_config(cfg, PluginRegistry.discover(), config_path=tmp_path / "config.yaml")


class _InlinePinnedData:
    """The deprecated style: an inline root + expected_prices_sha256 (dataset.curated.v1's params)."""

    def __init__(self, dataset_root: str, dataset: str, expected_prices_sha256: str = "") -> None:
        self.path = Path(dataset_root) / dataset

    def load_universe(self, params):
        return pd.DataFrame({"symbol": ["AAA", "BBB"]})

    def load_market_data(self, universe, asof, params):
        prices = pd.read_parquet(self.path / "prices.parquet")
        prices.index = pd.to_datetime(prices.index, utc=True)
        return {"prices": prices[prices.index <= pd.Timestamp(asof, tz="UTC")]}

    def load_fx(self, asof, params):
        return None


def test_deprecated_inline_pin_still_runs_and_warns_once(root, tmp_path):
    sha = _build(root, "etf-daily")
    cfg = _config(tmp_path / "artifacts")
    cfg["plugins"]["data"] = {
        "name": "test.inline_pinned.v1",
        "params_init": {"dataset_root": str(root), "dataset": "etf-daily", "expected_prices_sha256": sha},
    }
    reg = PluginRegistry.discover()
    reg.data["test.inline_pinned.v1"] = _InlinePinnedData

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_from_config(cfg, reg)

    pin_warnings = [w for w in caught if "datasets.lock" in str(w.message)]
    assert len(pin_warnings) == 1
    assert issubclass(pin_warnings[0].category, FutureWarning)  # shown to users by default
    assert result.metrics


def test_resolve_does_not_depend_on_a_copy_of_the_root(root, tmp_path, monkeypatch):
    """Moving the line elsewhere changes nothing: the root comes from the environment."""
    sha = _build(root, "etf-daily")
    for where in ("a", "b"):
        (tmp_path / where).mkdir()
        (tmp_path / where / "datasets.lock").write_text(f"etf-daily: {sha}\n")
    shutil.copytree(root, tmp_path / "a" / "datasets")  # a stray sibling copy is ignored

    outs = []
    for where in ("a", "b"):
        monkeypatch.chdir(tmp_path / where)
        outs.append(_resolve_json("etf-daily")[1]["path"])

    assert outs == [str(root / "etf-daily")] * 2


def test_a_restored_pin_is_what_the_run_reads_not_the_moved_on_checkout(root, tmp_path, monkeypatch):
    """PR #212 review: after a restore the run must read the pinned build, not $ROOT/<name>.

    The stub keeps the real library's contract: ``quantbox_datasets.lock.load(name, root=,
    sha256=)`` calls ``load_dataset``, which, given a sha, serves
    ``resolve_pinned_dataset(<root>/<name>, sha)`` — the build restored from git history.
    So load() must receive the env root AND the pin; a restored ``path`` is a cache entry
    named by the sha, not a root, and is never passed.
    """
    pinned = _build(tmp_path / "history", "etf-daily")
    (root / "etf-daily").mkdir(parents=True)
    pd.DataFrame({"AAA": [1.0]}, index=pd.date_range("2025-01-01", periods=1, name="date")).to_parquet(
        root / "etf-daily" / "prices.parquet"
    )  # the checkout has moved on
    (tmp_path / "datasets.lock").write_text(f"etf-daily: {pinned}\n")
    _fake_restore(monkeypatch, lambda path, sha: tmp_path / "history" / "etf-daily")

    calls: list[tuple[str, dict]] = []
    module = types.ModuleType("quantbox_datasets.lock")

    def load(name, **kwargs):
        calls.append((name, kwargs))
        path = Path(kwargs["root"]) / name
        if kwargs.get("sha256"):
            path = Path(sys.modules["quantbox_datasets.dataset"].resolve_pinned_dataset(path, kwargs["sha256"]))
        return _ParquetDataset(path)

    module.load = load
    monkeypatch.setitem(sys.modules, "quantbox_datasets.lock", module)
    monkeypatch.chdir(tmp_path)

    from quantbox.plugins.datasources.local_file_data import _load_pinned_dataset

    dataset, resolved = _load_pinned_dataset("etf-daily")

    assert resolved["restored"] is True and resolved["path"] == str(tmp_path / "history" / "etf-daily")
    assert calls == [("etf-daily", {"root": str(root), "sha256": pinned, "pinned": False})]
    pd.testing.assert_frame_equal(
        dataset.prices, pd.read_parquet(tmp_path / "history" / "etf-daily" / "prices.parquet")
    )

    manifest = _run(tmp_path / "config.yaml", tmp_path / "artifacts")
    assert manifest["dataset"]["resolved"]["path"] == str(tmp_path / "history" / "etf-daily")
    assert manifest["dataset"]["resolved"]["actual_sha256"] == pinned
    assert calls[-1][1]["sha256"] == pinned


@pytest.mark.parametrize("name", ["../escape", "..", ".", "/abs/escape", "a/b", "a\\b", ""])
def test_a_dataset_name_that_is_not_one_directory_under_the_root_is_refused(root, tmp_path, monkeypatch, name):
    """PR #212 review: a name must not reach outside $QUANTBOX_DATASETS_ROOT."""
    _build(tmp_path, "escape")  # a real dataset just outside the root
    _build(root / "a", "b")  # and one nested a level down
    monkeypatch.chdir(tmp_path)

    from quantbox.dataset_lock import DatasetResolveError, resolve_dataset

    with pytest.raises(DatasetResolveError, match="dataset name"):
        resolve_dataset(name)
    result = CliRunner().invoke(app, ["dataset", "resolve", name, "--json"])
    assert result.exit_code == 1
