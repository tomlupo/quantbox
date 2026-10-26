"""A requested symbol that local_file_data cannot serve is refused, not dropped (TOM-2421).

Guards whose job is to fail. The incident (TOM-2410): a line config on ``dataset: sbbi``
asked for LargeCap_IR and LargeCap_CAR, which the raw monthly table holds but the
manifest's ``serve.columns`` did not list. The run got a prices frame with ZERO columns
and no error. The first test is that shape; the others are the same class on the
inline-path readers (wide and long) and on a funding table, which the engine would
otherwise charge as zero.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quantbox.exceptions import DataLoadError
from quantbox.plugins.datasources.local_file_data import LocalFileDataPlugin


def _no_root():
    from quantbox.dataset_lock import DatasetResolveError

    raise DatasetResolveError("no datasets root")


@pytest.fixture
def line(tmp_path, monkeypatch):
    """A research line with no datasets root: only its own frozen copy."""
    monkeypatch.delenv("QUANTBOX_DATASETS_ROOT", raising=False)
    monkeypatch.setattr("quantbox.dataset_lock.datasets_root", _no_root)
    line = tmp_path / "line"
    line.mkdir()
    return line


def _sbbi_like(line: Path) -> Path:
    """sbbi at quantbox-datasets base: a monthly returns table whose serve block lists 2 of its 4 columns."""
    dates = pd.date_range("2000-01-31", periods=48, freq="ME", name="date")
    rng = np.random.default_rng(7)
    table = pd.DataFrame(
        rng.normal(0.006, 0.04, size=(48, 4)),
        index=dates,
        columns=["LargeCap_TR", "TBill_30d", "LargeCap_IR", "LargeCap_CAR"],
    )
    ds = line / "datasets" / "sbbi"
    ds.mkdir(parents=True)
    table.to_parquet(ds / "monthly.parquet")
    sha = hashlib.sha256((ds / "monthly.parquet").read_bytes()).hexdigest()
    manifest = {
        "name": "sbbi",
        "refresh": "frozen",
        "source": "test",
        "pin_file": "monthly.parquet",
        "content_hashes": {"monthly.parquet": {"sha256": sha}},
        "serve": {
            "table": "monthly",
            "kind": "returns",
            "frequency": "monthly",
            "columns": ["LargeCap_TR", "TBill_30d"],
        },
    }
    (ds / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    (line / "datasets.lock").write_text(f"sbbi: {sha}\n")
    return line / "datasets.lock"


def test_a_column_the_serve_block_leaves_out_is_refused_by_name(line):
    plugin = LocalFileDataPlugin(dataset="sbbi", dataset_lock=str(_sbbi_like(line)))
    universe = plugin.load_universe({"symbols": ["LargeCap_IR", "LargeCap_CAR"]})
    with pytest.raises(
        DataLoadError, match=r"lacks 2 requested symbol\(s\) at or before 2003-12-31: \['LargeCap_IR', 'LargeCap_CAR'\]"
    ):
        plugin.load_market_data(universe, "2003-12-31", {})


def test_a_served_column_still_loads(line):
    plugin = LocalFileDataPlugin(dataset="sbbi", dataset_lock=str(_sbbi_like(line)))
    universe = plugin.load_universe({"symbols": ["LargeCap_TR"]})
    assert list(plugin.load_market_data(universe, "2003-12-31", {})["prices"].columns) == ["LargeCap_TR"]


def _prices(symbols: list[str]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="D", name="date")
    return pd.DataFrame(np.linspace(1.0, 2.0, 10)[:, None].repeat(len(symbols), 1), index=dates, columns=symbols)


@pytest.mark.parametrize("layout", ["wide", "long"])
def test_a_prices_file_without_a_requested_symbol_is_refused(tmp_path, layout):
    wide = _prices(["AAA", "BBB"])
    path = tmp_path / "prices.parquet"
    if layout == "wide":
        wide.to_parquet(path)
    else:
        wide.stack().rename("close").rename_axis(["date", "symbol"]).reset_index().to_parquet(path, index=False)
    plugin = LocalFileDataPlugin(prices_path=str(path))
    with pytest.raises(DataLoadError, match=r"\['CCC'\]") as err:
        plugin.load_market_data(pd.DataFrame({"symbol": ["AAA", "CCC"]}), "2024-01-10", {})
    assert err.value.details["missing"] == ["CCC"]


def test_a_funding_file_without_a_requested_symbol_is_refused(tmp_path):
    _prices(["AAA", "BBB"]).to_parquet(tmp_path / "prices.parquet")
    _prices(["AAA"]).mul(1e-4).to_parquet(tmp_path / "funding.parquet")
    plugin = LocalFileDataPlugin(
        prices_path=str(tmp_path / "prices.parquet"), funding_rates_path=str(tmp_path / "funding.parquet")
    )
    with pytest.raises(DataLoadError, match=r"funding_rates table .* \['BBB'\]"):
        plugin.load_market_data(pd.DataFrame({"symbol": ["AAA", "BBB"]}), "2024-01-10", {})


@pytest.mark.parametrize("layout", ["wide", "long"])
def test_a_funding_file_with_none_of_the_requested_symbols_is_refused(tmp_path, layout):
    """PR #314 review: a long file is filtered as it is read, so holding none of the symbols gave an empty frame."""
    _prices(["AAA"]).to_parquet(tmp_path / "prices.parquet")
    funding = _prices(["BBB"]).mul(1e-4)
    if layout == "wide":
        funding.to_parquet(tmp_path / "funding.parquet")
    else:
        funding.stack().rename("rate").rename_axis(["date", "symbol"]).reset_index().to_parquet(
            tmp_path / "funding.parquet", index=False
        )
    plugin = LocalFileDataPlugin(
        prices_path=str(tmp_path / "prices.parquet"), funding_rates_path=str(tmp_path / "funding.parquet")
    )
    with pytest.raises(DataLoadError, match=r"funding_rates table .* \['AAA'\]"):
        plugin.load_market_data(pd.DataFrame({"symbol": ["AAA"]}), "2024-01-10", {})
