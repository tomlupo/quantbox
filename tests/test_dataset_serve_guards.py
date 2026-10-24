"""Replays of the three defects review round 1 of promotion PR #303 found in by-name serving (TOM-2232).

Each test replays a reviewer's probe through the real resolution and serve path
(``resolve_dataset`` → ``require_match`` → ``served_prices``) and must be refused:

1. a malformed or non-finite return at a series' edge became "not yet listed" history,
   or an infinite price;
2. the served table was re-read after its hash was checked, so a replaced file served
   under the resolution's original hash;
3. a manifest-named file that is a symlink out of the dataset directory was hashed
   and served.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quantbox.dataset import served_prices
from quantbox.dataset_lock import DatasetResolveError, DatasetServeError, require_match, resolve_dataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _no_root():
    raise DatasetResolveError("no datasets root")


@pytest.fixture
def line(tmp_path, monkeypatch):
    """A research line with no datasets root: only its own frozen copy under datasets/."""
    monkeypatch.delenv("QUANTBOX_DATASETS_ROOT", raising=False)
    monkeypatch.setattr("quantbox.dataset_lock.datasets_root", _no_root)
    line = tmp_path / "line"
    line.mkdir()
    return line


def _freeze(line: Path, table: pd.DataFrame, serve: dict, *, pin_file: str = "monthly", extra: dict | None = None):
    """A frozen line-local dataset ``kf`` (the table as its pin file) and its lock; returns its directory."""
    ds = line / "datasets" / "kf"
    ds.mkdir(parents=True)
    table.to_parquet(ds / f"{pin_file}.parquet")
    hashes = {f"{pin_file}.parquet": {"sha256": _sha(ds / f"{pin_file}.parquet")}}
    hashes.update(extra or {})
    manifest = {"name": "kf", "refresh": "frozen", "pin_file": f"{pin_file}.parquet", "content_hashes": hashes}
    manifest["serve"] = serve
    (ds / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    (line / "datasets.lock").write_text(f"kf: {hashes[f'{pin_file}.parquet']['sha256']}\n")
    return ds


def _serve(line: Path) -> pd.DataFrame:
    return served_prices(require_match(resolve_dataset("kf", lock=line / "datasets.lock")))


MONTHS = pd.date_range("2000-01-31", periods=3, freq="ME", name="date")
MONTHLY_RETURNS = {"table": "monthly", "kind": "returns", "frequency": "monthly"}


# -- 1. a malformed or non-finite value is refused, not read as missing history ----------


@pytest.mark.parametrize(
    ("kind", "values", "match"),
    [
        ("returns", ["bad", "0.1", "0.2"], r"'EQ' has a value that is not a number on \['2000-01-31'\]"),
        ("returns", ["0.1", "0.2", "bad"], r"'EQ' has a value that is not a number on \['2000-03-31'\]"),
        ("returns", [0.1, np.inf, 0.2], r"'EQ' has a non-finite value on \['2000-02-29'\]"),
        ("prices", [10.0, -np.inf, 12.0], r"'EQ' has a non-finite value on \['2000-02-29'\]"),
        ("prices", ["bad", "11", "12"], r"'EQ' has a value that is not a number on \['2000-01-31'\]"),
        ("prices", [10.0, 11.0, np.inf], r"'EQ' has a non-finite value on \['2000-03-31'\]"),
    ],
)
def test_a_malformed_or_non_finite_value_is_refused_not_served(line, kind, values, match):
    _freeze(line, pd.DataFrame({"EQ": values}, index=MONTHS), {"table": "monthly", "kind": kind})
    with pytest.raises(DatasetServeError, match=match):
        _serve(line)


@pytest.mark.parametrize(
    ("kind", "values"),
    [
        ("returns", [True, False, True]),  # was served as +100 % / 0 % returns
        ("prices", list(MONTHS)),  # was served as epoch nanoseconds
    ],
)
def test_a_column_pandas_converts_without_being_numbers_is_refused(line, kind, values):
    _freeze(line, pd.DataFrame({"EQ": values}, index=MONTHS), {"table": "monthly", "kind": kind})
    with pytest.raises(DatasetServeError, match="'EQ' holds .* values, not numbers"):
        _serve(line)


def test_a_served_funding_file_with_an_infinity_is_refused(line):
    from quantbox.plugins.datasources.local_file_data import LocalFileDataPlugin

    ds = _freeze(line, pd.DataFrame({"EQ": [10.0, 11.0, 12.0]}, index=MONTHS), {"table": "monthly", "kind": "prices"})
    pd.DataFrame({"EQ": [0.001, np.inf, 0.001]}, index=MONTHS).to_parquet(ds / "funding_rates.parquet")
    plugin = LocalFileDataPlugin(dataset="kf", dataset_lock=str(line / "datasets.lock"))
    with pytest.raises(DatasetServeError, match=r"funding_rates series 'EQ' has a non-finite value"):
        plugin.load_market_data(plugin.load_universe({}), "2000-03-31", {})


# -- independent review round 1 of #305: the value rule runs on the RAW values ------------


def _served_funding(line: Path, funding: pd.DataFrame, *, index: bool = True):
    from quantbox.plugins.datasources.local_file_data import LocalFileDataPlugin

    ds = _freeze(line, pd.DataFrame({"EQ": [10.0, 11.0, 12.0]}, index=MONTHS), {"table": "monthly", "kind": "prices"})
    funding.to_parquet(ds / "funding_rates.parquet", index=index)
    plugin = LocalFileDataPlugin(dataset="kf", dataset_lock=str(line / "datasets.lock"))
    return plugin.load_market_data(plugin.load_universe({}), "2000-03-31", {})


def test_a_nullable_bool_returns_column_is_refused_not_served_as_a_100_percent_return(line):
    # Parquet reads [None, True, False] back as object; coercion made it [nan, 1.0, 0.0] -> prices [1, 2, 2].
    _freeze(line, pd.DataFrame({"EQ": [None, True, False]}, index=MONTHS), MONTHLY_RETURNS)
    with pytest.raises(DatasetServeError, match=r"'EQ' holds bool values, not numbers, on \['2000-02-29'"):
        _serve(line)


def test_a_string_infinity_in_the_served_funding_file_is_refused(line):
    with pytest.raises(
        DatasetServeError, match=r"funding_rates series 'EQ' has a non-finite value on \['2000-02-29'\]"
    ):
        _served_funding(line, pd.DataFrame({"EQ": ["0.001", "inf", "0.001"]}, index=MONTHS))


def test_duplicate_plus_and_minus_infinity_in_a_long_funding_file_are_refused_before_the_pivot(line):
    # The pivot averages the two rows of 2000-02-29 to NaN: validated after it, the date just vanished.
    long = pd.DataFrame(
        {
            "date": [MONTHS[0], MONTHS[1], MONTHS[1], MONTHS[2]],
            "symbol": ["EQ"] * 4,
            "funding_rate": [0.001, np.inf, -np.inf, 0.001],
        }
    )
    with pytest.raises(DatasetServeError, match=r"funding_rates series 'funding_rate' has a non-finite value"):
        _served_funding(line, long, index=False)


def test_a_valid_funding_file_with_a_gap_is_still_served(line):
    wide = _served_funding(line, pd.DataFrame({"EQ": [0.001, None, 0.002]}, index=MONTHS))["funding_rates"]
    assert wide["EQ"].iloc[0] == 0.001 and np.isnan(wide["EQ"].iloc[1])


def test_a_null_at_a_series_edge_is_still_history_not_yet_listed(line):
    _freeze(line, pd.DataFrame({"EQ": [None, "0.1", "0.2"]}, index=MONTHS), MONTHLY_RETURNS)
    # The series starts at the second month: the first month is its 1.0 base, as before.
    np.testing.assert_allclose(_serve(line)["EQ"], [1.0, 1.1, 1.32])


# -- 2. the bytes served are the bytes the resolution hashed ---------------------------


def test_a_table_replaced_after_resolution_is_refused_not_served(line):
    ds = _freeze(line, pd.DataFrame({"EQ": [10.0, 11.0, 12.0]}, index=MONTHS), {"table": "monthly", "kind": "prices"})
    resolution = require_match(resolve_dataset("kf", lock=line / "datasets.lock"))
    assert resolution["matches"] is True
    pd.DataFrame({"EQ": [99.0, 98.0, 97.0]}, index=MONTHS).to_parquet(ds / "monthly.parquet")
    with pytest.raises(DatasetServeError, match="not the bytes it was resolved"):
        served_prices(resolution)


# -- 3. a manifest-named file stays inside the dataset directory -------------------------


def _outside(line: Path, frame: pd.DataFrame) -> Path:
    outside = line.parent / "elsewhere" / "outside.parquet"
    outside.parent.mkdir()
    frame.to_parquet(outside)
    return outside


def test_a_served_table_symlinked_out_of_the_dataset_is_refused(line):
    outside = _outside(line, pd.DataFrame({"EQ": [90.0, 91.0]}, index=MONTHS[:2]))
    ds = _freeze(
        line,
        pd.DataFrame({"EQ": [10.0, 11.0, 12.0]}, index=MONTHS),
        {"table": "external", "kind": "prices"},
        extra={"external.parquet": {"sha256": _sha(outside)}},
    )
    (ds / "external.parquet").symlink_to(outside)
    with pytest.raises(DatasetServeError, match="outside the dataset directory"):
        _serve(line)


def test_a_pin_file_symlinked_out_of_the_dataset_is_refused(line):
    outside = _outside(line, pd.DataFrame({"EQ": [90.0, 91.0]}, index=MONTHS[:2]))
    ds = line / "datasets" / "kf"
    ds.mkdir(parents=True)
    (ds / "monthly.parquet").symlink_to(outside)
    (ds / "manifest.yaml").write_text(yaml.safe_dump({"name": "kf", "pin_file": "monthly.parquet"}))
    (line / "datasets.lock").write_text(f"kf: {_sha(outside)}\n")
    with pytest.raises(DatasetResolveError, match="outside the dataset directory"):
        resolve_dataset("kf", lock=line / "datasets.lock")


def test_a_funding_file_symlinked_out_of_the_dataset_is_refused(line):
    outside = _outside(line, pd.DataFrame({"EQ": [0.001, 0.002]}, index=MONTHS[:2]))
    ds = _freeze(line, pd.DataFrame({"EQ": [10.0, 11.0, 12.0]}, index=MONTHS), {"table": "monthly", "kind": "prices"})
    (ds / "funding_rates.parquet").symlink_to(outside)
    with pytest.raises(DatasetResolveError, match="outside the dataset directory"):
        resolve_dataset("kf", lock=line / "datasets.lock")
