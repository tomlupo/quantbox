"""local_file_data cuts at asof on every date shape, and both readers agree (TOM-1681).

#270 made the pandas reader cut a parquet file saved with an index NAMED ``date``.
A writer that leaves the index unnamed, or names it ``timestamp``, still got no
cut: a backtest on such a file read rows after its asof date. The duckdb reader
was no better on those files: it returned the index as a plain column, cut
nothing, and ``_read_file`` then built a 1970 epoch index from the row numbers.
And on a tz-aware column duckdb compared ``date <= 'asof'`` in the BOX's local
timezone, so the same file gave different rows on forge, core and CI.

The rule, for both readers and ``load_fx``: the date is the ``date`` column, or
else the saved datetime index (any name). A row is kept when its UTC instant is
``<= asof`` 00:00 UTC (a naive date counts as UTC).

Each fixture shape runs through both readers. duckdb runs with its session in
Asia/Tokyo, so a cut that leans on the box's timezone shows on any box. The
pandas reader runs once in-process and once in a subprocess whose import system
refuses duckdb (a real base install).
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from test_base_install import NOT_IN_BASE
from test_without_vectorbt import _run

import quantbox.plugins.datasources.local_file_data as lfd
from quantbox.plugins.datasources.local_file_data import LocalFileDataPlugin

duckdb = pytest.importorskip("duckdb")

ASOF = "2024-01-05"
ASOF_UTC = pd.Timestamp(ASOF, tz="UTC")
TZS = [None, "UTC", "Europe/Warsaw", "America/New_York"]
SHAPES = ["index_date", "index_unnamed", "index_timestamp", "column_date", "multiindex_date_symbol"]


def _dates(tz, name="date"):
    return pd.date_range("2024-01-01", periods=10, freq="D", name=name, tz=tz)


def _write(shape: str, tz, path) -> pd.DatetimeIndex:
    """Write one fixture shape; return the dates it holds (rows after asof included)."""
    values = np.arange(30, dtype=float).reshape(10, 3)
    cols = ["BTC", "ETH", "SOL"]
    if shape == "index_date":
        pd.DataFrame(values, index=_dates(tz, "date"), columns=cols).to_parquet(path)
    elif shape == "index_unnamed":
        pd.DataFrame(values, index=_dates(tz, None), columns=cols).to_parquet(path)
    elif shape == "index_timestamp":
        pd.DataFrame(values, index=_dates(tz, "timestamp"), columns=cols).to_parquet(path)
    elif shape == "column_date":
        pd.DataFrame(values, index=_dates(tz), columns=cols).reset_index().to_parquet(path, index=False)
    elif shape == "multiindex_date_symbol":
        wide = pd.DataFrame(values, index=_dates(tz), columns=cols)
        long = wide.stack().rename("close")
        long.index.names = ["date", "symbol"]
        long.to_frame().to_parquet(path)
    else:
        raise AssertionError(shape)
    return _dates(tz)


def _kept(dates: pd.DatetimeIndex) -> int:
    utc = dates.tz_localize("UTC") if dates.tz is None else dates.tz_convert("UTC")
    return int((utc <= ASOF_UTC).sum())


@pytest.fixture
def tokyo_duckdb(monkeypatch):
    """duckdb whose session timezone is Asia/Tokyo, as on a box set to Tokyo."""
    real_connect = duckdb.connect

    def connect(*args, **kwargs):
        con = real_connect(*args, **kwargs)
        con.execute("SET TimeZone = 'Asia/Tokyo'")
        return con

    monkeypatch.setattr(lfd.duckdb, "connect", connect)


def _with_duckdb(monkeypatch, fn):
    monkeypatch.setattr(lfd, "DUCKDB_AVAILABLE", True)
    return fn()


def _with_pandas(monkeypatch, fn):
    monkeypatch.setattr(lfd, "DUCKDB_AVAILABLE", False)
    return fn()


def _assert_cut(frame: pd.DataFrame, dates: pd.DatetimeIndex) -> None:
    assert isinstance(frame.index, pd.DatetimeIndex), type(frame.index)
    assert str(frame.index.tz) == "UTC"
    assert frame.index.max() <= ASOF_UTC, f"a row after asof: {frame.index.max()}"
    # Not the 1970 epoch index that row numbers turn into.
    assert frame.index.min().year in (2023, 2024), frame.index.min()
    assert len(frame) == _kept(dates)
    assert list(frame.columns) == ["BTC", "ETH", "SOL"]


@pytest.mark.parametrize("tz", TZS, ids=str)
@pytest.mark.parametrize("shape", SHAPES)
def test_both_readers_cut_every_date_shape_at_asof(tmp_path, monkeypatch, tokyo_duckdb, shape, tz):
    path = tmp_path / f"{shape}.parquet"
    dates = _write(shape, tz, path)
    duck = _with_duckdb(monkeypatch, lambda: lfd._read_file(str(path), asof=ASOF))
    pand = _with_pandas(monkeypatch, lambda: lfd._read_file(str(path), asof=ASOF))
    _assert_cut(duck, dates)
    _assert_cut(pand, dates)
    # The same instants; duckdb hands back microsecond timestamps, pandas nanosecond.
    pd.testing.assert_frame_equal(duck, pand, check_dtype=False, check_freq=False, check_index_type=False)


@pytest.mark.parametrize("tz", TZS, ids=str)
@pytest.mark.parametrize("shape", SHAPES)
def test_both_readers_agree_with_no_asof(tmp_path, monkeypatch, tokyo_duckdb, shape, tz):
    path = tmp_path / f"{shape}.parquet"
    _write(shape, tz, path)
    duck = _with_duckdb(monkeypatch, lambda: lfd._read_file(str(path)))
    pand = _with_pandas(monkeypatch, lambda: lfd._read_file(str(path)))
    assert len(duck) == 10
    assert duck.index.min().year in (2023, 2024)
    pd.testing.assert_frame_equal(duck, pand, check_dtype=False, check_freq=False, check_index_type=False)


@pytest.mark.parametrize("tz", TZS, ids=str)
@pytest.mark.parametrize("shape", ["index_date", "index_unnamed", "index_timestamp", "column_date"])
def test_load_fx_cuts_every_date_shape_at_asof(tmp_path, monkeypatch, tokyo_duckdb, shape, tz):
    path = tmp_path / f"{shape}.parquet"
    dates = _write(shape, tz, path)
    plugin = LocalFileDataPlugin()
    duck = _with_duckdb(monkeypatch, lambda: plugin.load_fx(ASOF, {"fx_path": str(path)}))
    pand = _with_pandas(monkeypatch, lambda: plugin.load_fx(ASOF, {"fx_path": str(path)}))
    for frame in (duck, pand):
        assert "date" in frame.columns, list(frame.columns)
        assert (pd.to_datetime(frame["date"], utc=True) <= ASOF_UTC).all(), "a row after asof"
        assert len(frame) == _kept(dates)
    # check_like: duckdb puts a saved index column last, pandas' reset_index first.
    # The same instants: compare them in UTC, as duckdb may hand back the session timezone.
    duck = duck.assign(date=pd.to_datetime(duck["date"], utc=True))
    pand = pand.assign(date=pd.to_datetime(pand["date"], utc=True))
    pd.testing.assert_frame_equal(duck, pand, check_dtype=False, check_like=True)


def test_a_base_install_without_duckdb_reads_every_shape_as_duckdb_does(tmp_path, monkeypatch, tokyo_duckdb):
    """The pandas reader on a real base install (duckdb refused at import) matches duckdb."""
    paths = {}
    for shape in SHAPES:
        for tz in TZS:
            path = tmp_path / f"{shape}-{str(tz).replace('/', '_')}.parquet"
            _write(shape, tz, path)
            paths[str(path)] = str(path) + ".pkl"
    proc = _run(
        f"""
        import json, pickle
        import quantbox.plugins.datasources.local_file_data as lfd
        assert not lfd.DUCKDB_AVAILABLE
        for src, out in {paths!r}.items():
            with open(out, "wb") as fh:
                pickle.dump(lfd._read_file(src, asof={ASOF!r}), fh)
        print(json.dumps({{"read": len({paths!r})}}))
        """,
        blocked=NOT_IN_BASE,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == {"read": len(SHAPES) * len(TZS)}
    for src, out in paths.items():
        with open(out, "rb") as fh:
            base = pickle.load(fh)
        duck = _with_duckdb(monkeypatch, lambda src=src: lfd._read_file(src, asof=ASOF))
        assert base.index.max() <= ASOF_UTC, src
        pd.testing.assert_frame_equal(duck, base, check_dtype=False, check_freq=False, check_index_type=False)
