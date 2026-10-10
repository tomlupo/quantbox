"""local_file_data cuts at asof on every date shape, and both readers agree (TOM-1681).

#270 made the pandas reader cut a parquet file saved with an index NAMED ``date``.
A writer that leaves the index unnamed, or names it ``timestamp``, still got no
cut: a backtest on such a file read rows after its asof date. The duckdb reader
was no better on those files: it returned the index as a plain column, cut
nothing, and ``_read_file`` then built a 1970 epoch index from the row numbers.
And on a tz-aware column duckdb compared ``date <= 'asof'`` in the BOX's local
timezone, so the same file gave different rows on forge, core and CI.

The rule, for both readers and ``load_fx``: the date is the ``date`` column, or
else the saved datetime index (any name). A row is kept when its WALL-CLOCK time
in its own timezone is ``<= asof`` 00:00; a naive date is compared as is. #271
compared UTC instants, which dropped the asof-day bar of a daily file stamped at
local midnight west of UTC (New York 00:00 = 05:00 UTC); the wall-clock rule keeps
it. ``_read_file`` labels each row with its own calendar date at UTC midnight.

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
    """Rows on or before asof by each row's wall-clock time in its own timezone (TOM-1681).

    Until the wall-clock rule this counted UTC instants, which dropped the asof-day
    bar of a New York file (00:00 local = 05:00 UTC): the stated requirement change.
    """
    wall = dates.tz_localize(None) if dates.tz is not None else dates
    return int((wall <= pd.Timestamp(ASOF)).sum())


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
        dates_read = pd.DatetimeIndex(frame["date"])
        wall = dates_read.tz_localize(None) if dates_read.tz is not None else dates_read
        assert (wall <= pd.Timestamp(ASOF)).all(), "a row after asof"
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


# --- The wall-clock rule, case by case (TOM-1681) -----------------------------


def _ns(frame: pd.DataFrame) -> pd.DataFrame:
    """duckdb hands back microsecond timestamps, pandas nanosecond: compare in ns."""
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(frame.index).as_unit("ns")
    frame.index.freq = None
    return frame


def _both_readers(monkeypatch, fn) -> dict[str, object]:
    """*fn* through duckdb (its silent fallback to pandas refused) and through pandas."""
    with monkeypatch.context() as m:

        def no_fallback(*args, **kwargs):
            raise AssertionError("duckdb fell back to the pandas reader")

        m.setattr(lfd, "_read_via_pandas", no_fallback)
        duck = _with_duckdb(m, fn)
    return {"duckdb": duck, "pandas": _with_pandas(monkeypatch, fn)}


def _daily(tz, periods=10, start="2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="D", name="date", tz=tz)
    return pd.DataFrame(np.arange(periods * 2, dtype=float).reshape(periods, 2), index=idx, columns=["BTC", "ETH"])


def _expected_daily(n: int) -> pd.DataFrame:
    """The first *n* rows of ``_daily``, labelled by their own calendar date at UTC midnight."""
    expected = _daily(None).iloc[:n].copy()
    expected.index = expected.index.tz_localize("UTC")
    return _ns(expected)


@pytest.mark.parametrize("tz", ["America/New_York", "Europe/Warsaw", "UTC", None], ids=str)
def test_a_midnight_daily_file_keeps_the_asof_bar_and_drops_the_next(tmp_path, monkeypatch, tokyo_duckdb, tz):
    """The bar of day D belongs to D, in any timezone: 5 rows, labelled Jan 1..Jan 5.

    At #271 a New York file lost its Jan 5 bar (00:00 local = 05:00 UTC > asof 00:00
    UTC) and a Warsaw file kept it but labelled every bar a day early (Jan 5 00:00
    local = Jan 4 23:00 UTC).
    """
    path = tmp_path / "daily.parquet"
    _daily(tz).to_parquet(path)
    for reader, frame in _both_readers(monkeypatch, lambda: lfd._read_file(str(path), asof=ASOF)).items():
        pd.testing.assert_frame_equal(_ns(frame), _expected_daily(5), obj=f"{reader} {tz}")


def test_an_intraday_tz_aware_file_keeps_rows_up_to_asof_midnight_local(tmp_path, monkeypatch, tokyo_duckdb):
    """Hourly New York bars: kept through Jan 5 00:00 New York time, the 01:00 bar is not."""
    idx = pd.date_range("2024-01-04", "2024-01-06", freq="h", name="date", tz="America/New_York")
    pd.DataFrame({"EURUSD": np.arange(len(idx), dtype=float)}, index=idx).to_parquet(tmp_path / "fx.parquet")
    plugin = LocalFileDataPlugin()
    out = _both_readers(monkeypatch, lambda: plugin.load_fx(ASOF, {"fx_path": str(tmp_path / "fx.parquet")}))
    for reader, frame in out.items():
        dates = pd.DatetimeIndex(frame["date"])
        assert str(dates.tz) == "America/New_York", (reader, dates.tz)
        assert len(frame) == 25, (reader, len(frame))
        assert dates.max() == pd.Timestamp("2024-01-05 00:00", tz="America/New_York"), reader
        assert frame["EURUSD"].tolist() == [float(i) for i in range(25)], reader


def test_naive_daily_and_intraday_files_are_cut_as_before(tmp_path, monkeypatch, tokyo_duckdb):
    """A naive date is compared as is: the exact frames #271 and v0.11.0 duckdb gave on a UTC box."""
    _daily(None).to_parquet(tmp_path / "daily.parquet")
    for reader, frame in _both_readers(
        monkeypatch, lambda: lfd._read_file(str(tmp_path / "daily.parquet"), asof=ASOF)
    ).items():
        pd.testing.assert_frame_equal(_ns(frame), _expected_daily(5), obj=f"{reader} naive daily")

    idx = pd.date_range("2024-01-04", "2024-01-06", freq="h", name="date")
    pd.DataFrame({"BTC": np.arange(len(idx), dtype=float)}, index=idx).to_parquet(tmp_path / "hourly.parquet")
    expected = pd.DataFrame(
        {"BTC": [float(i) for i in range(25)]},
        index=pd.DatetimeIndex(["2024-01-04"] * 24 + ["2024-01-05"], name="date").tz_localize("UTC"),
    )
    for reader, frame in _both_readers(
        monkeypatch, lambda: lfd._read_file(str(tmp_path / "hourly.parquet"), asof=ASOF)
    ).items():
        pd.testing.assert_frame_equal(_ns(frame), _ns(expected), obj=f"{reader} naive hourly")


def test_a_csv_with_mixed_offsets_cuts_each_row_on_its_own_wall_clock(tmp_path, monkeypatch, tokyo_duckdb):
    """New York Jan 5 00:00 and Warsaw Jan 5 00:00 are both on asof; Warsaw Jan 6 is not."""
    path = tmp_path / "fx.csv"
    path.write_text(
        "date,EURUSD\n"
        "2024-01-04 00:00:00-05:00,1\n"
        "2024-01-05 00:00:00-05:00,2\n"
        "2024-01-05 00:00:00+01:00,3\n"
        "2024-01-06 00:00:00+01:00,4\n"
        "2024-01-05 01:00:00+01:00,5\n"
    )
    plugin = LocalFileDataPlugin()
    for reader, frame in _both_readers(monkeypatch, lambda: plugin.load_fx(ASOF, {"fx_path": str(path)})).items():
        assert frame["EURUSD"].tolist() == [1, 2, 3], (reader, frame)


@pytest.mark.parametrize("tz", ["America/New_York", "Europe/Warsaw", "UTC", None], ids=str)
def test_the_dataset_path_labels_and_cuts_as_the_file_path_does(tz):
    """``_clip_frame`` (a by-name dataset) gives the same frame as ``_read_file``: Jan 1..Jan 5."""
    frame = lfd._clip_frame(_daily(tz), ASOF)
    pd.testing.assert_frame_equal(_ns(frame), _expected_daily(5), obj=str(tz))
