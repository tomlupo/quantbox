"""duckdb is the [data] extra: each place that branches on it is loud or equivalent (TOM-1451).

Since 4b-2 a base install has no duckdb. Every module that chose a duckdb path
when duckdb was installed, and another path silently when it was not, is one of:

- **equivalent**: both paths give the same frame on the same input. The fallback
  stays and logs once, at INFO, which path ran. Tested here on fixtures:
  ``local_file_data`` (prices, volume, market cap, FX) and
  ``FileArtifactStore.query_artifacts``.
- **not equivalent**: the two paths can give a different result. The absence is
  then loud: ``MissingExtraError`` naming ``[data]``. That is the universe
  selection (``quantbox.universe.select_universe_duckdb`` and
  ``strategy.crypto_trend.v1``): the duckdb path ranks market cap only within the
  price columns and uses it unfilled; ``select_universe`` ranks the whole
  market-cap frame, forward-filled. The fixtures below show both differences.

The absence checks run in a subprocess whose import system refuses duckdb (the
``test_without_vectorbt`` harness, which first proves the block is live).
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest
from test_base_install import NOT_IN_BASE
from test_without_vectorbt import _run

import quantbox.plugins.datasources.local_file_data as lfd
from quantbox import store as store_mod
from quantbox.plugins.datasources.local_file_data import LocalFileDataPlugin
from quantbox.store import FileArtifactStore
from quantbox.universe import select_universe, select_universe_duckdb

pytest.importorskip("duckdb")

KW = dict(top_by_mcap=30, top_by_volume=10, exclude_tickers=[])


def _panel(n_dates: int = 60, n_coins: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    cols = [f"C{i:02d}" for i in range(n_coins)]
    prices = pd.DataFrame(rng.uniform(1, 100, (n_dates, n_coins)), index=dates, columns=cols)
    volume = pd.DataFrame(rng.uniform(1e3, 1e6, (n_dates, n_coins)), index=dates, columns=cols)
    mcap = pd.DataFrame(rng.uniform(1e6, 1e9, (n_dates, n_coins)), index=dates, columns=cols)
    return prices, volume, mcap


# --------------------------------------------------------------------------
# Universe selection: NOT equivalent, so a missing duckdb is loud
# --------------------------------------------------------------------------


def test_the_two_universe_paths_agree_on_a_clean_aligned_panel():
    prices, volume, mcap = _panel()
    duck = select_universe_duckdb(prices, volume, mcap, **KW)
    pd.testing.assert_frame_equal(duck.reindex_like(prices), select_universe(prices, volume, mcap, **KW))


@pytest.mark.parametrize("case", ["off_venue_mcap", "month_end_mcap"])
def test_the_two_universe_paths_differ_on_real_market_cap_shapes(case):
    prices, volume, mcap = _panel()
    if case == "off_venue_mcap":
        # A market-cap source ranks the whole market: coins the venue does not list.
        rng = np.random.default_rng(1)
        extra = pd.DataFrame(
            rng.uniform(5e8, 2e9, (len(mcap), 8)), index=mcap.index, columns=[f"X{i}" for i in range(8)]
        )
        mcap = pd.concat([mcap, extra], axis=1)
    else:
        # The curated datasets store market cap at month end only.
        mcap = mcap.resample("ME").last()
    duck = select_universe_duckdb(prices, volume, mcap, **KW).reindex_like(prices)
    vec = select_universe(prices, volume, mcap, **KW)
    assert int((duck != vec).sum().sum()) > 0, "the paths agree here; the loud refusal would be unneeded"


def test_select_universe_duckdb_without_duckdb_names_the_data_extra():
    proc = _run(
        """
        import json
        import pandas as pd
        from quantbox.exceptions import MissingExtraError
        from quantbox.universe import DUCKDB_AVAILABLE, select_universe, select_universe_duckdb

        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        prices = pd.DataFrame(1.0, index=dates, columns=["A", "B"])
        volume = pd.DataFrame(1.0, index=dates, columns=["A", "B"])
        mcap = pd.DataFrame({"A": [2.0] * 3, "B": [1.0] * 3}, index=dates)
        out = {"duckdb": DUCKDB_AVAILABLE}
        try:
            select_universe_duckdb(prices, volume, mcap, 1, 1, [])
            out["with_mcap"] = "ran"
        except MissingExtraError as exc:
            out["with_mcap"] = exc.extra
        # No market cap: the duckdb path never applies, select_universe answers.
        no_mcap = select_universe_duckdb(prices, volume, None, 1, 1, [])
        out["no_mcap_equal"] = no_mcap.equals(select_universe(prices, volume, None, 1, 1, []))
        print(json.dumps(out))
        """,
        blocked=NOT_IN_BASE,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == {"duckdb": False, "with_mcap": "data", "no_mcap_equal": True}


def test_crypto_trend_without_duckdb_names_the_data_extra_unless_told_to_use_pandas():
    proc = _run(
        """
        import json
        import numpy as np
        import pandas as pd
        from quantbox.exceptions import MissingExtraError
        from quantbox.plugins.strategies.crypto_trend import CryptoTrendStrategy

        rng = np.random.default_rng(0)
        dates = pd.date_range("2023-01-01", periods=400, freq="D")
        cols = [f"C{i}" for i in range(6)]
        prices = pd.DataFrame(np.exp(rng.normal(0, 0.02, (400, 6)).cumsum(axis=0)) * 100, index=dates, columns=cols)
        volume = pd.DataFrame(rng.uniform(1e3, 1e6, (400, 6)), index=dates, columns=cols)
        mcap = pd.DataFrame(rng.uniform(1e6, 1e9, (400, 6)), index=dates, columns=cols)
        data = {"prices": prices, "volume": volume, "market_cap": mcap}
        # Base-asset volume (volume_is_dollar: false), as quantbox-live's Kraken
        # book runs it: the one shape that takes the duckdb path.
        kw = dict(top_by_mcap=5, top_by_volume=3, exclude_tickers=[], volume_is_dollar=False)
        out = {}
        try:
            CryptoTrendStrategy(**kw).run(data)
            out["duckdb_path"] = "ran"
        except MissingExtraError as exc:
            out["duckdb_path"] = exc.extra
        out["use_duckdb_false"] = "weights" in CryptoTrendStrategy(**kw, use_duckdb=False).run(data)
        # The default (dollar volume) never took the duckdb path, so it runs.
        out["default"] = "weights" in CryptoTrendStrategy(top_by_mcap=5, top_by_volume=3, exclude_tickers=[]).run(data)
        print(json.dumps(out))
        """,
        blocked=NOT_IN_BASE,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == {"duckdb_path": "data", "use_duckdb_false": True, "default": True}


# --------------------------------------------------------------------------
# local_file_data: equivalent, so the pandas fallback stays and logs once
# --------------------------------------------------------------------------


@pytest.fixture
def frames(tmp_path):
    dates = pd.date_range("2024-01-01", periods=10, freq="D", name="date")
    wide = pd.DataFrame(np.arange(30, dtype=float).reshape(10, 3), index=dates, columns=["BTC", "ETH", "SOL"])
    long = wide.reset_index().melt(id_vars="date", var_name="symbol", value_name="close")
    files = {
        # A wide frame saved with its date INDEX: the pandas path once skipped
        # the asof cut on it (it looked only for a date column), a look-ahead.
        "wide_date_index.parquet": lambda p: wide.to_parquet(p),
        "wide_date_column.parquet": lambda p: wide.reset_index().to_parquet(p, index=False),
        "wide.csv": lambda p: wide.reset_index().to_csv(p, index=False),
        "long.parquet": lambda p: long.to_parquet(p, index=False),
        "long.csv": lambda p: long.to_csv(p, index=False),
    }
    paths = {}
    for name, write in files.items():
        write(tmp_path / name)
        paths[name] = str(tmp_path / name)
    return paths


@pytest.mark.parametrize(
    ("asof", "symbols"), [(None, None), ("2024-01-05", None), ("2024-01-05", ["BTC", "SOL"])], ids=str
)
@pytest.mark.parametrize(
    "name", ["wide_date_index.parquet", "wide_date_column.parquet", "wide.csv", "long.parquet", "long.csv"]
)
def test_local_file_reads_the_same_frame_with_and_without_duckdb(frames, monkeypatch, name, asof, symbols):
    monkeypatch.setattr(lfd, "DUCKDB_AVAILABLE", True)
    with_duckdb = lfd._read_file(frames[name], asof=asof, symbols=symbols)
    monkeypatch.setattr(lfd, "DUCKDB_AVAILABLE", False)
    with_pandas = lfd._read_file(frames[name], asof=asof, symbols=symbols)
    assert len(with_duckdb) == (5 if asof else 10)
    # The same instants; duckdb hands back microsecond timestamps, pandas nanosecond.
    pd.testing.assert_frame_equal(with_duckdb, with_pandas, check_dtype=False, check_freq=False, check_index_type=False)


@pytest.mark.parametrize("name", ["wide_date_index.parquet", "wide_date_column.parquet", "wide.csv"])
def test_load_fx_reads_the_same_frame_with_and_without_duckdb(frames, monkeypatch, name):
    plugin = LocalFileDataPlugin()
    monkeypatch.setattr(lfd, "DUCKDB_AVAILABLE", True)
    with_duckdb = plugin.load_fx("2024-01-05", {"fx_path": frames[name]})
    monkeypatch.setattr(lfd, "DUCKDB_AVAILABLE", False)
    with_pandas = plugin.load_fx("2024-01-05", {"fx_path": frames[name]})
    assert len(with_duckdb) == 5
    # check_like: duckdb puts a saved index column last, pandas' reset_index first.
    pd.testing.assert_frame_equal(with_duckdb, with_pandas, check_dtype=False, check_like=True)


@pytest.mark.parametrize(("available", "word"), [(True, "duckdb"), (False, "pandas")])
def test_local_file_logs_once_at_info_which_reader_ran(frames, monkeypatch, caplog, available, word):
    monkeypatch.setattr(lfd, "DUCKDB_AVAILABLE", available)
    monkeypatch.setattr(lfd, "_READER_LOGGED", False)
    with caplog.at_level(logging.INFO, logger=lfd.logger.name):
        lfd._read_file(frames["wide.csv"])
        lfd._read_file(frames["long.csv"])
    notes = [r for r in caplog.records if "reads files with" in r.getMessage()]
    assert len(notes) == 1
    assert notes[0].levelno == logging.INFO
    assert f"reads files with {word}" in notes[0].getMessage()


# --------------------------------------------------------------------------
# The same class beyond duckdb: hmmlearn absent used to mean an all-cash book
# --------------------------------------------------------------------------


def test_hmm_regime_without_hmmlearn_refuses_instead_of_returning_zero_weights():
    proc = _run(
        """
        import json
        import numpy as np
        import pandas as pd
        from quantbox.plugins.strategies.hmm_regime_allocation import HmmRegimeAllocation

        dates = pd.date_range("2024-01-01", periods=300, freq="D")
        prices = pd.DataFrame(np.linspace(100, 200, 300), index=dates, columns=["BTC"])
        try:
            out = HmmRegimeAllocation().run({"prices": prices}, {})
            print(json.dumps({"result": "ran", "keys": sorted(out)}))
        except ImportError as exc:
            print(json.dumps({"result": "refused", "message": str(exc)}))
        """,
        blocked=("hmmlearn",),
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["result"] == "refused", out
    assert "hmmlearn" in out["message"]


# --------------------------------------------------------------------------
# FileArtifactStore.query_artifacts: equivalent, so the pandas fallback stays
# --------------------------------------------------------------------------


def _three_runs(root):
    for i in range(3):
        s = FileArtifactStore(str(root), f"run_{i}")
        s.put_parquet(
            "weights",
            pd.DataFrame(
                {
                    "date": pd.to_datetime([f"2026-01-0{i + 1}"] * 2),
                    "symbol": ["BTC", "ETH"],
                    "weight": [0.1 * (i + 1), 0.2 * (i + 1)],
                }
            ),
        )
        s.put_json("run_manifest", {"run_id": f"run_{i}", "pipeline_name": "t", "mode": "paper", "asof": "2026-01-01"})


def test_query_artifacts_returns_the_same_rows_with_and_without_duckdb(tmp_path, monkeypatch, caplog):
    _three_runs(tmp_path)
    monkeypatch.setattr(store_mod, "_ACCELERATOR_LOGGED", False)
    with caplog.at_level(logging.INFO, logger=store_mod.logger.name):
        with_duckdb = FileArtifactStore.query_artifacts(str(tmp_path), "weights")

    def no_duckdb(target, *, extra=None):
        from quantbox.exceptions import MissingExtraError

        raise MissingExtraError(extra, target, "duckdb")

    monkeypatch.setattr(store_mod, "load", no_duckdb)
    monkeypatch.setattr(store_mod, "_ACCELERATOR_LOGGED", False)
    with caplog.at_level(logging.INFO, logger=store_mod.logger.name):
        with_pandas = FileArtifactStore.query_artifacts(str(tmp_path), "weights")
        FileArtifactStore.query_artifacts(str(tmp_path), "weights")

    assert len(with_duckdb) == 6
    pd.testing.assert_frame_equal(with_duckdb, with_pandas, check_dtype=False)
    notes = [r.getMessage() for r in caplog.records if "query_artifacts reads with" in r.getMessage()]
    assert [n.split("reads with ")[1].split()[0] for n in notes] == ["duckdb", "pandas"]
