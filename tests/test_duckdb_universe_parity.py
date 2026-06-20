"""Regression: select_universe_duckdb must be byte-identical to select_universe.

Guards the DuckDB universe NaN-leak fixed in fix/duckdb-universe-fix.

The DuckDB path emits one SQL row only for (date, ticker) pairs that clear the
day's top-``top_by_mcap`` cut. ``pivot()`` therefore leaves a *per-cell* NaN for
any coin that qualifies on SOME dates (so its column exists) but is excluded on
others. ``reindex(fill_value=0)`` fills only wholly-absent columns, never these
per-cell NaNs, so the excluded cell stayed NaN. Downstream
``construct_weights`` masks with ``signal.where(universe != 0, 0.0)`` and
``NaN != 0`` is True, so the excluded coin leaked into the book.

This fixture builds exactly that situation: a coin sits at mcap rank 1 on day 0
(selected → creates the column) and at mcap rank 31 on day 1 (just below
``top_by_mcap=30`` → must be excluded). Pre-fix, the DuckDB frame had a NaN at
(day 1, that coin) and did not equal the vectorized frame; post-fix it is 0.0.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantbox.plugins.strategies._universe import (
    DUCKDB_AVAILABLE,
    select_universe,
    select_universe_duckdb,
)

pytestmark = pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="duckdb not installed; duckdb universe path unavailable")

TOP_BY_MCAP = 30
TOP_BY_VOLUME = 10
N_COINS = 35  # > top_by_mcap so a rank-31 coin exists
BOUNDARY = "COIN30"  # the coin we move across the rank-30/31 boundary


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    cols = [f"COIN{i:02d}" for i in range(N_COINS)]

    # Base mcap: strictly descending by index, so COIN00..COIN29 occupy the
    # top-30 and COIN30..COIN34 sit at ranks 31..35.
    mcap = pd.DataFrame(
        {c: [float(1000 - i * 10)] * len(dates) for i, c in enumerate(cols)},
        index=dates,
    )
    # Day 0: lift the boundary coin to the very top (rank 1) -> it qualifies the
    # mcap cut, so the DuckDB SQL emits a row and pivot() creates its column.
    mcap.loc[dates[0], BOUNDARY] = 99_999.0
    # Day 1: leave it at its base mcap -> rank 31 -> excluded (the leak day).

    # Volume is huge for the boundary coin on BOTH days, so the *volume* cut
    # never excludes it; only the mcap cut does. This isolates the mcap-rank
    # boundary as the sole reason it is dropped on day 1.
    vol = pd.DataFrame(
        {c: [float(1000 - i)] * len(dates) for i, c in enumerate(cols)},
        index=dates,
    )
    vol[BOUNDARY] = [99_999.0] * len(dates)

    prices = pd.DataFrame(1.0, index=dates, columns=cols)
    return prices, vol, mcap


def test_duckdb_matches_vectorized_at_mcap_rank_boundary() -> None:
    prices, vol, mcap = _fixture()
    kwargs = dict(
        top_by_mcap=TOP_BY_MCAP,
        top_by_volume=TOP_BY_VOLUME,
        exclude_tickers=[],  # pin exclusion explicitly (DEFAULT_STABLECOINS varies by env)
    )

    vectorized = select_universe(prices, vol, mcap, **kwargs)
    duckdb_uni = select_universe_duckdb(prices, vol, mcap, **kwargs)

    # The DuckDB frame must carry NO NaN cells (the bug left per-cell NaN for
    # the boundary coin on the day it dropped below the mcap cut).
    assert not duckdb_uni.isna().any().any(), "duckdb universe must not contain NaN cells"

    # The rank-31 coin must be EXCLUDED (0.0) on day 1 in BOTH paths, and
    # selected (1.0) on day 0 when it sits at rank 1 — i.e. the column exists
    # precisely because it qualified on day 0, which is what created the leak.
    assert vectorized[BOUNDARY].tolist() == [1.0, 0.0]
    assert duckdb_uni[BOUNDARY].reindex(vectorized.index).tolist() == [1.0, 0.0]

    # Byte-identical to the vectorized path over the full frame.
    pd.testing.assert_frame_equal(
        duckdb_uni.reindex(index=vectorized.index, columns=vectorized.columns),
        vectorized,
        check_dtype=False,
    )
