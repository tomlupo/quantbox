"""Parquet reads that survive interpreter teardown.

``pandas.read_parquet`` does not hand PyArrow a path. For a local file it opens
the file itself with :func:`pandas.io.common.get_handle` and passes PyArrow the
resulting *Python* file object, so PyArrow's reader threads call back into the
interpreter through that handle. On the production paper-trading box that race
aborts the process at teardown with::

    terminate called without an active exception

which is a native ``std::terminate`` (SIGABRT, exit 134) raised *after* the work
has completed and every artifact has been flushed — the process prints its
result, then dies with a non-zero status and no Python traceback.

Measured on prod (Python 3.12.3, pandas 2.3.3, pyarrow 24.0.0), reading the same
366×100 parquet file in a fresh process per trial:

- ``pd.read_parquet(path)``                    5 aborts / 380
- ``pd.read_parquet(path, use_threads=False)``  0 aborts / 200
- ``pq.read_table(path).to_pandas()``           0 aborts / 570

That ``use_threads=False`` row is the control that names the mechanism: remove
the reader threads and the abort goes with them.

Reading by path keeps the file handle entirely on the C++ side and removes the
callback, so the teardown race cannot happen. See TOM-435: eight of these
aborts over five weeks of the ``quantbox-paper`` cron, across seven unrelated
strategies, two of which turned a fully successful run into ``exit=1``.

Every *pandas-bound* parquet read in quantbox goes through :func:`read_parquet`
so the library cannot re-introduce the crash one call site at a time; a test
enforces that no module calls ``pd.read_parquet`` directly. Reads that quantbox
hands to DuckDB as SQL (``SELECT * FROM read_parquet(...)``) are a different
engine, never touch a Python file handle, and are out of scope here. It is a
drop-in for ``pd.read_parquet`` and falls back to it whenever the fast path does
not apply (non-local input, file-like object, unsupported keyword, or no
pyarrow).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except ImportError:  # pragma: no cover - pyarrow is a hard dep in practice
    pq = None  # type: ignore[assignment]
    PYARROW_AVAILABLE = False

__all__ = ["read_parquet"]


def _local_file(path: Any) -> str | None:
    """Return ``path`` as a plain filesystem path, or ``None`` if it isn't one.

    Only a ``str``/``os.PathLike`` naming an existing regular file qualifies.
    URLs, fsspec targets, directories (partitioned datasets) and open file
    objects are all delegated to pandas untouched.
    """
    if not isinstance(path, (str, os.PathLike)):
        return None
    text = os.fspath(path)
    if "://" in text:
        return None
    try:
        if not Path(text).is_file():
            return None
    except OSError:
        return None
    return text


def read_parquet(path: Any, columns: list[str] | None = None, **kwargs: Any) -> pd.DataFrame:
    """Read a parquet file into a DataFrame without the teardown-abort race.

    Drop-in replacement for :func:`pandas.read_parquet`. ``columns`` is
    supported on the fast path; any other keyword (``filters``,
    ``dtype_backend``, ``storage_options``, ...) falls back to pandas, which
    remains correct — only the rare native abort comes back with it.
    """
    local = _local_file(path)
    if local is None or not PYARROW_AVAILABLE or kwargs:
        return pd.read_parquet(path, columns=columns, **kwargs)
    # ``use_pandas_metadata`` is what makes a projected read faithful: pandas'
    # own reader adds the stored index columns back to ``columns``, and
    # without it a ``columns=[...]`` read silently returns a RangeIndex.
    return pq.read_table(local, columns=columns, use_pandas_metadata=True).to_pandas()
