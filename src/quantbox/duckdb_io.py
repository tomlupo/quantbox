"""DuckDB query results as pandas frames, with tz-aware timestamps kept tz-aware.

duckdb 2.0 reads a nanosecond tz-aware parquet column (what pandas writes for
``datetime64[ns, tz]``) as the new type ``TIMESTAMPTZ_NS``. Its pandas export
(``.df()`` / ``.fetchdf()``) drops the zone of that type: the column comes back
as NAIVE ``datetime64[ns]`` holding the UTC wall clock. duckdb 1.5 read the same
column as ``TIMESTAMP WITH TIME ZONE`` and handed back ``datetime64[us, <session
tz>]``. A caller that reads the naive value as a wall-clock time is then off by
the zone's UTC offset: the asof cut of ``local_file_data`` labelled a Warsaw
daily bar one day early and dropped the asof-day bar of a New York file.

:func:`fetch_df` restores what 1.5 returned: every ``TIMESTAMPTZ_NS`` column
that came back naive is localized to UTC (the instant duckdb stored) and
converted to the session's timezone. A column that already carries a zone is
left alone, so the helper is a no-op on duckdb 1.5 and on a duckdb that fixes
its export. Every duckdb -> pandas fetch in quantbox goes through here.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["fetch_df"]

_TZ_NS = "TIMESTAMPTZ_NS"


def fetch_df(con: Any, query: str, params: dict | None = None) -> pd.DataFrame:
    """Run *query* on duckdb connection (or module) *con*; return the result as pandas.

    Same frame as ``con.sql(query, params=params).df()``, except that a
    ``TIMESTAMPTZ_NS`` column is tz-aware in the session's timezone, as duckdb 1.5
    returned it.
    """
    rel = con.sql(query, params=params) if params else con.sql(query)
    if rel is None:  # a statement with no result set (DDL, SET)
        return pd.DataFrame()
    df = rel.df()
    session_tz: str | None = None
    for pos, typ in enumerate(rel.types):
        if str(typ) != _TZ_NS:
            continue
        col = df.iloc[:, pos]
        if isinstance(col.dtype, pd.DatetimeTZDtype):
            continue
        if session_tz is None:
            # fetchall, not fetchone: duckdb 2.0 relations have no fetchone().
            session_tz = con.sql("SELECT current_setting('TimeZone')").fetchall()[0][0]
        df.isetitem(pos, col.dt.tz_localize("UTC").dt.tz_convert(session_tz))
    return df
