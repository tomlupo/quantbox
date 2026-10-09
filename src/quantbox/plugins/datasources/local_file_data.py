"""Local file data plugin — load Parquet/CSV via DuckDB.

Replaces the old ``DuckDBParquetData`` stub with a real implementation that:
- Reads ``.parquet`` and ``.csv`` files via DuckDB SQL
- Auto-detects wide vs long format and pivots to wide
- Returns wide-format ``Dict[str, pd.DataFrame]`` from ``load_market_data()``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from quantbox._deprecation import moved
from quantbox.contracts import PluginMeta
from quantbox.dataset import _resolve_pinned, load_pinned_dataset
from quantbox.parquet_io import read_parquet

logger = logging.getLogger(__name__)

# ``_load_pinned_dataset`` is public now as quantbox.dataset.load_pinned_dataset (TOM-1449).
__getattr__ = moved(
    "quantbox.plugins.datasources.local_file_data",
    "quantbox.dataset",
    card="TOM-1449",
    names={"_load_pinned_dataset": "load_pinned_dataset"},
)

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    duckdb = None  # type: ignore[assignment]
    DUCKDB_AVAILABLE = False
    logger.warning("duckdb not installed — LocalFileDataPlugin will use pandas fallback")


def _universe_with_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """One row per symbol: ``symbol`` plus every column constant per symbol (``asset_class``, ``category``, ...).

    That metadata is what ``group_limits.by`` reads (quantbox.engine.groups). A column
    that varies within a symbol (a date, a market cap) is not metadata and is dropped,
    as every column but ``symbol`` was before.
    """
    df = df.dropna(subset=["symbol"])
    others = [c for c in df.columns if c != "symbol"]
    if others:
        per_symbol = df.groupby("symbol", sort=False)[others].nunique(dropna=False)
        others = [c for c in others if (per_symbol[c] <= 1).all()]
    return df[["symbol", *others]].drop_duplicates(subset=["symbol"]).reset_index(drop=True)


def _read_file(path: str, asof: str | None = None, symbols: list[str] | None = None) -> pd.DataFrame:
    """Read a Parquet or CSV file, optionally filtering by date and symbols.

    Auto-detects wide vs long format:
    - Long: has ``symbol`` + ``date`` columns → pivot to wide (date index, symbol columns)
    - Wide: date-like index/column + ticker columns → return as-is with date index
    """
    p = Path(path)
    if not p.exists():
        logger.warning("File not found: %s", path)
        return pd.DataFrame()

    ext = p.suffix.lower()

    if DUCKDB_AVAILABLE:
        df = _read_via_duckdb(path, ext, asof, symbols)
    else:
        df = _read_via_pandas(path, ext, asof, symbols)

    if df.empty:
        return df

    # Auto-detect long format and pivot to wide
    if "symbol" in df.columns and "date" in df.columns:
        df = _pivot_long_to_wide(df)

    # Ensure date index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date").sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()

    # Normalize tz-aware index to UTC midnight (DuckDB may return local tz)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").normalize()

    return df


def _read_via_duckdb(path: str, ext: str, asof: str | None, symbols: list[str] | None) -> pd.DataFrame:
    """Read file using DuckDB SQL for efficient filtering."""
    con = duckdb.connect()
    try:
        if ext == ".parquet":
            read_fn = f"read_parquet('{path}')"
        elif ext == ".csv":
            read_fn = f"read_csv_auto('{path}')"
        else:
            logger.warning("Unsupported file extension: %s, trying pandas", ext)
            return _read_via_pandas(path, ext, asof, symbols)

        # Build WHERE clause
        conditions: list[str] = []
        if asof:
            conditions.append(f"date <= '{asof}'")
        if symbols:
            sym_list = ", ".join(f"'{s}'" for s in symbols)
            conditions.append(f"symbol IN ({sym_list})")

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""

        # Check if columns exist before filtering
        # First, read schema to see if symbol/date columns exist
        schema_df = con.execute(f"SELECT * FROM {read_fn} LIMIT 0").fetchdf()
        cols = set(schema_df.columns)

        # Only apply symbol filter if column exists
        actual_conditions: list[str] = []
        if asof and "date" in cols:
            actual_conditions.append(f"date <= '{asof}'")
        if symbols and "symbol" in cols:
            sym_list = ", ".join(f"'{s}'" for s in symbols)
            actual_conditions.append(f"symbol IN ({sym_list})")

        where = f" WHERE {' AND '.join(actual_conditions)}" if actual_conditions else ""
        sql = f"SELECT * FROM {read_fn}{where}"
        return con.execute(sql).fetchdf()
    except Exception as exc:
        logger.warning("DuckDB read failed for %s: %s, falling back to pandas", path, exc)
        return _read_via_pandas(path, ext, asof, symbols)
    finally:
        con.close()


def _read_via_pandas(path: str, ext: str, asof: str | None, symbols: list[str] | None) -> pd.DataFrame:
    """Fallback reader using pandas."""
    if ext == ".parquet":
        df = read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if asof and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= asof]

    if symbols and "symbol" in df.columns:
        df = df[df["symbol"].isin(symbols)]

    return df


def _clip_frame(df: pd.DataFrame, asof: str | None) -> pd.DataFrame:
    """Give a dataset frame the same UTC date index and ``<= asof`` cut as ``_read_file``."""
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    df = df.copy()
    df.index = (df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")).normalize()
    if asof:
        df = df[df.index <= pd.Timestamp(asof, tz="UTC")]
    return df


def _pivot_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format (date, symbol, value) to wide (date index, symbol columns).

    Detects the value column automatically: first numeric column that isn't date/symbol.
    """
    # Identify value column(s) — everything except date and symbol
    value_cols = [c for c in df.columns if c not in ("date", "symbol")]
    if not value_cols:
        return df

    # Use first value column for the pivot
    value_col = value_cols[0]
    try:
        wide = df.pivot_table(index="date", columns="symbol", values=value_col)
        wide.columns.name = None
        return wide.reset_index()
    except Exception:
        return df


@dataclass
class LocalFileDataPlugin:
    """Data plugin that loads Parquet/CSV files via DuckDB.

    Supports wide and long input formats. Returns wide-format DataFrames
    from ``load_market_data()``.

    Config example::

        plugins:
          data:
            name: local_file_data
            params_init:
              prices_path: ./data/prices.parquet
              volume_path: ./data/volume.parquet

    Or, for a quantbox-datasets dataset, by name — rooted at ``$QUANTBOX_DATASETS_ROOT``
    and served at the build pinned in the ``datasets.lock`` nearest the config
    (``quantbox-datasets pin <name>``); ``quantbox dataset resolve <name> -c <config> --json`` shows
    what will be read. The ``*_path`` params are then ignored::

            params_init:
              dataset: etf-daily
    """

    meta = PluginMeta(
        name="local_file_data",
        kind="data",
        version="0.2.0",
        core_compat=">=0.1.0",
        description="Load market data from local Parquet/CSV files via DuckDB",
        inputs=(),
        outputs=("universe", "prices", "volume", "market_cap", "funding_rates", "fx"),
        params_schema={
            "type": "object",
            "properties": {
                "prices_path": {"description": "Parquet/CSV of close prices (wide or long)."},
                "volume_path": {"description": "Parquet/CSV of volume."},
                "market_cap_path": {"description": "Parquet/CSV of market cap."},
                "universe_path": {"description": "Parquet/CSV listing the universe symbols."},
                "funding_rates_path": {"description": "Parquet/CSV of funding rates."},
                "fx_path": {"description": "Parquet/CSV of FX rates."},
                "dataset": {
                    "description": "quantbox-datasets name served at the build pinned in datasets.lock; *_path are then ignored."
                },
                "dataset_lock": {
                    "description": "datasets.lock that pins `dataset`; the runner defaults it to the lock nearest the config."
                },
                "mode": {
                    "description": "Accepted for compatibility and unused: the run mode reaches the plugin per request."
                },
            },
        },
    )

    prices_path: str | None = None
    volume_path: str | None = None
    market_cap_path: str | None = None
    universe_path: str | None = None
    funding_rates_path: str | None = None
    fx_path: str | None = None
    dataset: str | None = None
    # The datasets.lock pinning ``dataset``; the runner sets it to the lock nearest the
    # config, so a run resolves the same build from any working directory.
    dataset_lock: str | None = None
    mode: str | None = None
    _dataset: Any = field(default=None, init=False, repr=False)
    #: The file each key was ACTUALLY read from by the last ``load_market_data``
    #: call — load-time ``params`` override the constructor paths, so these,
    #: not the fields above, are what the run manifest hashes (TOM-1348).
    loaded_paths: dict[str, str | None] = field(default_factory=dict, init=False, repr=False)
    # What ``quantbox dataset resolve -c <config> --json`` reports; the runner records it in the manifest.
    dataset_resolution: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def _pinned(self) -> Any:
        if self._dataset is None:
            self._dataset, self.dataset_resolution = load_pinned_dataset(
                self.dataset, self.dataset_lock, self.dataset_resolution
            )
        return self._dataset

    def planned_paths(self, params: dict[str, Any]) -> dict[str, str | None]:
        """The file ``prices`` and ``funding_rates`` are read from for load *params* — nothing read.

        :meth:`load_market_data` records exactly this as ``loaded_paths``, and
        ``quantbox config explain`` asks for it before a run. A by-name dataset is
        RESOLVED here (refused when its bytes are not the pin), never loaded: the
        paths are the build actually served (possibly restored from git history),
        and the ``*_path`` params are ignored.
        """
        if self.dataset:
            if self.dataset_resolution is None:
                self.dataset_resolution = _resolve_pinned(self.dataset, self.dataset_lock)
            served = self.dataset_resolution.get("path")
            return {
                "prices": str(Path(served) / "prices.parquet") if served else None,
                "funding_rates": self.dataset_resolution.get("funding_rates"),
            }
        ppath = params.get("prices_path") or params.get("path") or self.prices_path
        fpath = params.get("funding_rates_path") or self.funding_rates_path
        return {"prices": str(ppath) if ppath else None, "funding_rates": str(fpath) if fpath else None}

    def planned_market(self) -> str | None:
        """A by-name dataset's ``market`` (its manifest, as resolved — nothing loaded); None for inline paths.

        The funding guard reads it (:func:`quantbox.funding_guard.planned_market`): a perp
        market is refused on an engine that does not charge funding (TOM-1619).
        """
        if not self.dataset:
            return None
        if self.dataset_resolution is None:
            self.dataset_resolution = _resolve_pinned(self.dataset, self.dataset_lock)
        market = self.dataset_resolution.get("market")
        return str(market) if market else None

    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        """Load trading universe from file or params.

        Priority:
        1. ``params["symbols"]`` — explicit list
        2. ``universe_path`` / ``params["path"]`` — load from file
        3. Extract unique columns from prices file
        """
        symbols = params.get("symbols")
        if symbols:
            return pd.DataFrame({"symbol": symbols})

        if self.dataset:
            df = self._pinned().universe
            if "symbol" in df.columns:
                return _universe_with_metadata(df)
            return pd.DataFrame({"symbol": list(self._pinned().prices.columns)})

        path = params.get("path") or self.universe_path
        if path and Path(path).exists():
            df = _read_file(path)
            if "symbol" in df.columns:
                return _universe_with_metadata(df)
            # Wide format: column names are symbols
            return pd.DataFrame({"symbol": list(df.columns)})

        # Fallback: extract symbols from prices file
        ppath = params.get("prices_path") or self.prices_path
        if ppath and Path(ppath).exists():
            df = _read_file(ppath)
            if isinstance(df.index, pd.DatetimeIndex):
                return pd.DataFrame({"symbol": list(df.columns)})
            if "symbol" in df.columns:
                return pd.DataFrame({"symbol": df["symbol"].unique().tolist()})

        return pd.DataFrame(columns=["symbol"])

    def load_market_data(
        self,
        universe: pd.DataFrame,
        asof: str,
        params: dict[str, Any],
    ) -> dict[str, pd.DataFrame]:
        """Load market data as wide-format DataFrames.

        Returns dict with keys: ``prices``, ``volume``, ``market_cap``.
        Only ``prices`` is required; others default to empty DataFrames.
        """
        symbols = universe["symbol"].tolist() if not universe.empty and "symbol" in universe.columns else None

        def _select_cols(df: pd.DataFrame) -> pd.DataFrame:
            """For wide-format DataFrames, restrict to the requested symbol columns."""
            if symbols and isinstance(df.index, pd.DatetimeIndex) and "symbol" not in df.columns:
                available = [s for s in symbols if s in df.columns]
                return df[available]
            return df

        if self.dataset:
            ds = self._pinned()
            self.loaded_paths = self.planned_paths(params)
            return {
                key: _select_cols(_clip_frame(getattr(ds, key), asof))
                for key in ("prices", "volume", "market_cap", "funding_rates")
            }

        result: dict[str, pd.DataFrame] = {}

        # Prices
        self.loaded_paths = self.planned_paths(params)
        ppath, fpath = self.loaded_paths["prices"], self.loaded_paths["funding_rates"]
        if ppath:
            result["prices"] = _select_cols(_read_file(ppath, asof=asof, symbols=symbols))
        else:
            result["prices"] = pd.DataFrame()

        # Volume
        vpath = params.get("volume_path") or self.volume_path
        if vpath:
            result["volume"] = _select_cols(_read_file(vpath, asof=asof, symbols=symbols))
        else:
            result["volume"] = pd.DataFrame()

        # Market cap
        mpath = params.get("market_cap_path") or self.market_cap_path
        if mpath:
            result["market_cap"] = _select_cols(_read_file(mpath, asof=asof, symbols=symbols))
        else:
            result["market_cap"] = pd.DataFrame()

        # Funding rates
        if fpath:
            result["funding_rates"] = _select_cols(_read_file(fpath, asof=asof, symbols=symbols))
        else:
            result["funding_rates"] = pd.DataFrame()

        return result

    def load_fx(
        self,
        asof: str,
        params: dict[str, Any],
    ) -> pd.DataFrame | None:
        """Load FX rates from file."""
        path = params.get("fx_path") or self.fx_path
        if path is None:
            return None

        p = Path(path)
        if not p.exists():
            return None

        ext = p.suffix.lower()
        if DUCKDB_AVAILABLE:
            con = duckdb.connect()
            try:
                read_fn = f"read_parquet('{path}')" if ext == ".parquet" else f"read_csv_auto('{path}')"
                return con.execute(f"SELECT * FROM {read_fn} WHERE date <= '{asof}'").fetchdf()
            except Exception:
                pass
            finally:
                con.close()

        # Pandas fallback
        if ext == ".parquet":
            df = read_parquet(path)
        elif ext == ".csv":
            df = pd.read_csv(path)
        else:
            return None

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] <= asof]
        return df
