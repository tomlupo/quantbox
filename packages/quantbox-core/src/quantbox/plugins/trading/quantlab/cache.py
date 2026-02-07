import os
import shutil
import pickle
import logging
from pathlib import Path
from typing import Union, List, Any, Optional

import pandas as pd
import duckdb
from .utils import get_logger


class FastParquetCache:
    """Fast Parquet-based cache with DuckDB query support."""

    def __init__(self, cache_dir: str, logger: logging.Logger = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger('fast_parquet_cache')

    def _get_parquet_dir(self, key: str) -> Path:
        return self.cache_dir / key

    def _get_pickle_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def set(self, key: str, data: Any,
            partition_cols: Optional[List[str]] = None,
            primary_keys: Optional[List[str]] = None) -> None:
        """
        Store data in cache. DataFrames as Parquet, others as pickle.
        If `primary_keys` is provided, only rows not already stored (by those columns)
        will be written (dedup on append) using DuckDB anti-join.
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()

            # Optional partition columns (keeps your current behavior)
            if partition_cols and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                for col in partition_cols:
                    if col == "year":
                        df["year"] = df["date"].dt.year
                    elif col == "month":
                        df["month"] = df["date"].dt.month

            parquet_path = self._get_parquet_dir(key)
            parquet_path.mkdir(parents=True, exist_ok=True)

            # === NEW: filter-out already-stored rows via DuckDB anti-join ===
            if False:
                if primary_keys:
                    
                    con = duckdb.connect()
                    try:
                        # Register incoming df
                        con.register("new_df", df)

                        # Build a readable parquet source (handles partitioned dirs too)
                        # We read all existing shards lazily via glob. If none exist, anti-join yields all rows.
                        src = str(parquet_path / "**/*.parquet").replace("\\", "/")  # windows-safe

                        pk = ", ".join(primary_keys)  # only for display; we'll build predicates
                        predicates = " AND ".join([f"e.{c} = n.{c}" for c in primary_keys])

                        query = f"""
                        SELECT n.*
                        FROM new_df n
                        WHERE NOT EXISTS (
                        SELECT 1
                        FROM read_parquet('{src}', hive_partitioning=1) e
                        WHERE {predicates}
                        )
                        """
                        df = con.execute(query).df()
                    finally:
                        con.close()

                # If nothing new to write, just return
                if df.empty:
                    return

            # Write data to Parquet file with your existing sharding style
            now_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S%f')
            if partition_cols:
                df.to_parquet(
                    parquet_path,
                    basename_template=f'{now_str}-{{i}}.parquet',
                    partition_cols=partition_cols,
                    engine="pyarrow"
                )
            else:
                fname = f'{now_str}.parquet'
                path = parquet_path.joinpath(fname)
                df.to_parquet(path, engine="pyarrow")

        else:
            with self._get_pickle_path(key).open("wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get(self, key: str, columns: List[str] = None, sql_where: str = "") -> Union[pd.DataFrame, Any, None]:
        """
        Retrieve data from cache with optional SQL filtering.
        """
        parquet_dir = self._get_parquet_dir(key)
        if parquet_dir.exists() and parquet_dir.is_dir():
            cols = ", ".join(columns) if columns else "*"
            query = f"SELECT {cols} FROM read_parquet('{parquet_dir}/**/*.parquet')"
            if sql_where:
                query += f" WHERE {sql_where}"
            try:
                return duckdb.query(query).to_df()
            except Exception as e:
                self.logger.warning(f"DuckDB query error for key '{key}': {e}")
                return None

        pickle_path = self._get_pickle_path(key)
        if pickle_path.exists():
            with pickle_path.open("rb") as f:
                return pickle.load(f)

        return None

    def get_latest(
        self,
        key: str,
        order_col: str = "fetch_timestamp",
        sql_where: str = "",
        return_type: str = "timestamp",
        ticker: str = None,
        limit: int = None
    ) -> Union[pd.Timestamp, pd.DataFrame, None]:
        """
        Retrieve the latest entry from a cached DataFrame.
        """
        parquet_dir = self._get_parquet_dir(key)
        if not parquet_dir.exists() or not parquet_dir.is_dir():
            return pd.DataFrame() if return_type == "record" else None

        data_path = parquet_dir/'**'/'*.parquet'

        try:
            if return_type == "date":
                if not ticker:
                    raise ValueError("ticker parameter is required when return_type='date'")
                query = f"SELECT MAX({order_col}) AS last_date FROM read_parquet('{data_path}') WHERE ticker = '{ticker}'"
                if sql_where:
                    query += f" AND {sql_where}"
                result = duckdb.query(query).to_df()
                return pd.to_datetime(result["last_date"][0]) if not result.empty else None

            elif return_type == "timestamp":
                query = f"SELECT MAX({order_col}) AS last_timestamp FROM read_parquet('{data_path}')"
                if sql_where:
                    query += f" WHERE {sql_where}"
                result = duckdb.query(query).to_df()
                return pd.to_datetime(result["last_timestamp"][0]) if not result.empty else None

            elif return_type == "record":
                subquery = f"SELECT MAX({order_col}) AS latest_ts FROM read_parquet('{data_path}')"
                if sql_where:
                    subquery += f" WHERE {sql_where}"
                latest = duckdb.query(subquery).to_df()
                if latest.empty or pd.isnull(latest["latest_ts"][0]):
                    return pd.DataFrame()

                latest_ts = latest["latest_ts"][0]
                query = f"SELECT * FROM read_parquet('{data_path}') WHERE {order_col} = '{latest_ts}'"
                if sql_where:
                    query += f" AND {sql_where}"
                if limit:
                    query += f" LIMIT {limit}"
                return duckdb.query(query).to_df()

            else:
                raise ValueError(f"Invalid return_type: {return_type}")

        except Exception as e:
            self.logger.warning(f"get_latest error [{key}/{return_type}]: {e}")
            return pd.DataFrame() if return_type == "record" else None

    def clear(self, key: str = None) -> None:
        """
        Clear cache. If key is None, clears all entries.
        """
        if key:
            for path in [self._get_parquet_dir(key), self._get_pickle_path(key)]:
                if path.exists():
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path)
        else:
            for path in self.cache_dir.iterdir():
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)

    def list_keys(self) -> List[str]:
        """
        List all cache keys.
        """
        keys = []
        for path in self.cache_dir.iterdir():
            if path.is_file() and path.suffix == ".pkl":
                keys.append(path.stem)
            elif path.is_dir():
                keys.append(path.name)
        return sorted(set(keys))

    def has(self, key: str) -> bool:
        """
        Check if a key exists in the cache.
        """
        return self._get_parquet_dir(key).exists() or self._get_pickle_path(key).exists()
