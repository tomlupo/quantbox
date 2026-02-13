"""Persistent, queryable data warehouse for quantbox.

Provides a ``Warehouse`` that combines a layered Parquet lake
(bronze/silver/gold) with a DuckDB analytical engine.

Usage::

    from quantbox.warehouse import Warehouse

    with Warehouse("./warehouse") as wh:
        wh.ingest("prices", prices_df)
        wh.register_dataset("crypto_spot", "/path/to/datasets/crypto-spot-daily")
        result = wh.query("SELECT * FROM crypto_spot__prices LIMIT 10")
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa

from .backends import DuckDBEngine, ParquetLake
from .ingestion import register_dataset as _register_dataset
from .ingestion import restore_dataset_views

logger = logging.getLogger(__name__)

__all__ = [
    "Warehouse",
    "ParquetLake",
    "DuckDBEngine",
]


class Warehouse:
    """High-level warehouse API wrapping ParquetLake + DuckDBEngine.

    Args:
        root: Base directory for Parquet bronze/silver/gold layers.
        database: Path to DuckDB database file. Defaults to ``root/warehouse.duckdb``.
        compression: Parquet compression codec (default: zstd).
    """

    def __init__(
        self,
        root: str | Path,
        database: str | Path | None = None,
        *,
        compression: str = "zstd",
    ):
        self.lake = ParquetLake(root, compression=compression)
        db_path = database or (Path(root) / "warehouse.duckdb")
        self.db = DuckDBEngine(db_path)

        # Restore any previously registered dataset views
        restore_dataset_views(self)

    # ── ingest ────────────────────────────────────────────────

    def ingest(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        layer: str = "bronze",
        partition_cols: list[str] | None = None,
        mode: str = "append",
    ) -> None:
        """Ingest a pandas DataFrame into the warehouse.

        Args:
            name: Logical table name.
            df: Data to ingest.
            layer: Target layer (bronze/silver/gold).
            partition_cols: Hive partition columns.
            mode: "append" or "overwrite".
        """
        table = pa.Table.from_pandas(df, preserve_index=False)
        self.lake.write(
            name,
            table,
            layer=layer,
            partition_cols=partition_cols,
            mode=mode,
        )

    # ── query ─────────────────────────────────────────────────

    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return pandas DataFrame."""
        return self.db.query_df(sql)

    def query_arrow(self, sql: str) -> pa.Table:
        """Execute SQL query and return PyArrow Table."""
        return self.db.query(sql)

    # ── views & materialization ───────────────────────────────

    def create_view(self, name: str, parquet_path_or_sql: str) -> None:
        """Create a DuckDB view over a Parquet path or SQL query."""
        self.db.create_view(name, parquet_path_or_sql)

    def create_lake_view(self, name: str, table: str, layer: str = "bronze") -> None:
        """Create a DuckDB view over a Parquet lake table."""
        glob = self.lake.get_parquet_glob(table, layer)
        self.db.create_view(name, glob)

    def materialize(self, name: str, sql: str) -> int:
        """Materialize SQL query as a DuckDB table. Returns row count."""
        return self.db.materialize(name, sql)

    # ── datasets ──────────────────────────────────────────────

    def register_dataset(self, name: str, path: str | Path) -> list[str]:
        """Register an external dataset (e.g. from quantbox-datasets).

        Creates zero-copy DuckDB views over the dataset's Parquet files.
        No data is copied — DuckDB reads directly from the source files.

        Returns list of created view names.
        """
        return _register_dataset(self, name, path)

    # ── metadata ──────────────────────────────────────────────

    def list_tables(self) -> dict[str, list[str]]:
        """List all tables across layers, DuckDB tables, and views."""
        result: dict[str, list[str]] = {}
        for layer in ("bronze", "silver", "gold"):
            tables = self.lake.list_tables(layer)
            if tables:
                result[layer] = tables
        db_tables = self.db.list_tables()
        if db_tables:
            result["duckdb_tables"] = db_tables
        views = self.db.list_views()
        if views:
            result["views"] = views
        return result

    def describe(self, table: str) -> list[dict]:
        """Describe a DuckDB table or view's columns."""
        return self.db.describe(table)

    # ── lifecycle ─────────────────────────────────────────────

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Warehouse:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
