"""Storage backends for the warehouse module.

ParquetLake — bronze/silver/gold layered Parquet storage with Hive partitioning.
DuckDBEngine — SQL queries, views, materializations over the lake.

Adapted from quantlabnew/datalayer backends.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional, Union

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

LAYERS = ("bronze", "silver", "gold")


# ── Parquet Lake ──────────────────────────────────────────────


class ParquetLake:
    """Layered Parquet storage with Hive partitioning.

    Directory layout::

        root/
          bronze/   ← raw, append-only
          silver/   ← cleaned / curated
          gold/     ← aggregated / materialized
    """

    def __init__(self, root: Union[str, Path], *, compression: str = "zstd"):
        self.root = Path(root)
        self.compression = compression
        for layer in LAYERS:
            (self.root / layer).mkdir(parents=True, exist_ok=True)

    def _table_path(self, table: str, layer: str = "bronze") -> Path:
        return self.root / layer / table

    # ── write ──

    def write(
        self,
        table: str,
        data: pa.Table,
        *,
        layer: str = "bronze",
        partition_cols: Optional[list[str]] = None,
        mode: str = "append",
    ) -> None:
        """Write a PyArrow Table to Parquet.

        Args:
            table: Logical table name.
            data: Data to write.
            layer: bronze / silver / gold.
            partition_cols: Hive partition columns (e.g. ["date"]).
            mode: "append" or "overwrite".
        """
        path = self._table_path(table, layer)

        if mode == "overwrite" and path.exists():
            import shutil
            shutil.rmtree(path)

        path.mkdir(parents=True, exist_ok=True)

        if partition_cols:
            pq.write_to_dataset(
                data,
                root_path=str(path),
                partition_cols=partition_cols,
                compression=self.compression,
                existing_data_behavior="overwrite_or_ignore",
            )
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            pq.write_table(
                data,
                path / f"part-{ts}.parquet",
                compression=self.compression,
            )

    # ── read ──

    def read(
        self,
        table: str,
        *,
        layer: str = "bronze",
        columns: Optional[list[str]] = None,
        filters: Optional[list[tuple]] = None,
    ) -> pa.Table:
        """Read Parquet data with optional predicate pushdown.

        Args:
            table: Logical table name.
            layer: bronze / silver / gold.
            columns: Subset of columns to read.
            filters: List of (col, op, val) tuples.
                     Operators: ==, !=, >, >=, <, <=, in.
        """
        path = self._table_path(table, layer)
        if not path.exists():
            raise FileNotFoundError(f"Table not found: {table} in {layer}")

        dataset = ds.dataset(path, format="parquet", partitioning="hive")
        return dataset.to_table(
            columns=columns, filter=self._build_filter(filters)
        )

    @staticmethod
    def _build_filter(
        filters: Optional[list[tuple]],
    ) -> Optional[ds.Expression]:
        if not filters:
            return None
        exprs = []
        for col, op, val in filters:
            field = ds.field(col)
            ops = {
                "==": field.__eq__,
                "!=": field.__ne__,
                ">": field.__gt__,
                ">=": field.__ge__,
                "<": field.__lt__,
                "<=": field.__le__,
            }
            if op == "in":
                exprs.append(field.isin(val))
            elif op in ops:
                exprs.append(ops[op](val))
        result = exprs[0]
        for e in exprs[1:]:
            result = result & e
        return result

    # ── metadata ──

    def list_tables(self, layer: str = "bronze") -> list[str]:
        layer_path = self.root / layer
        if not layer_path.exists():
            return []
        return sorted(d.name for d in layer_path.iterdir() if d.is_dir())

    def table_exists(self, table: str, layer: str = "bronze") -> bool:
        path = self._table_path(table, layer)
        return path.exists() and any(path.rglob("*.parquet"))

    def get_schema(self, table: str, layer: str = "bronze") -> pa.Schema:
        path = self._table_path(table, layer)
        if not path.exists():
            raise FileNotFoundError(f"Table not found: {table}")
        return ds.dataset(path, format="parquet", partitioning="hive").schema

    def get_row_count(self, table: str, layer: str = "bronze") -> int:
        path = self._table_path(table, layer)
        if not path.exists():
            return 0
        return ds.dataset(path, format="parquet", partitioning="hive").count_rows()

    def get_parquet_glob(self, table: str, layer: str = "bronze") -> str:
        """Return glob pattern for DuckDB ``read_parquet()``."""
        return str(self._table_path(table, layer)) + "/**/*.parquet"


# ── DuckDB Engine ─────────────────────────────────────────────


class DuckDBEngine:
    """DuckDB analytical engine for querying the lake.

    Supports SQL queries, view creation, materialization, and Parquet export.
    """

    def __init__(self, database: Union[str, Path]):
        import duckdb

        self._db_path = Path(database)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    @property
    def conn(self):
        if self._conn is None:
            import duckdb
            self._conn = duckdb.connect(str(self._db_path))
            self._conn.execute("SET enable_progress_bar = false")
        return self._conn

    # ── query ──

    def execute(self, sql: str, params: Optional[dict] = None):
        if params:
            return self.conn.execute(sql, params)
        return self.conn.execute(sql)

    def query(self, sql: str, params: Optional[dict] = None) -> pa.Table:
        """Execute SQL and return PyArrow Table."""
        return self.execute(sql, params).arrow()

    def query_df(self, sql: str, params: Optional[dict] = None):
        """Execute SQL and return pandas DataFrame."""
        return self.execute(sql, params).df()

    # ── views ──

    def create_view(
        self,
        name: str,
        sql_or_parquet: str,
        *,
        replace: bool = True,
    ) -> None:
        """Create a view from SQL or a Parquet glob path.

        If ``sql_or_parquet`` ends with ``.parquet`` or contains ``*``,
        it is treated as a Parquet path; otherwise as a SQL query.
        """
        create = "CREATE OR REPLACE VIEW" if replace else "CREATE VIEW"
        if sql_or_parquet.endswith(".parquet") or "*" in sql_or_parquet:
            query = f"SELECT * FROM read_parquet('{sql_or_parquet}', hive_partitioning=true)"
        else:
            query = sql_or_parquet
        self.conn.execute(f"{create} {name} AS {query}")

    # ── materialization ──

    def materialize(self, table: str, sql: str, *, replace: bool = True) -> int:
        """Materialize a SQL query as a DuckDB table. Returns row count."""
        if replace:
            self.conn.execute(f"DROP TABLE IF EXISTS {table}")
        self.conn.execute(f"CREATE TABLE {table} AS {sql}")
        result = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return result[0] if result else 0

    def export_parquet(
        self,
        sql: str,
        path: str,
        *,
        partition_by: Optional[list[str]] = None,
    ) -> None:
        """Export SQL query results to Parquet file(s)."""
        if partition_by:
            cols = ", ".join(partition_by)
            self.conn.execute(
                f"COPY ({sql}) TO '{path}' "
                f"(FORMAT PARQUET, PARTITION_BY ({cols}))"
            )
        else:
            self.conn.execute(f"COPY ({sql}) TO '{path}' (FORMAT PARQUET)")

    # ── write ──

    def write(
        self,
        table: str,
        data: pa.Table,
        *,
        mode: str = "append",
    ) -> None:
        """Write PyArrow Table to a DuckDB table."""
        if mode == "overwrite":
            self.conn.execute(f"DROP TABLE IF EXISTS {table}")
            self.conn.execute(f"CREATE TABLE {table} AS SELECT * FROM data")
        elif self.table_exists(table):
            self.conn.execute(f"INSERT INTO {table} SELECT * FROM data")
        else:
            self.conn.execute(f"CREATE TABLE {table} AS SELECT * FROM data")

    # ── metadata ──

    def list_tables(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_type = 'BASE TABLE'"
        ).fetchall()
        return sorted(r[0] for r in rows)

    def list_views(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_type = 'VIEW'"
        ).fetchall()
        return sorted(r[0] for r in rows)

    def table_exists(self, table: str) -> bool:
        result = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            f"WHERE table_name = '{table}'"
        ).fetchone()
        return bool(result and result[0] > 0)

    def describe(self, table: str) -> list[dict]:
        rows = self.conn.execute(f"DESCRIBE {table}").fetchall()
        return [
            {"column_name": r[0], "column_type": r[1], "null": r[2]}
            for r in rows
        ]

    # ── lifecycle ──

    def vacuum(self) -> None:
        self.conn.execute("VACUUM")

    @contextmanager
    def transaction(self) -> Generator:
        self.conn.execute("BEGIN TRANSACTION")
        try:
            yield self.conn
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()
