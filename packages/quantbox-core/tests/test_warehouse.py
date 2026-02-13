"""Tests for quantbox.warehouse module."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from quantbox.warehouse import Warehouse
from quantbox.warehouse.backends import DuckDBEngine, ParquetLake
from quantbox.warehouse.ingestion import ingest_run
from quantbox.warehouse.schemas import (
    PortfolioSnapshotSchema,
    PriceBarSchema,
)

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def tmp_warehouse(tmp_path):
    with Warehouse(tmp_path / "wh") as wh:
        yield wh


@pytest.fixture
def tmp_lake(tmp_path):
    return ParquetLake(tmp_path / "lake")


@pytest.fixture
def tmp_db(tmp_path):
    engine = DuckDBEngine(tmp_path / "test.duckdb")
    yield engine
    engine.close()


@pytest.fixture
def sample_prices_df():
    return pd.DataFrame(
        {
            "symbol": ["BTC", "ETH", "BTC", "ETH"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]).date,
            "open": [95000.0, 3200.0, 95500.0, 3250.0],
            "high": [96000.0, 3300.0, 96500.0, 3350.0],
            "low": [94000.0, 3100.0, 94500.0, 3150.0],
            "close": [95500.0, 3250.0, 96000.0, 3300.0],
            "volume": [1000.0, 5000.0, 1100.0, 5100.0],
        }
    )


@pytest.fixture
def sample_arrow_table(sample_prices_df):
    return pa.Table.from_pandas(sample_prices_df, preserve_index=False)


# ── ParquetLake ───────────────────────────────────────────────


class TestParquetLake:
    def test_directories_created(self, tmp_lake):
        assert (tmp_lake.root / "bronze").is_dir()
        assert (tmp_lake.root / "silver").is_dir()
        assert (tmp_lake.root / "gold").is_dir()

    def test_write_and_read(self, tmp_lake, sample_arrow_table):
        tmp_lake.write("prices", sample_arrow_table)
        result = tmp_lake.read("prices")
        assert len(result) == 4
        assert "symbol" in result.column_names

    def test_write_overwrite(self, tmp_lake, sample_arrow_table):
        tmp_lake.write("prices", sample_arrow_table)
        tmp_lake.write("prices", sample_arrow_table, mode="overwrite")
        result = tmp_lake.read("prices")
        assert len(result) == 4  # overwritten, not doubled

    def test_write_append(self, tmp_lake, sample_arrow_table):
        tmp_lake.write("prices", sample_arrow_table)
        tmp_lake.write("prices", sample_arrow_table)
        result = tmp_lake.read("prices")
        assert len(result) == 8  # doubled

    def test_read_not_found(self, tmp_lake):
        with pytest.raises(FileNotFoundError):
            tmp_lake.read("nonexistent")

    def test_list_tables(self, tmp_lake, sample_arrow_table):
        assert tmp_lake.list_tables() == []
        tmp_lake.write("prices", sample_arrow_table)
        tmp_lake.write("volume", sample_arrow_table)
        assert sorted(tmp_lake.list_tables()) == ["prices", "volume"]

    def test_table_exists(self, tmp_lake, sample_arrow_table):
        assert not tmp_lake.table_exists("prices")
        tmp_lake.write("prices", sample_arrow_table)
        assert tmp_lake.table_exists("prices")

    def test_get_schema(self, tmp_lake, sample_arrow_table):
        tmp_lake.write("prices", sample_arrow_table)
        schema = tmp_lake.get_schema("prices")
        assert "symbol" in schema.names

    def test_get_row_count(self, tmp_lake, sample_arrow_table):
        tmp_lake.write("prices", sample_arrow_table)
        assert tmp_lake.get_row_count("prices") == 4

    def test_get_parquet_glob(self, tmp_lake):
        glob = tmp_lake.get_parquet_glob("prices")
        assert glob.endswith("/**/*.parquet")
        assert "bronze" in glob

    def test_layers(self, tmp_lake, sample_arrow_table):
        tmp_lake.write("prices", sample_arrow_table, layer="bronze")
        tmp_lake.write("prices", sample_arrow_table, layer="silver")
        assert tmp_lake.table_exists("prices", "bronze")
        assert tmp_lake.table_exists("prices", "silver")
        assert not tmp_lake.table_exists("prices", "gold")

    def test_read_with_filters(self, tmp_lake, sample_arrow_table):
        tmp_lake.write("prices", sample_arrow_table)
        result = tmp_lake.read("prices", filters=[("symbol", "==", "BTC")])
        assert len(result) == 2
        assert all(v.as_py() == "BTC" for v in result.column("symbol"))

    def test_partitioned_write(self, tmp_lake, sample_arrow_table):
        tmp_lake.write("prices", sample_arrow_table, partition_cols=["symbol"])
        assert tmp_lake.table_exists("prices")
        result = tmp_lake.read("prices")
        assert len(result) == 4


# ── DuckDBEngine ──────────────────────────────────────────────


class TestDuckDBEngine:
    def test_query(self, tmp_db):
        result = tmp_db.query_df("SELECT 1 AS x")
        assert result.iloc[0]["x"] == 1

    def test_write_and_read(self, tmp_db, sample_arrow_table):
        tmp_db.write("prices", sample_arrow_table)
        result = tmp_db.query_df("SELECT COUNT(*) AS n FROM prices")
        assert result.iloc[0]["n"] == 4

    def test_write_overwrite(self, tmp_db, sample_arrow_table):
        tmp_db.write("prices", sample_arrow_table)
        tmp_db.write("prices", sample_arrow_table, mode="overwrite")
        result = tmp_db.query_df("SELECT COUNT(*) AS n FROM prices")
        assert result.iloc[0]["n"] == 4

    def test_write_append(self, tmp_db, sample_arrow_table):
        tmp_db.write("prices", sample_arrow_table)
        tmp_db.write("prices", sample_arrow_table)
        result = tmp_db.query_df("SELECT COUNT(*) AS n FROM prices")
        assert result.iloc[0]["n"] == 8

    def test_create_view(self, tmp_db, tmp_path, sample_arrow_table):
        # Write parquet file
        pq_path = tmp_path / "test.parquet"
        pa.parquet.write_table(sample_arrow_table, pq_path)
        tmp_db.create_view("v_prices", str(pq_path))
        result = tmp_db.query_df("SELECT COUNT(*) AS n FROM v_prices")
        assert result.iloc[0]["n"] == 4

    def test_materialize(self, tmp_db, sample_arrow_table):
        tmp_db.write("raw_prices", sample_arrow_table)
        count = tmp_db.materialize(
            "btc_prices",
            "SELECT * FROM raw_prices WHERE symbol = 'BTC'",
        )
        assert count == 2

    def test_list_tables_and_views(self, tmp_db, sample_arrow_table):
        tmp_db.write("prices", sample_arrow_table)
        tmp_db.create_view("v_test", "SELECT 1 AS x")
        assert "prices" in tmp_db.list_tables()
        assert "v_test" in tmp_db.list_views()

    def test_table_exists(self, tmp_db, sample_arrow_table):
        assert not tmp_db.table_exists("prices")
        tmp_db.write("prices", sample_arrow_table)
        assert tmp_db.table_exists("prices")

    def test_describe(self, tmp_db, sample_arrow_table):
        tmp_db.write("prices", sample_arrow_table)
        desc = tmp_db.describe("prices")
        col_names = [d["column_name"] for d in desc]
        assert "symbol" in col_names
        assert "close" in col_names

    def test_transaction_commit(self, tmp_db, sample_arrow_table):
        tmp_db.write("prices", sample_arrow_table)
        with tmp_db.transaction():
            tmp_db.conn.execute("DELETE FROM prices WHERE symbol = 'BTC'")
        result = tmp_db.query_df("SELECT COUNT(*) AS n FROM prices")
        assert result.iloc[0]["n"] == 2

    def test_transaction_rollback(self, tmp_db, sample_arrow_table):
        tmp_db.write("prices", sample_arrow_table)
        with pytest.raises(ValueError), tmp_db.transaction():
            tmp_db.conn.execute("DELETE FROM prices WHERE symbol = 'BTC'")
            raise ValueError("force rollback")
        result = tmp_db.query_df("SELECT COUNT(*) AS n FROM prices")
        assert result.iloc[0]["n"] == 4

    def test_export_parquet(self, tmp_db, tmp_path, sample_arrow_table):
        tmp_db.write("prices", sample_arrow_table)
        out_path = tmp_path / "export.parquet"
        tmp_db.export_parquet("SELECT * FROM prices", str(out_path))
        assert out_path.exists()

    def test_close_and_reopen(self, tmp_path, sample_arrow_table):
        db_path = tmp_path / "reopen.duckdb"
        engine = DuckDBEngine(db_path)
        engine.write("prices", sample_arrow_table)
        engine.close()
        engine2 = DuckDBEngine(db_path)
        result = engine2.query_df("SELECT COUNT(*) AS n FROM prices")
        assert result.iloc[0]["n"] == 4
        engine2.close()


# ── Warehouse Schema ──────────────────────────────────────────


class TestWarehouseSchema:
    def test_fields(self):
        fields = PriceBarSchema.fields()
        names = [f.name for f in fields]
        assert "symbol" in names
        assert "close" in names
        assert "volume" in names

    def test_to_arrow_schema(self):
        schema = PriceBarSchema.to_arrow_schema()
        assert isinstance(schema, pa.Schema)
        assert "symbol" in schema.names

    def test_validate_valid(self, sample_arrow_table):
        errors = PriceBarSchema.validate(sample_arrow_table)
        assert errors == []

    def test_validate_missing_required(self):
        table = pa.table({"foo": [1, 2]})
        errors = PriceBarSchema.validate(table)
        assert any("symbol" in e for e in errors)
        assert any("close" in e for e in errors)

    def test_cast(self):
        # Int volume should cast to float
        table = pa.table(
            {
                "symbol": ["BTC"],
                "date": [pa.scalar(pd.Timestamp("2026-01-01").date(), type=pa.date32())],
                "open": [95000.0],
                "high": [96000.0],
                "low": [94000.0],
                "close": [95500.0],
                "volume": pa.array([1000], type=pa.int64()),
            }
        )
        result = PriceBarSchema.cast(table)
        assert result.column("volume").type == pa.float64()

    def test_empty_table(self):
        table = PriceBarSchema.empty_table()
        assert len(table) == 0
        assert "symbol" in table.column_names

    def test_portfolio_snapshot_schema(self):
        fields = PortfolioSnapshotSchema.field_names()
        assert "nav" in fields
        assert "gross_exposure" in fields


# ── Warehouse (end-to-end) ────────────────────────────────────


class TestWarehouse:
    def test_ingest_and_query(self, tmp_warehouse, sample_prices_df):
        tmp_warehouse.ingest("prices", sample_prices_df)
        glob = tmp_warehouse.lake.get_parquet_glob("prices")
        result = tmp_warehouse.query(f"SELECT COUNT(*) AS n FROM read_parquet('{glob}')")
        assert result.iloc[0]["n"] == 4

    def test_create_lake_view(self, tmp_warehouse, sample_prices_df):
        tmp_warehouse.ingest("prices", sample_prices_df)
        tmp_warehouse.create_lake_view("v_prices", "prices")
        result = tmp_warehouse.query("SELECT COUNT(*) AS n FROM v_prices")
        assert result.iloc[0]["n"] == 4

    def test_materialize(self, tmp_warehouse, sample_prices_df):
        tmp_warehouse.ingest("prices", sample_prices_df)
        tmp_warehouse.create_lake_view("v_prices", "prices")
        count = tmp_warehouse.materialize("btc_only", "SELECT * FROM v_prices WHERE symbol = 'BTC'")
        assert count == 2

    def test_list_tables(self, tmp_warehouse, sample_prices_df):
        tmp_warehouse.ingest("prices", sample_prices_df)
        tables = tmp_warehouse.list_tables()
        assert "prices" in tables.get("bronze", [])

    def test_register_dataset(self, tmp_warehouse, tmp_path, sample_prices_df):
        # Create a mock dataset directory with parquet files
        ds_dir = tmp_path / "datasets" / "crypto-spot-daily"
        ds_dir.mkdir(parents=True)
        sample_prices_df.to_parquet(ds_dir / "prices.parquet", index=False)
        sample_prices_df.to_parquet(ds_dir / "volume.parquet", index=False)

        views = tmp_warehouse.register_dataset("crypto_spot", ds_dir)
        assert len(views) == 2
        assert "crypto_spot__prices" in views
        assert "crypto_spot__volume" in views

        result = tmp_warehouse.query("SELECT COUNT(*) AS n FROM crypto_spot__prices")
        assert result.iloc[0]["n"] == 4

    def test_dataset_views_persist(self, tmp_path, sample_prices_df):
        # Create dataset
        ds_dir = tmp_path / "datasets" / "test"
        ds_dir.mkdir(parents=True)
        sample_prices_df.to_parquet(ds_dir / "data.parquet", index=False)

        wh_root = tmp_path / "wh"

        # Register and close
        with Warehouse(wh_root) as wh:
            wh.register_dataset("test_ds", ds_dir)
            result = wh.query("SELECT COUNT(*) AS n FROM test_ds__data")
            assert result.iloc[0]["n"] == 4

        # Reopen — views should be restored
        with Warehouse(wh_root) as wh:
            result = wh.query("SELECT COUNT(*) AS n FROM test_ds__data")
            assert result.iloc[0]["n"] == 4


# ── Ingestion ─────────────────────────────────────────────────


class TestIngestion:
    def test_ingest_run(self, tmp_warehouse, tmp_path, sample_prices_df):
        from quantbox.store import FileArtifactStore

        store = FileArtifactStore(str(tmp_path / "artifacts"), "test_run_001")
        store.put_parquet("targets", sample_prices_df)
        store.put_json(
            "run_manifest",
            {
                "run_id": "test_run_001",
                "asof": "2026-01-02",
                "artifacts": {"targets": str(tmp_path / "artifacts" / "test_run_001" / "targets.parquet")},
            },
        )

        results = ingest_run(tmp_warehouse, store)
        assert "targets" in results
        assert results["targets"] == 4

        # Verify data is in warehouse
        glob = tmp_warehouse.lake.get_parquet_glob("targets")
        df = tmp_warehouse.query(f"SELECT * FROM read_parquet('{glob}')")
        assert len(df) == 4
        assert "run_id" in df.columns
        assert df["run_id"].iloc[0] == "test_run_001"

    def test_ingest_run_selective(self, tmp_warehouse, tmp_path, sample_prices_df):
        from quantbox.store import FileArtifactStore

        _art_root = tmp_path / "artifacts" / "test_run_002"  # noqa: F841
        store = FileArtifactStore(str(tmp_path / "artifacts"), "test_run_002")
        targets_path = store.put_parquet("targets", sample_prices_df)
        fills_path = store.put_parquet("fills", sample_prices_df)
        store.put_json(
            "run_manifest",
            {
                "run_id": "test_run_002",
                "asof": "2026-01-02",
                "artifacts": {"targets": targets_path, "fills": fills_path},
            },
        )

        results = ingest_run(tmp_warehouse, store, tables=["targets"])
        assert "targets" in results
        assert "fills" not in results
