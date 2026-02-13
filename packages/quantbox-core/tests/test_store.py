"""Tests for FileArtifactStore read/query capabilities."""

import pandas as pd
import pytest

from quantbox.store import FileArtifactStore


@pytest.fixture
def store(tmp_path):
    return FileArtifactStore(str(tmp_path), "run_001")


class TestReadWrite:
    def test_roundtrip_parquet(self, store):
        df = pd.DataFrame({"symbol": ["BTC", "ETH"], "weight": [0.6, 0.4]})
        store.put_parquet("weights", df)
        loaded = store.read_parquet("weights")
        pd.testing.assert_frame_equal(df, loaded)

    def test_roundtrip_json(self, store):
        obj = {"pipeline": "test", "asof": "2026-02-01", "metrics": {"pv": 100.0}}
        store.put_json("run_manifest", obj)
        loaded = store.read_json("run_manifest")
        assert loaded == obj

    def test_read_parquet_missing(self, store):
        with pytest.raises((FileNotFoundError, KeyError)):
            store.read_parquet("nonexistent")

    def test_read_json_missing(self, store):
        with pytest.raises((FileNotFoundError, KeyError)):
            store.read_json("nonexistent")


class TestListArtifacts:
    def test_list_from_manifest(self, store):
        store.put_parquet("weights", pd.DataFrame({"a": [1]}))
        store.put_parquet("orders", pd.DataFrame({"b": [2]}))
        manifest = {"artifacts": {"weights": "weights.parquet", "orders": "orders.parquet"}}
        store.put_json("run_manifest", manifest)
        names = store.list_artifacts()
        assert "weights" in names
        assert "orders" in names

    def test_list_fallback_dir_scan(self, store):
        store.put_parquet("weights", pd.DataFrame({"a": [1]}))
        store.put_json("config", {"x": 1})
        names = store.list_artifacts()
        assert "weights" in names
        assert "config" in names

    def test_get_manifest(self, store):
        manifest = {"pipeline_name": "test.v1", "asof": "2026-02-01"}
        store.put_json("run_manifest", manifest)
        loaded = store.get_manifest()
        assert loaded["pipeline_name"] == "test.v1"


class TestListRuns:
    def _make_run(self, root, run_id, pipeline="p.v1", mode="paper", asof="2026-01-01"):
        s = FileArtifactStore(str(root), run_id)
        manifest = {
            "run_id": run_id,
            "pipeline_name": pipeline,
            "mode": mode,
            "asof": asof,
        }
        s.put_json("run_manifest", manifest)
        return s

    def test_list_runs_basic(self, tmp_path):
        self._make_run(tmp_path, "run_a", asof="2026-01-01")
        self._make_run(tmp_path, "run_b", asof="2026-01-02")
        runs = FileArtifactStore.list_runs(str(tmp_path))
        assert len(runs) == 2
        # sorted desc by asof
        assert runs[0]["asof"] == "2026-01-02"

    def test_filter_pipeline(self, tmp_path):
        self._make_run(tmp_path, "run_a", pipeline="alpha")
        self._make_run(tmp_path, "run_b", pipeline="beta")
        runs = FileArtifactStore.list_runs(str(tmp_path), pipeline="alpha")
        assert len(runs) == 1
        assert runs[0]["pipeline_name"] == "alpha"

    def test_filter_mode(self, tmp_path):
        self._make_run(tmp_path, "run_a", mode="paper")
        self._make_run(tmp_path, "run_b", mode="live")
        runs = FileArtifactStore.list_runs(str(tmp_path), mode="live")
        assert len(runs) == 1

    def test_filter_since(self, tmp_path):
        self._make_run(tmp_path, "run_a", asof="2025-12-01")
        self._make_run(tmp_path, "run_b", asof="2026-01-15")
        runs = FileArtifactStore.list_runs(str(tmp_path), since="2026-01-01")
        assert len(runs) == 1
        assert runs[0]["asof"] == "2026-01-15"

    def test_limit(self, tmp_path):
        for i in range(5):
            self._make_run(tmp_path, f"run_{i}", asof=f"2026-01-0{i + 1}")
        runs = FileArtifactStore.list_runs(str(tmp_path), limit=2)
        assert len(runs) == 2


class TestOpenRun:
    def test_open_existing(self, tmp_path):
        original = FileArtifactStore(str(tmp_path), "run_x")
        original.put_json("run_manifest", {"asof": "2026-02-01"})

        opened = FileArtifactStore.open_run(str(tmp_path), "run_x")
        assert opened.run_id == "run_x"
        manifest = opened.read_json("run_manifest")
        assert manifest["asof"] == "2026-02-01"

    def test_open_nonexistent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FileArtifactStore.open_run(str(tmp_path), "no_such_run")

    def test_readonly_no_mkdir(self, tmp_path):
        s = FileArtifactStore(str(tmp_path), "run_ro", _readonly=True)
        # root dir should NOT have been created
        assert not s.root.exists()


class TestQueryArtifacts:
    def test_query_union(self, tmp_path):
        for i in range(3):
            s = FileArtifactStore(str(tmp_path), f"run_{i}")
            df = pd.DataFrame({"symbol": ["BTC"], "weight": [0.1 * (i + 1)]})
            s.put_parquet("weights", df)
            s.put_json(
                "run_manifest",
                {
                    "run_id": f"run_{i}",
                    "pipeline_name": "test",
                    "mode": "paper",
                    "asof": f"2026-01-0{i + 1}",
                },
            )

        result = FileArtifactStore.query_artifacts(str(tmp_path), "weights")
        assert len(result) == 3

    def test_query_missing_artifact(self, tmp_path):
        s = FileArtifactStore(str(tmp_path), "run_0")
        s.put_json(
            "run_manifest",
            {
                "run_id": "run_0",
                "pipeline_name": "test",
                "mode": "paper",
                "asof": "2026-01-01",
            },
        )
        result = FileArtifactStore.query_artifacts(str(tmp_path), "nonexistent")
        assert len(result) == 0


class TestSchemaIntegration:
    def test_put_parquet_with_schema(self, store):
        from quantbox.schemas import AGGREGATED_WEIGHTS_SCHEMA

        df = pd.DataFrame({"symbol": ["BTC", "ETH"], "weight": [0.6, 0.4]})
        path = store.put_parquet("weights", df, schema=AGGREGATED_WEIGHTS_SCHEMA)
        assert path.endswith(".parquet")

    def test_put_parquet_schema_warning(self, store, caplog):
        import logging

        from quantbox.schemas import AGGREGATED_WEIGHTS_SCHEMA

        # missing required 'weight' column
        df = pd.DataFrame({"symbol": ["BTC"], "bad_col": [1.0]})
        with caplog.at_level(logging.WARNING):
            store.put_parquet("weights", df, schema=AGGREGATED_WEIGHTS_SCHEMA)
        assert "Schema validation" in caplog.text
