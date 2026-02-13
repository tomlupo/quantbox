from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class FileArtifactStore:
    def __init__(self, root: str, run_id: str, *, _readonly: bool = False):
        self._run_id = run_id
        self.root = Path(root) / run_id
        self._readonly = _readonly
        if not _readonly:
            self.root.mkdir(parents=True, exist_ok=True)

    @property
    def run_id(self) -> str:
        return self._run_id

    def get_path(self, name: str) -> str:
        return str(self.root / name)

    # ── write ────────────────────────────────────────────────

    def put_parquet(self, name: str, df: pd.DataFrame, *, schema=None) -> str:
        path = self.root / f"{name}.parquet"
        if schema is not None:
            try:
                errors = schema.validate(df)
                if errors:
                    logger.warning("Schema validation warnings for %s: %s", name, errors)
            except Exception as exc:
                logger.warning("Schema validation failed for %s: %s", name, exc)
        df.to_parquet(path, index=False)
        return str(path)

    def put_json(self, name: str, obj: dict[str, Any]) -> str:
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
        return str(path)

    def append_event(self, line: str) -> str:
        path = self.root / "events.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return str(path)

    # ── read ─────────────────────────────────────────────────

    def read_parquet(self, name: str) -> pd.DataFrame:
        path = self.root / f"{name}.parquet"
        return pd.read_parquet(path)

    def read_json(self, name: str) -> dict[str, Any]:
        path = self.root / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def list_artifacts(self) -> list[str]:
        manifest_path = self.root / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if "artifacts" in manifest:
                return list(manifest["artifacts"].keys())
        # fallback: scan directory for parquet/json files
        names = []
        for p in sorted(self.root.iterdir()):
            if p.suffix in (".parquet", ".json") and p.stem != "run_manifest":
                names.append(p.stem)
        return names

    def get_manifest(self) -> dict[str, Any]:
        path = self.root / "run_manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))

    # ── class methods for cross-run queries ──────────────────

    @classmethod
    def list_runs(
        cls,
        root: str,
        *,
        pipeline: str | None = None,
        mode: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        root_path = Path(root)
        manifests = sorted(root_path.glob("*/run_manifest.json"), reverse=True)
        runs: list[dict[str, Any]] = []
        for mp in manifests:
            try:
                data = json.loads(mp.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if pipeline and data.get("pipeline_name") != pipeline:
                continue
            if mode and data.get("mode") != mode:
                continue
            if since and data.get("asof", "") < since:
                continue
            data["_run_dir"] = str(mp.parent)
            runs.append(data)
            if len(runs) >= limit:
                break
        # sort by asof descending
        runs.sort(key=lambda r: r.get("asof", ""), reverse=True)
        return runs

    @classmethod
    def open_run(cls, root: str, run_id: str) -> FileArtifactStore:
        store = cls(root, run_id, _readonly=True)
        if not store.root.exists():
            raise FileNotFoundError(f"Run directory not found: {store.root}")
        return store

    @classmethod
    def query_artifacts(
        cls,
        root: str,
        artifact_name: str,
        **filters,
    ) -> pd.DataFrame:
        runs = cls.list_runs(root, **filters)
        paths = []
        for run in runs:
            run_dir = Path(run["_run_dir"])
            parquet_path = run_dir / f"{artifact_name}.parquet"
            if parquet_path.exists():
                paths.append(str(parquet_path))
        if not paths:
            return pd.DataFrame()
        try:
            import duckdb

            query = f"SELECT * FROM read_parquet({paths!r})"
            return duckdb.sql(query).df()
        except ImportError:
            frames = [pd.read_parquet(p) for p in paths]
            return pd.concat(frames, ignore_index=True)
