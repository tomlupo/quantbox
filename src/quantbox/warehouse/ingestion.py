"""Ingestion utilities for the warehouse.

- ``ingest_run``: bridge FileArtifactStore artifacts → warehouse bronze layer.
- ``register_dataset``: create zero-copy DuckDB views over quantbox-datasets Parquet files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa

if TYPE_CHECKING:
    from ..store import FileArtifactStore
    from . import Warehouse

logger = logging.getLogger(__name__)


def ingest_run(
    warehouse: Warehouse,
    store: FileArtifactStore,
    *,
    tables: list[str] | None = None,
) -> dict[str, int]:
    """Ingest run artifacts from a FileArtifactStore into the warehouse.

    Reads each Parquet artifact from the store, adds ``run_id`` and ``asof``
    metadata columns, and writes to the warehouse bronze layer.

    Args:
        warehouse: Target warehouse.
        store: Source artifact store (from a completed pipeline run).
        tables: Artifact names to ingest (default: all Parquet artifacts).

    Returns:
        Dict mapping table name → rows ingested.
    """
    # Determine which artifacts to ingest
    available = store.list_artifacts()
    if tables:
        to_ingest = [t for t in tables if t in available]
    else:
        to_ingest = available

    # Extract run metadata
    run_id = store.run_id
    try:
        manifest = store.get_manifest()
        asof = manifest.get("asof", "")
    except (FileNotFoundError, json.JSONDecodeError):
        asof = ""

    results: dict[str, int] = {}

    for name in to_ingest:
        path = Path(store.get_path(name))
        parquet_path = path if path.suffix == ".parquet" else path.with_suffix(".parquet")
        if not parquet_path.exists():
            continue

        try:
            df = pd.read_parquet(parquet_path)
            if df.empty:
                continue

            # Add metadata columns
            df["run_id"] = run_id
            if asof:
                df["asof"] = asof

            arrow_table = pa.Table.from_pandas(df, preserve_index=False)
            warehouse.lake.write(name, arrow_table, layer="bronze")
            results[name] = len(df)
            logger.info("Ingested %s: %d rows into bronze", name, len(df))
        except Exception:
            logger.exception("Failed to ingest artifact: %s", name)

    return results


def register_dataset(
    warehouse: Warehouse,
    name: str,
    path: str | Path,
) -> list[str]:
    """Create zero-copy DuckDB views over quantbox-datasets Parquet files.

    Scans ``path`` for ``.parquet`` files and creates a DuckDB view for each::

        datasets/crypto-spot-daily/prices.parquet  → view: crypto_spot_daily__prices
        datasets/etf-daily/volume.parquet          → view: etf_daily__volume

    No data is copied — DuckDB reads directly from the dataset Parquet files.

    The registration is also persisted as a JSON metadata file so views
    can be recreated when the warehouse is reopened.

    Args:
        warehouse: Target warehouse.
        name: Short name for the dataset (used as view prefix).
        path: Root directory of the dataset (contains .parquet files).

    Returns:
        List of created view names.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    safe_name = name.replace("-", "_").replace(".", "_")
    views_created: list[str] = []

    # Find all parquet files (non-recursive for top-level datasets)
    parquet_files = sorted(dataset_path.glob("*.parquet"))
    if not parquet_files:
        # Try recursive (Hive-partitioned datasets)
        parquet_files = sorted(dataset_path.glob("**/*.parquet"))

    if not parquet_files:
        logger.warning("No parquet files found in %s", path)
        return views_created

    # For top-level parquet files, create one view per file
    top_level = [f for f in parquet_files if f.parent == dataset_path]
    if top_level:
        for pf in top_level:
            stem = pf.stem.replace("-", "_").replace(".", "_")
            view_name = f"{safe_name}__{stem}"
            warehouse.db.create_view(view_name, str(pf))
            views_created.append(view_name)
            logger.info("Created view %s → %s", view_name, pf)
    else:
        # Single glob view over all partitioned files
        glob_pattern = str(dataset_path) + "/**/*.parquet"
        view_name = safe_name
        warehouse.db.create_view(view_name, glob_pattern)
        views_created.append(view_name)
        logger.info("Created view %s → %s", view_name, glob_pattern)

    # Persist registration metadata
    _save_dataset_registration(warehouse, name, str(dataset_path), views_created)

    return views_created


def restore_dataset_views(warehouse: Warehouse) -> list[str]:
    """Restore dataset views from persisted registrations.

    Called during ``Warehouse.__init__`` to recreate views
    from previous ``register_dataset`` calls.
    """
    meta_path = warehouse.lake.root / "_meta" / "datasets.json"
    if not meta_path.exists():
        return []

    try:
        registrations = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    all_views: list[str] = []
    for reg in registrations:
        dataset_path = Path(reg["path"])
        if not dataset_path.exists():
            logger.warning("Dataset path no longer exists: %s", reg["path"])
            continue
        try:
            views = register_dataset(warehouse, reg["name"], reg["path"])
            all_views.extend(views)
        except Exception:
            logger.exception("Failed to restore dataset: %s", reg["name"])

    return all_views


def _save_dataset_registration(
    warehouse: Warehouse,
    name: str,
    path: str,
    views: list[str],
) -> None:
    """Persist dataset registration to JSON metadata file."""
    meta_dir = warehouse.lake.root / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "datasets.json"

    registrations: list[dict[str, Any]] = []
    if meta_path.exists():
        try:
            registrations = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            registrations = []

    # Update or add
    existing = {r["name"]: i for i, r in enumerate(registrations)}
    entry = {"name": name, "path": path, "views": views}
    if name in existing:
        registrations[existing[name]] = entry
    else:
        registrations.append(entry)

    meta_path.write_text(
        json.dumps(registrations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
