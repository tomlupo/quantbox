"""Date-partitioned PyArrow dataset cache for strategy weights.

Stores strategy weights as a PyArrow partitioned parquet dataset,
enabling incremental computation: each new date is written as its
own parquet file, no merge needed. On load, PyArrow reads all
partitions into a single Arrow table.

Storage layout::

    {cache_dir}/{strategy_name}/{config_hash}/
        date=2026-04-09/part-0.parquet
        date=2026-04-10/part-0.parquet
        ...
        meta.json   — config snapshot for traceability

Schema: ``date (timestamp[ns])``, ``ticker (string)``, ``weight_key
(string)``, ``weight (float64)``.

Append-only writes mean parallel writers for different dates are
safe. Same-date overwrites are handled by replacing the partition
directory before writing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


_SCHEMA = pa.schema(
    [
        ("date", pa.date32()),
        ("ticker", pa.string()),
        ("weight_key", pa.string()),
        ("weight", pa.float64()),
    ]
)


def _deterministic_hash(obj: dict | list | str) -> str:
    """SHA-256 of JSON-serialized config, truncated to 12 hex chars."""
    raw = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


class StrategyCache:
    """Incremental PyArrow dataset cache for strategy weights.

    Parameters
    ----------
    cache_dir : str or Path
        Root directory for all cached strategy results.
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def config_hash(self, strategy_cfg: dict, indicator_cfg: dict | None = None) -> str:
        """Deterministic hash of strategy + indicator configuration."""
        combined: dict = {"strategy": strategy_cfg}
        if indicator_cfg is not None:
            combined["indicators"] = indicator_cfg
        return _deterministic_hash(combined)

    def _cache_path(self, strategy_name: str, cfg_hash: str) -> Path:
        return self.cache_dir / strategy_name / cfg_hash

    def save_weights(
        self,
        strategy_name: str,
        cfg_hash: str,
        weights: dict[str, pd.DataFrame],
        meta: dict | None = None,
    ) -> None:
        """Save weights as a date-partitioned parquet dataset.

        Parameters
        ----------
        weights : dict
            ``{weight_key: DataFrame}`` where each DataFrame is
            date-indexed with ticker columns (the CAA/IT output
            format). All dates present in the input are saved as
            partitions; existing same-date partitions are replaced.
        meta : dict, optional
            Config snapshot saved alongside the dataset.
        """
        rows = []
        for key, df in weights.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            stacked = df.stack().rename("weight").reset_index()
            cols = list(stacked.columns)
            stacked = stacked.rename(columns={cols[0]: "date", cols[1]: "ticker"})
            stacked["weight_key"] = str(key)
            rows.append(stacked[["date", "ticker", "weight_key", "weight"]])

        if not rows:
            return

        new_df = pd.concat(rows, ignore_index=True)
        new_df["date"] = pd.to_datetime(new_df["date"]).dt.date
        new_df["ticker"] = new_df["ticker"].astype(str)
        new_df["weight_key"] = new_df["weight_key"].astype(str)
        new_df["weight"] = new_df["weight"].astype(float)

        cache_path = self._cache_path(strategy_name, cfg_hash)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Remove any existing partitions for the dates we're about to write,
        # so overwrites replace rather than duplicate
        for dt in pd.Series(new_df["date"]).unique():
            part_dir = cache_path / f"date={pd.Timestamp(dt).strftime('%Y-%m-%d')}"
            if part_dir.exists():
                shutil.rmtree(part_dir)

        table = pa.Table.from_pandas(new_df, schema=_SCHEMA, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(cache_path),
            partition_cols=["date"],
            existing_data_behavior="overwrite_or_ignore",
        )

        if meta is not None:
            (cache_path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    def load_weights(
        self,
        strategy_name: str,
        cfg_hash: str,
    ) -> dict[str, pd.DataFrame] | None:
        """Load cached weights as ``{weight_key: DataFrame(date x ticker)}``.

        Returns ``None`` if no cache exists.
        """
        cache_path = self._cache_path(strategy_name, cfg_hash)
        files = sorted(cache_path.glob("date=*/*.parquet"))
        if not files:
            return None

        try:
            dataset = pads.dataset(
                [str(f) for f in files],
                format="parquet",
                partitioning="hive",
                partition_base_dir=str(cache_path),
            )
            df = dataset.to_table().to_pandas()
        except Exception as e:
            logger.warning("Corrupt cache at %s (%s), ignoring", cache_path, e)
            return None

        if df.empty:
            return {}

        df["date"] = pd.to_datetime(df["date"])
        result: dict[str, pd.DataFrame] = {}
        for key, group in df.groupby("weight_key"):
            pivot = group.pivot(index="date", columns="ticker", values="weight")
            pivot.index.name = "date"
            pivot.columns.name = "ticker"
            result[str(key)] = pivot.sort_index()
        return result

    def cached_dates(
        self,
        strategy_name: str,
        cfg_hash: str,
    ) -> set[pd.Timestamp]:
        """Return the set of dates already cached (read from partition dirs)."""
        cache_path = self._cache_path(strategy_name, cfg_hash)
        if not cache_path.exists():
            return set()
        dates: set[pd.Timestamp] = set()
        for child in cache_path.iterdir():
            if child.is_dir() and child.name.startswith("date="):
                try:
                    dates.add(pd.Timestamp(child.name.split("=", 1)[1]))
                except (ValueError, TypeError):
                    continue
        return dates

    def clear(self, strategy_name: str, cfg_hash: str | None = None) -> None:
        """Remove cached data for a strategy (optionally specific config)."""
        if cfg_hash:
            path = self._cache_path(strategy_name, cfg_hash)
        else:
            path = self.cache_dir / strategy_name
        if path.exists():
            shutil.rmtree(path)
            logger.info("Cleared cache: %s", path)
