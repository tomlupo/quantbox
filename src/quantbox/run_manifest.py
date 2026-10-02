"""The run manifest, ``quantbox/run@1``.

``run_manifest.json`` is the machine-readable record of a run, and what
``quantbox run --json`` prints. Its contract is the JSON Schema shipped at
``artifact_schemas/run_manifest.schema.json`` — that file owns the field list
and the versioning rule; this module only reads it and fills the run@1 fields
the runner owes it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from functools import lru_cache
from importlib.resources import files as _res_files
from pathlib import Path
from typing import Any

SCHEMA_ID = "quantbox/run@1"

#: The canonical files a run@1 manifest points at (``RunResult.artifacts`` keys).
CANONICAL_FILES: tuple[str, ...] = ("returns", "traded_weights", "metrics")

#: engine name -> the distribution whose version IS the engine's version (rsims lives in quantbox).
_ENGINES = {"vectorbt": "vectorbt", "rsims": "quantbox"}


@lru_cache(maxsize=1)
def load_run_schema() -> dict[str, Any]:
    path = _res_files("quantbox").joinpath("artifact_schemas").joinpath("run_manifest.schema.json")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_run_manifest(manifest: dict[str, Any]) -> list[str]:
    """Every way ``manifest`` fails run@1, as messages; ``[]`` means it validates."""
    import jsonschema

    validator = jsonschema.Draft202012Validator(load_run_schema())
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in validator.iter_errors(manifest)
    ]


def run_kind(manifest: dict[str, Any]) -> str | None:
    """``research`` or ``backtest`` — what a run's result IS (docs/adr/0006); None when it simulated nothing.

    ``run.kind`` when the manifest carries it (minor 2). An older manifest that
    recorded ``execution.same_bar: true`` (v0.8.0 let ``lag_bars: 0`` run with a
    warning) is research too: it filled at the close it decided on.
    """
    kind = (manifest.get("run") or {}).get("kind")
    if kind:
        return str(kind)
    execution = manifest.get("execution")
    if not isinstance(execution, dict):
        return None
    return "research" if execution.get("same_bar") else "backtest"


def research_note(manifest: dict[str, Any]) -> str | None:
    """The one sentence every reader prints next to a research run's numbers; None for anything else."""
    if run_kind(manifest) != "research":
        return None
    reason = (manifest.get("execution") or {}).get("same_bar_reason") or "no reason recorded (written before ADR-0006)"
    return (
        "RESEARCH run, not a backtest: same-bar fills (lag_bars 0) under the explicit execution.same_bar "
        f"override — {reason}"
    )


def run_files(manifest: dict[str, Any]) -> dict[str, str | None]:
    """The canonical files a manifest lists, relative to its run directory."""
    return dict(manifest.get("files") or {})


def _sha256_file(path: str | Path | None) -> str | None:
    if not path:
        return None
    try:
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def engine_block(notes: dict[str, Any]) -> dict[str, Any] | None:
    name = notes.get("engine")
    if not name:
        return None
    import importlib.metadata as md

    try:
        version = md.version(_ENGINES.get(str(name), str(name)))
    except md.PackageNotFoundError:
        version = None
    return {"name": str(name), "version": version}


def _effective_path(data: Any, key: str, attr: str) -> str | None:
    """The file ``data`` actually read ``key`` from, else its constructor path.

    Load-time params (``plugins.pipeline.params.prices``) override a data
    plugin's constructor paths, so a plugin that records ``loaded_paths`` is
    believed over its fields; hashing the field would attribute the run to a
    file it never read.
    """
    loaded = getattr(data, "loaded_paths", None)
    if isinstance(loaded, dict) and loaded:
        return loaded.get(key)
    return getattr(data, attr, None)


def dataset_fields(data: Any, dataset_block: dict[str, Any]) -> dict[str, Any]:
    """``{name, sha256, source}`` from what the runner already knows about the data plugin.

    ``lock``: the plugin loaded a dataset by name (``params_init.dataset``). The
    sha256 is the hash of the prices.parquet the resolution served
    (:func:`quantbox.dataset_lock.resolve_dataset`, recorded as
    ``dataset.resolved``) — equal to the pin whenever one exists, since the
    loader refuses a mismatch; null only when the dataset was never resolved.
    ``inline``: everything else — paths in the config, or a plugin. The sha256 is
    the dataset plugin's manifest hash, else the content hash of the prices file
    the plugin actually loaded, else
    null (a source the runner cannot hash, e.g. synthetic or an API).
    """
    pinned = getattr(data, "dataset", None)
    if isinstance(pinned, str) and pinned:
        resolution = dataset_block.get("resolved")
        if not isinstance(resolution, dict):
            resolution = getattr(data, "dataset_resolution", None)
        sha = None
        if isinstance(resolution, dict):
            sha = resolution.get("actual_sha256") or resolution.get("sha256")
        return {"name": pinned, "sha256": sha, "source": "lock"}
    if dataset_block.get("tier") == "plugin":
        manifest = dataset_block.get("manifest") or {}
        return {"name": dataset_block.get("id"), "sha256": manifest.get("sha256"), "source": "inline"}
    prices_path = _effective_path(data, "prices", "prices_path")
    if prices_path:
        return {"name": Path(prices_path).name, "sha256": _sha256_file(prices_path), "source": "inline"}
    meta = getattr(data, "meta", None)
    return {"name": getattr(meta, "name", None), "sha256": None, "source": "inline"}


def funding_block(data: Any, notes: dict[str, Any]) -> dict[str, Any]:
    path = _effective_path(data, "funding_rates", "funding_rates_path")
    return {
        "modelled": bool((notes.get("funding") or {}).get("modelled", False)),
        "source_path": str(path) if path else None,
        "sha256": _sha256_file(path),
    }


def files_block(artifacts: dict[str, str], run_dir: str | Path) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for logical in CANONICAL_FILES:
        path = artifacts.get(logical)
        out[logical] = Path(os.path.relpath(path, run_dir)).as_posix() if path else None
    return out


def json_safe(obj: Any) -> Any:
    """``obj`` with every NaN / ±Infinity float replaced by ``None``.

    The manifest must be STRICT JSON (jq, JavaScript and most non-Python
    readers refuse ``NaN``/``Infinity``); an undefined metric reads as null.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def n_trials(cfg: dict[str, Any]) -> int | None:
    """``run.n_trials`` — a positive integer, or None when the config does not state it.

    Raises ``ValueError`` on anything else: the runner calls this BEFORE the
    run, so a typo costs nothing rather than a whole backtest.
    """
    value = (cfg.get("run") or {}).get("n_trials")
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"run.n_trials must be a positive integer, got {value!r}")
    return value
