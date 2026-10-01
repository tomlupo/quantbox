"""Resolve a dataset by name: the lock pins it, the environment roots it (TOM-1349).

A config names a dataset and nothing else. Where its bytes live and which build they
must be come from two places that are NOT the config and NOT the working directory:

- the root: ``$QUANTBOX_DATASETS_ROOT``, else the clone quantbox-datasets was installed
  from (its own ``datasets_root()``);
- the pin: the ``datasets.lock`` nearest the config (``quantbox-datasets pin <name>``).

:func:`resolve_dataset` is the one answer both ``quantbox dataset resolve --json`` and
the runner use, and the runner records it verbatim in ``run_manifest.json``.

quantbox does not depend on quantbox-datasets. Resolution needs nothing from it beyond
the clone fallback above; restoring a pinned build from git history (quantbox-datasets
ADR-0004) is used when it is installed, and a mismatch is reported when it is not.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

LOCK_NAME = "datasets.lock"
ROOT_ENV = "QUANTBOX_DATASETS_ROOT"


class DatasetResolveError(RuntimeError):
    """A dataset name could not be resolved to a root, or its bytes are not the pinned build."""


def find_lock(start: str | Path | None = None) -> Path | None:
    """The nearest datasets.lock in *start* (default: cwd) or any parent."""
    here = Path(start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        if (directory / LOCK_NAME).is_file():
            return directory / LOCK_NAME
    return None


def datasets_root() -> Path:
    """``$QUANTBOX_DATASETS_ROOT``, else the installed quantbox-datasets clone's ``datasets/``."""
    env_root = os.environ.get(ROOT_ENV, "")
    if env_root:
        return Path(env_root)
    try:
        from quantbox_datasets.lock import datasets_root as clone_root
    except ImportError:
        clone_root = None
    if clone_root is not None:
        try:
            return Path(clone_root())
        except FileNotFoundError:
            pass
    raise DatasetResolveError(f"no datasets root: set {ROOT_ENV} to the quantbox-datasets clone's datasets/ directory")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _restore_pinned(path: Path, pinned: str) -> tuple[Path | None, str]:
    """The pinned build restored from quantbox-datasets' git history, or (None, why not)."""
    try:
        from quantbox_datasets.dataset import resolve_pinned_dataset
    except ImportError as exc:
        return None, f"no restore from git history: quantbox_datasets.dataset cannot be imported ({exc})"
    try:
        return Path(resolve_pinned_dataset(path, pinned)), ""
    except Exception as exc:  # noqa: BLE001 — the reason is carried into the mismatch error
        return None, f"no restore from git history: {exc}"


def resolve_dataset(name: str, *, lock: str | Path | None = None, start: str | Path | None = None) -> dict[str, Any]:
    """What the runner will read for *name*.

    *lock* names the lock file; without it the nearest ``datasets.lock`` above *start*
    (default: cwd) is used, and with none found the dataset is unpinned.

    Returns a JSON-safe dict: ``name``, ``root``, ``path``, ``lock``, ``sha256`` (the pin,
    or None), ``actual_sha256`` (of the prices.parquet that will be read), ``matches``
    (None when unpinned), ``restored`` (served from git history), ``market`` and
    ``funding_rates`` (its file, or None). A mismatch is reported in ``matches`` and
    ``error``, never raised here — :func:`require_match` is the gate.
    """
    # A name is one directory under the root: an absolute name, a separator or ``..``
    # would read (and record as pinned) bytes from outside $QUANTBOX_DATASETS_ROOT.
    if not name or name in (".", "..") or "/" in name or "\\" in name or Path(name).is_absolute():
        raise DatasetResolveError(f"dataset name {name!r} must be one directory name under the datasets root")
    root = datasets_root()
    lock_path = Path(lock).resolve() if lock is not None else find_lock(start)
    pinned = None
    if lock_path is not None:
        pins = yaml.safe_load(lock_path.read_text()) or {}
        # A non-string value is refused, not skipped: YAML reads an all-digit sha as an int.
        if not (isinstance(pins, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in pins.items())):
            raise DatasetResolveError(f"{lock_path} is not a mapping of dataset name to sha256")
        pinned = pins.get(name) or None

    path = root / name
    if not path.is_dir():
        raise DatasetResolveError(f"dataset {name!r} not found under {root} (from ${ROOT_ENV} or the clone)")
    on_disk = _sha256(path / "prices.parquet")
    actual, restored, error = on_disk, False, None
    if pinned and on_disk != pinned:
        restored_path, why_not = _restore_pinned(path, pinned)
        if restored_path is not None and _sha256(restored_path / "prices.parquet") == pinned:
            path, actual, restored = restored_path, pinned, True
        else:
            error = (
                f"dataset {name!r}: {path / 'prices.parquet'} has sha256 {on_disk or '(no file)'}, "
                f"but {lock_path} pins {pinned}; {why_not or 'the restored build does not hash to the pin'}"
            )

    manifest_file = path / "manifest.yaml"
    manifest = (yaml.safe_load(manifest_file.read_text()) or {}) if manifest_file.is_file() else {}
    funding = path / "funding_rates.parquet"
    out: dict[str, Any] = {
        "name": name,
        "root": str(root),
        "path": str(path),
        "lock": str(lock_path) if lock_path is not None else None,
        "sha256": pinned,
        "actual_sha256": actual,
        "matches": None if not pinned else error is None,
        "restored": restored,
        "market": manifest.get("market"),
        "funding_rates": str(funding) if funding.is_file() else None,
    }
    if error:
        out["error"] = error
    return out


def require_match(resolved: dict[str, Any]) -> dict[str, Any]:
    """*resolved*, or a DatasetResolveError naming both shas when its bytes are not the pin."""
    if resolved["matches"] is False:
        raise DatasetResolveError(resolved["error"])
    return resolved
