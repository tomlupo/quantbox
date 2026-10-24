"""Resolve a dataset by name: the lock pins it, the environment roots it (TOM-1349).

A config names a dataset and nothing else. Where its bytes live and which build they
must be come from two places that are NOT the config and NOT the working directory:

- the root: ``$QUANTBOX_DATASETS_ROOT``, else the clone quantbox-datasets was installed
  from (its own ``datasets_root()``); or, for a dataset frozen inside a research line
  (TOM-2232, D5), the ``datasets/`` directory beside the line's ``datasets.lock``;
- the pin: the ``datasets.lock`` nearest the config (``quantbox-datasets pin <name>``),
  the sha256 of the manifest's ``pin_file`` (``prices.parquet`` unless it names one).

A manifest may say how the dataset is SERVED by name (:func:`serve_spec`): which table
is its prices, whether that table holds prices or returns, and its bar frequency. A
manifest without that block is served exactly as before.

:func:`resolve_dataset` is the one answer both ``quantbox dataset resolve -c <config> --json``
and the runner use (both pick the lock with :func:`lock_for_config`), and the runner
records it verbatim in ``run_manifest.json``.

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

from .exceptions import ConfigValidationError, ValidationFinding

LOCK_NAME = "datasets.lock"
ROOT_ENV = "QUANTBOX_DATASETS_ROOT"


class DatasetResolveError(RuntimeError):
    """A dataset name could not be resolved to a root, or its bytes are not the pinned build."""


class DatasetServeError(DatasetResolveError, ConfigValidationError):
    """A manifest's ``serve`` block cannot be served: an unknown kind or frequency, a missing
    or altered table, a gap in a returns series. Refused, never served as empty data (TOM-2232)."""

    def __init__(self, message: str) -> None:
        ConfigValidationError.__init__(self, message, [ValidationFinding("error", message, code="dataset_serve")])


#: ``serve.kind``: ``prices`` is served as is; ``returns`` as a total-return index built from them.
SERVE_KINDS = ("prices", "returns")
#: The frequencies a ``serve`` block may declare, by their spelling in quantbox.frequency.
SERVE_FREQUENCIES = {"daily": "1d", "1d": "1d", "weekly": "1w", "1w": "1w", "monthly": "1M", "1M": "1M"}
#: How a ``kind: returns`` table becomes prices (quantbox.dataset.total_return_index); recorded in the resolution.
TOTAL_RETURN_CONSTRUCTION = (
    "total-return index: 1.0 one bar before each series' first return, then the cumulative product of (1 + return)"
)
#: Where a line keeps the datasets it froze itself: beside its datasets.lock (D5).
LINE_DATASETS_DIR = "datasets"


def find_lock(start: str | Path | None = None) -> Path | None:
    """The nearest datasets.lock in *start* (default: cwd) or any parent."""
    here = Path(start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        if (directory / LOCK_NAME).is_file():
            return directory / LOCK_NAME
    return None


def lock_for_config(config_path: str | Path) -> Path | None:
    """The datasets.lock a run of *config_path* binds: the nearest above the config.

    None means no lock above the config; the data plugin then falls back to the nearest
    one above the cwd. The runner and ``quantbox dataset resolve --config`` both call
    this, so they cannot disagree on which lock pins a by-name dataset.
    """
    return find_lock(Path(config_path).resolve().parent)


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


def _one_name(value: Any) -> bool:
    """True when *value* names one entry in a directory: no separator, not ``.`` or ``..``, not absolute."""
    return (
        isinstance(value, str)
        and bool(value)
        and value not in (".", "..")
        and "/" not in value
        and "\\" not in value
        and not Path(value).is_absolute()
    )


def _pin_file(manifest: dict[str, Any]) -> str:
    """The file a pin hashes: the manifest's ``pin_file``, else prices.parquet (quantbox-datasets ADR-0015).

    A frozen dataset need not hold a prices.parquet (sbbi holds annual and monthly
    tables): before TOM-2232 such a dataset resolved with no sha256 at all.
    """
    named = manifest.get("pin_file")
    return named if _one_name(named) else "prices.parquet"


def _inside(
    path: Path, file_name: str, name: str, what: str, error: type[DatasetResolveError] = DatasetResolveError
) -> Path:
    """``path / file_name``, refused when it resolves outside the dataset directory *path*.

    A name that passes :func:`_one_name` can still be a symlink: without this check a
    manifest-named table, the pin file, the funding file or the manifest itself could be
    hashed and served from anywhere on disk (PR #303 review, finding 103). A file that
    does not exist is returned as is; its caller reports it missing.
    """
    file = path / file_name
    if not (file.exists() or file.is_symlink()):
        return file
    real, home = file.resolve(), path.resolve()
    if not real.is_relative_to(home):
        raise error(f"dataset {name!r}: {what} {file} resolves to {real}, outside the dataset directory {home}")
    return file


def _manifest(path: Path, name: str) -> dict[str, Any]:
    file = _inside(path, "manifest.yaml", name, "manifest")
    loaded = yaml.safe_load(file.read_text()) if file.is_file() else None
    return loaded if isinstance(loaded, dict) else {}


def serve_spec(manifest: dict[str, Any], name: str) -> dict[str, Any] | None:
    """The manifest's ``serve`` block, checked; None when it has none (served as before).

    ``table`` (default ``prices``) names ``<table>.parquet``, the frame served as prices;
    ``kind`` (default ``prices``) is ``prices`` or ``returns``; ``frequency`` (optional)
    is the bar, which the run's StrategyContext follows; ``columns`` (optional) the series
    served, in order. Anything else is refused by name.
    """
    block = manifest.get("serve")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise DatasetServeError(f"dataset {name!r}: manifest serve must be a mapping, got {block!r}")
    unknown = sorted(set(block) - {"table", "kind", "frequency", "columns"})
    if unknown:
        raise DatasetServeError(f"dataset {name!r}: manifest serve has unknown keys {unknown}")
    table = block.get("table", "prices")
    if not _one_name(table):
        raise DatasetServeError(f"dataset {name!r}: serve.table must name one table in the dataset, got {table!r}")
    kind = block.get("kind", "prices")
    if kind not in SERVE_KINDS:
        raise DatasetServeError(f"dataset {name!r}: serve.kind {kind!r} is not one of {list(SERVE_KINDS)}")
    frequency = block.get("frequency")
    if frequency is not None and frequency not in SERVE_FREQUENCIES:
        raise DatasetServeError(
            f"dataset {name!r}: serve.frequency {frequency!r} is not one of {sorted(SERVE_FREQUENCIES)}"
        )
    columns = block.get("columns")
    if columns is not None and not (isinstance(columns, list) and columns and all(isinstance(c, str) for c in columns)):
        raise DatasetServeError(f"dataset {name!r}: serve.columns must be a non-empty list of names")
    return {
        "table": table,
        "kind": kind,
        "frequency": SERVE_FREQUENCIES[frequency] if frequency is not None else None,
        "columns": columns,
    }


def _served_table(path: Path, manifest: dict[str, Any], spec: dict[str, Any], name: str) -> dict[str, Any]:
    """The table *spec* serves: its file and sha256, refused when absent or not the bytes the manifest vouches for.

    The pin covers only the pin file; a served table that is not the pin file is held to
    its ``content_hashes`` entry, so an edited table cannot ride under an unchanged pin.
    """
    file = _inside(path, f"{spec['table']}.parquet", name, "serve.table", DatasetServeError)
    sha = _sha256(file)
    if sha is None:
        raise DatasetServeError(f"dataset {name!r}: serve.table {spec['table']!r} but {file} does not exist")
    vouched = ((manifest.get("content_hashes") or {}).get(file.name) or {}).get("sha256")
    if file.name != _pin_file(manifest) and vouched != sha:
        raise DatasetServeError(
            f"dataset {name!r}: {file} has sha256 {sha}, but its manifest content_hashes "
            f"{'vouch for ' + vouched if vouched else 'carry no entry for it'}; the served table must be vouched for"
        )
    served = {**spec, "file": str(file), "sha256": sha}
    if spec["kind"] == "returns":
        served["construction"] = TOTAL_RETURN_CONSTRUCTION
    return served


def served_bytes(served: dict[str, Any], name: str) -> bytes:
    """The served table's bytes, read ONCE, refused unless they hash to the sha256 its resolution recorded.

    Resolution and loading are two steps in the data plugin; the caller parses exactly
    these bytes, so a table replaced between them is refused rather than served under
    the resolution's hash (PR #303 review, finding 102).
    """
    file = Path(served["file"])
    try:
        data = file.read_bytes()
    except OSError as exc:
        raise DatasetServeError(f"dataset {name!r}: {file} cannot be read to serve it ({exc})") from exc
    sha = hashlib.sha256(data).hexdigest()
    if sha != served["sha256"]:
        raise DatasetServeError(
            f"dataset {name!r}: {file} now has sha256 {sha}, not the bytes it was resolved at "
            f"({served['sha256']}); it changed after resolution"
        )
    return data


def _locate(name: str, lock_path: Path | None) -> tuple[Path, Path]:
    """(root, dataset directory) for *name*: under the datasets root, else in the line beside the lock.

    A line freezes the data it fetched under ``<lock dir>/datasets/<name>`` and keeps the
    same name when the dataset moves to quantbox-datasets at promotion (TOM-2232, D5).
    The root wins whenever it holds the name, so a line copy beside a lock never shadows
    it (a stray copy is ignored, TOM-1349); the pin then decides whether its bytes serve.
    """
    try:
        root: Path | None = datasets_root()
    except DatasetResolveError:
        root = None
    if root is not None and (root / name).is_dir():
        return root, root / name
    line_root = lock_path.parent / LINE_DATASETS_DIR if lock_path is not None else None
    if line_root is not None and (line_root / name).is_dir():
        return line_root, line_root / name
    if root is None:
        raise DatasetResolveError(
            f"dataset {name!r} not found in the line ({line_root or 'no datasets.lock'}), and no datasets root: "
            f"set {ROOT_ENV} to the quantbox-datasets clone's datasets/ directory"
        )
    where = f"{root} (from ${ROOT_ENV} or the clone)" + (f" or {line_root}" if line_root is not None else "")
    raise DatasetResolveError(f"dataset {name!r} not found under {where}")


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
    or None), ``actual_sha256`` (of the pin file that will be read: prices.parquet, or the
    manifest's ``pin_file``, then also recorded as ``pin_file``), ``matches`` (None when
    unpinned), ``restored`` (served from git history), ``market`` and ``funding_rates``
    (its file, or None); with a manifest ``serve`` block, ``serve`` (:func:`serve_spec`
    plus the served ``file`` and its ``sha256``). A mismatch is reported in ``matches``
    and ``error``, never raised here — :func:`require_match` is the gate; a ``serve``
    block that cannot be served raises :class:`DatasetServeError`.
    """
    # A name is one directory under the root: an absolute name, a separator or ``..``
    # would read (and record as pinned) bytes from outside $QUANTBOX_DATASETS_ROOT.
    if not _one_name(name):
        raise DatasetResolveError(f"dataset name {name!r} must be one directory name under the datasets root")
    lock_path = Path(lock).resolve() if lock is not None else find_lock(start)
    pinned = None
    if lock_path is not None:
        pins = yaml.safe_load(lock_path.read_text()) or {}
        # A non-string value is refused, not skipped: YAML reads an all-digit sha as an int.
        if not (isinstance(pins, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in pins.items())):
            raise DatasetResolveError(f"{lock_path} is not a mapping of dataset name to sha256")
        pinned = pins.get(name) or None

    root, path = _locate(name, lock_path)
    # Read once: the pin file and the serve block come from the same manifest bytes.
    manifest = _manifest(path, name)
    pin_name = _pin_file(manifest)
    on_disk = _sha256(_inside(path, pin_name, name, "pin file"))
    actual, restored, error = on_disk, False, None
    if pinned and on_disk != pinned:
        restored_path, why_not = _restore_pinned(path, pinned)
        if restored_path is not None and _sha256(_inside(restored_path, pin_name, name, "pin file")) == pinned:
            path, actual, restored = restored_path, pinned, True
            manifest = _manifest(path, name)
        else:
            error = (
                f"dataset {name!r}: {path / pin_name} has sha256 {on_disk or '(no file)'}, "
                f"but {lock_path} pins {pinned}; {why_not or 'the restored build does not hash to the pin'}"
            )

    funding = _inside(path, "funding_rates.parquet", name, "funding_rates")
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
    if pin_name != "prices.parquet":
        out["pin_file"] = pin_name
    spec = serve_spec(manifest, name)
    if spec is not None and error is None:
        out["serve"] = _served_table(path, manifest, spec, name)
    if error:
        out["error"] = error
    return out


def require_match(resolved: dict[str, Any]) -> dict[str, Any]:
    """*resolved*, or a DatasetResolveError naming both shas when its bytes are not the pin."""
    if resolved["matches"] is False:
        raise DatasetResolveError(resolved["error"])
    return resolved
