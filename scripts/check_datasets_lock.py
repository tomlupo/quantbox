"""Validate every committed ``datasets.lock`` in this repo.

A lock-only PR is classified trivial and skips the independent review gate, so the
file it changes has to be checked by CI instead. This is that check.

Three parts, in order:

1. SYNTAX — each lock parses as YAML and is a flat mapping of dataset name to a
   64-hex sha256. Empty values, duplicate keys, non-hex digests and nested blocks
   are all errors.
2. NAMES — every pinned dataset exists in quantbox-datasets' catalog.
3. PINS — with a datasets root reachable AND a quantbox_datasets that carries
   ``lock`` importable, each pinned sha256 resolves (the pin loads). Without
   either, this part is SKIPPED and said so out loud.

Know what that buys and what it does not. CI has no datasets root — the artifacts
are far too large to pull on every PR — so there it runs stages 1 and 2 only, and a
well-formed but WRONG sha256 passes it. Stage 3 is the tier that catches that one,
and it runs where the data is: a box with $QUANTBOX_DATASETS_ROOT set and a
quantbox-datasets new enough to expose ``quantbox_datasets.lock``.

Datasets the catalog declares ``in_git: false`` (e.g. crypto-spot-hourly) live only
under ``$QUANTBOX_DATASETS_ROOT`` and cannot be restored from git history, so a pin on
one is verified when the artifact is on disk and reported as unverifiable otherwise —
never a failure.

The mode actually run is always printed, and the final line names the stages that ran:
"could not check" never passes silently, and never as the sentence a full run prints.

USAGE:
    uv run python scripts/check_datasets_lock.py [--require-catalog]

    --require-catalog           fail when no catalog is reachable instead of skipping
                                the name stage. CI passes it: there the catalog is
                                fetched deliberately, so an unreachable one means the
                                fetch is broken, not that the check may run partially.

    QUANTBOX_DATASETS_CATALOG   path to quantbox-datasets' catalog.yaml (CI fetches it)
    QUANTBOX_DATASETS_ROOT      the datasets/ directory of a quantbox-datasets clone
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCK_NAME = "datasets.lock"
SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
TOP_LEVEL_KEY_RE = re.compile(r"\A([^\s#][^:]*):")


def find_locks() -> list[Path]:
    """Every committed datasets.lock, or — outside a git repo — every one on disk."""

    def git(*args: str) -> str:
        # Strip GIT_* from the environment: run from a git hook, GIT_DIR and friends
        # are exported and would point this at the hook's repo rather than at ROOT.
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        return subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True, env=env
        ).stdout

    try:
        if Path(git("rev-parse", "--show-toplevel").strip()).resolve() != ROOT:
            raise subprocess.CalledProcessError(1, "git")  # ROOT is not a repo of its own
        listed = git("ls-files").splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return sorted(ROOT.rglob(LOCK_NAME))
    return sorted(ROOT / p for p in listed if Path(p).name == LOCK_NAME)


def check_syntax(path: Path) -> tuple[dict[str, str], list[str]]:
    """Parse one lock. Returns its pins (empty when unusable) and any errors."""
    rel = path.relative_to(ROOT)
    text = path.read_text()

    # YAML keeps the LAST of duplicate keys, so a duplicate pin is invisible after
    # parsing — catch it on the raw text instead.
    seen: set[str] = set()
    errors: list[str] = []
    for line in text.splitlines():
        match = TOP_LEVEL_KEY_RE.match(line)
        if match:
            key = match.group(1).strip()
            if key in seen:
                errors.append(f"{rel}: duplicate entry for {key!r}")
            seen.add(key)

    try:
        parsed: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return {}, [*errors, f"{rel}: not valid YAML: {exc}"]

    if parsed is None:
        return {}, errors  # an empty lock pins nothing; that is not a violation
    if not isinstance(parsed, dict):
        return {}, [*errors, f"{rel}: not a mapping of dataset name to sha256 (got {type(parsed).__name__})"]

    pins: dict[str, str] = {}
    for name, value in parsed.items():
        if not isinstance(name, str):
            errors.append(f"{rel}: dataset name {name!r} is not a string")
            continue
        if isinstance(value, dict | list):
            errors.append(f"{rel}: {name} maps to a nested block — a lock is flat, name -> sha256")
            continue
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{rel}: {name} has an empty or non-string sha256 ({value!r})")
            continue
        if not SHA256_RE.match(value.strip()):
            errors.append(f"{rel}: {name} is pinned to {value!r} — must be a 64-char hex sha256")
            continue
        pins[name] = value.strip()
    return pins, errors


def load_catalog() -> tuple[dict[str, Any] | None, str]:
    """quantbox-datasets' catalog entries and where they came from, or (None, why)."""
    explicit = os.environ.get("QUANTBOX_DATASETS_CATALOG", "")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            return None, f"QUANTBOX_DATASETS_CATALOG={explicit} does not exist"
        entries = (yaml.safe_load(path.read_text()) or {}).get("datasets") or {}
        return (entries, str(path)) if entries else (None, f"{path} lists no datasets")

    try:  # an editable install or a clone on sys.path carries catalog.yaml
        from quantbox_datasets.catalog import load_catalog as _load_catalog

        entries = _load_catalog()
        if entries:
            return entries, "installed quantbox_datasets"
    except Exception:  # noqa: BLE001 — absence is a mode, not a failure
        pass

    root = datasets_root()
    if root is not None and (root.parent / "catalog.yaml").is_file():
        path = root.parent / "catalog.yaml"
        entries = (yaml.safe_load(path.read_text()) or {}).get("datasets") or {}
        if entries:
            return entries, str(path)
    return None, "no catalog.yaml reachable (set QUANTBOX_DATASETS_CATALOG)"


def datasets_root() -> Path | None:
    """The reachable datasets root, or None."""
    env_root = os.environ.get("QUANTBOX_DATASETS_ROOT", "")
    if env_root:
        return Path(env_root) if Path(env_root).is_dir() else None
    try:
        from quantbox_datasets.lock import datasets_root as _datasets_root

        root = _datasets_root()
    except Exception:  # noqa: BLE001 — no clone installed
        return None
    return root if root.is_dir() else None


def main(argv: list[str] | None = None) -> int:
    # --require-catalog: the caller states the name stage MUST run. CI passes it, because
    # there the catalog is fetched deliberately and an unreachable one means the fetch,
    # not the environment, is broken — and a stage that silently stopped running is how a
    # check goes green having read nothing.
    require_catalog = "--require-catalog" in (argv if argv is not None else sys.argv[1:])

    locks = find_locks()
    print(f"datasets.lock check — {len(locks)} lock file(s) under {ROOT}")
    if not locks:
        # A check that read nothing must not exit like a clean one: this repo commits
        # locks, so finding none means discovery is looking in the wrong place (or the
        # locks were deleted), never that everything is fine.
        print(f"\nFound no {LOCK_NAME} to check — this repo commits them, so this is a defect.\n")
        return 1

    errors: list[str] = []
    pins_by_lock: dict[Path, dict[str, str]] = {}
    for lock in locks:
        pins, lock_errors = check_syntax(lock)
        pins_by_lock[lock] = pins
        errors.extend(lock_errors)

    catalog, catalog_source = load_catalog()
    skipped: list[str] = []
    if catalog is None:
        print(f"  names: SKIPPED — {catalog_source}")
        skipped.append("names")
        if require_catalog:
            errors.append(f"--require-catalog was given and no catalog was reachable: {catalog_source}")
    else:
        print(f"  names: checked against {catalog_source} ({len(catalog)} datasets)")
        for lock, pins in pins_by_lock.items():
            for name in pins:
                if name not in catalog:
                    errors.append(
                        f"{lock.relative_to(ROOT)}: {name} is not a dataset in quantbox-datasets' catalog "
                        f"(known: {', '.join(sorted(catalog))})"
                    )

    root = datasets_root()
    try:
        from quantbox_datasets.lock import load as load_dataset
    except Exception:  # noqa: BLE001
        load_dataset = None  # type: ignore[assignment]

    if root is None:
        env_root = os.environ.get("QUANTBOX_DATASETS_ROOT", "")
        why = f"QUANTBOX_DATASETS_ROOT={env_root} is not a directory" if env_root else "no datasets root reachable"
        print(f"  pins:  SKIPPED — {why} (set QUANTBOX_DATASETS_ROOT)")
        skipped.append("pins")
    elif load_dataset is None:
        print(f"  pins:  SKIPPED — quantbox_datasets is not importable (root {root} is reachable)")
        skipped.append("pins")
    else:
        print(f"  pins:  resolved against {root}")
        not_in_git = {name for name, entry in (catalog or {}).items() if entry.get("in_git") is False}
        for lock, pins in pins_by_lock.items():
            for name, sha in sorted(pins.items()):
                if not (root / name).is_dir():
                    # A dataset absent from this root cannot be verified here. For an
                    # `in_git: false` one that is the whole story — its artifact is never
                    # committed, so no commit can restore it — and this does not depend on
                    # the catalog being reachable to say so.
                    why = "in_git: false" if name in not_in_git else "not built here"
                    print(f"         - {name}: UNVERIFIABLE — {why}, no artifact under {root}")
                    continue
                try:
                    load_dataset(name, root=root, sha256=sha)
                except Exception as exc:  # noqa: BLE001 — DataPinMismatch and friends
                    errors.append(f"{lock.relative_to(ROOT)}: pin for {name} does not resolve: {exc}")
                else:
                    print(f"         - {name}: ok")

    if errors:
        print(f"\n{len(errors)} datasets.lock violation(s):\n")
        for error in errors:
            print(f"  - {error}")
        print()
        return 1
    # Never the same sentence over a run that skipped a stage: the summary says what
    # was actually checked, so a partial run cannot be mistaken for a full one.
    ran = [stage for stage in ("syntax", "names", "pins") if stage not in skipped]
    summary = f"\nPASS — checked: {', '.join(ran)}."
    if skipped:
        summary += f" NOT checked: {', '.join(skipped)} (see the SKIPPED lines above)."
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
