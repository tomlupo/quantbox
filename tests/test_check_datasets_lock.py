"""Smoke tests for scripts/check_datasets_lock.py — the datasets.lock validator.

A lock-only PR skips the review gate, so this check is the only thing standing
between a bad pin and main. These tests are what stands behind the check.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_datasets_lock.py"

GOOD_SHA = "f5d95f6635cc383332e0c425d1af2dc893acf074d3deeb63f8aaf3ad4601bc4f"
CATALOG = """
datasets:
  crypto-spot-hourly:
    description: hourly crypto
    in_git: false
  etf-daily:
    description: daily etfs
  fx-daily:
    description: daily fx
"""


def _sandbox(tmp_path: Path, lock_body: str) -> Path:
    """A fake repo with the script, a catalog and one research line's lock."""
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / "check_datasets_lock.py")
    (tmp_path / "catalog.yaml").write_text(CATALOG)
    line = tmp_path / "research" / "a_line"
    line.mkdir(parents=True)
    (line / "datasets.lock").write_text(lock_body)
    return tmp_path


def _run(repo: Path, *, root: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["QUANTBOX_DATASETS_CATALOG"] = str(repo / "catalog.yaml")
    # A path that does not exist means "no root reachable" — deterministic whether or
    # not the machine running the tests happens to have a quantbox-datasets clone.
    env["QUANTBOX_DATASETS_ROOT"] = root if root is not None else str(repo / "no-such-root")
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "check_datasets_lock.py")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_real_repo_passes_its_own_locks() -> None:
    """Whatever mode it runs in, this repo's committed locks must pass."""
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"check failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"


def test_good_lock_passes(tmp_path: Path) -> None:
    repo = _sandbox(tmp_path, f"crypto-spot-hourly: {GOOD_SHA}\n")
    result = _run(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All committed datasets.lock files pass." in result.stdout


def test_no_root_mode_is_announced_and_passes(tmp_path: Path) -> None:
    """No data root: parts 1 and 2 still run, part 3 says out loud that it did not."""
    repo = _sandbox(tmp_path, f"crypto-spot-hourly: {GOOD_SHA}\n")
    result = _run(repo)
    assert result.returncode == 0
    assert "pins:  SKIPPED" in result.stdout
    assert "names: checked against" in result.stdout


def test_malformed_lock_fails(tmp_path: Path) -> None:
    """Short digest, empty value, nested block and a duplicate key are all errors."""
    repo = _sandbox(
        tmp_path,
        f"etf-daily: abc123\nfx-daily:\ncrypto-spot-hourly:\ncrypto-spot-hourly: {GOOD_SHA}\nnested-block:\n  sha256: deadbeef\n",
    )
    result = _run(repo)
    assert result.returncode == 1
    assert "must be a 64-char hex sha256" in result.stdout
    assert "duplicate entry" in result.stdout
    assert "empty or non-string sha256" in result.stdout


def test_unknown_dataset_name_fails(tmp_path: Path) -> None:
    repo = _sandbox(tmp_path, f"not-a-dataset: {GOOD_SHA}\n")
    result = _run(repo)
    assert result.returncode == 1
    assert "not a dataset in quantbox-datasets' catalog" in result.stdout


def test_catalog_unavailable_is_announced(tmp_path: Path) -> None:
    """A missing catalog degrades loudly rather than passing as if it had checked."""
    repo = _sandbox(tmp_path, f"not-a-dataset: {GOOD_SHA}\n")
    (repo / "catalog.yaml").unlink()
    result = _run(repo)
    assert "names: SKIPPED" in result.stdout


def test_discovery_ignores_an_ambient_git_dir(tmp_path: Path) -> None:
    """Run from a git hook, GIT_DIR points at the hook's repo — not at the tree to check."""
    repo = _sandbox(tmp_path, f"crypto-spot-hourly: {GOOD_SHA}\n")
    env = dict(os.environ)
    env["QUANTBOX_DATASETS_CATALOG"] = str(repo / "catalog.yaml")
    env["QUANTBOX_DATASETS_ROOT"] = str(repo / "no-such-root")
    env["GIT_DIR"] = str(REPO_ROOT / ".git")
    env["GIT_WORK_TREE"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, str(repo / "scripts" / "check_datasets_lock.py")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "1 lock file(s)" in result.stdout, result.stdout
    assert result.returncode == 0, result.stdout


STUB_LOCK = '''
from pathlib import Path


def datasets_root(root=None):
    return Path(root) if root else Path("/nonexistent")


def load(name, *, root=None, sha256=None, **_):
    """Stand in for quantbox_datasets.lock.load: the pin resolves or it does not."""
    recorded = (Path(root) / name / "prices.sha256").read_text().strip()
    if sha256 and sha256 != recorded:
        raise ValueError(f"{name} is pinned to {sha256} but this root holds {recorded}")
    return object()
'''


def _with_stub_datasets(repo: Path, built: dict[str, str]) -> dict[str, str]:
    """Env with a stub quantbox_datasets importable and a root holding *built* datasets."""
    pkg = repo / "stub" / "quantbox_datasets"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "lock.py").write_text(STUB_LOCK)
    root = repo / "datasets-root"
    for name, sha in built.items():
        (root / name).mkdir(parents=True)
        (root / name / "prices.sha256").write_text(sha)
    root.mkdir(exist_ok=True)
    env = dict(os.environ)
    env["QUANTBOX_DATASETS_CATALOG"] = str(repo / "catalog.yaml")
    env["QUANTBOX_DATASETS_ROOT"] = str(root)
    env["PYTHONPATH"] = str(repo / "stub")
    return env


def _run_with(repo: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "check_datasets_lock.py")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_resolving_pin_is_verified(tmp_path: Path) -> None:
    """With a root reachable the check actually loads the pin — stage 3 runs."""
    repo = _sandbox(tmp_path, f"crypto-spot-hourly: {GOOD_SHA}\n")
    result = _run_with(repo, _with_stub_datasets(repo, {"crypto-spot-hourly": GOOD_SHA}))
    assert result.returncode == 0, result.stdout
    assert "pins:  resolved against" in result.stdout
    assert "crypto-spot-hourly: ok" in result.stdout


def test_a_pin_that_does_not_resolve_fails(tmp_path: Path) -> None:
    """A well-formed sha256 that no build matches is the defect a re-pin introduces."""
    repo = _sandbox(tmp_path, f"crypto-spot-hourly: {GOOD_SHA}\n")
    result = _run_with(repo, _with_stub_datasets(repo, {"crypto-spot-hourly": "0" * 64}))
    assert result.returncode == 1
    assert "does not resolve" in result.stdout


def test_an_absent_in_git_false_artifact_is_unverifiable_not_a_failure(tmp_path: Path) -> None:
    """crypto-spot-hourly is never committed: absent from the root it is reported, not failed."""
    repo = _sandbox(tmp_path, f"crypto-spot-hourly: {GOOD_SHA}\n")
    result = _run_with(repo, _with_stub_datasets(repo, {}))
    assert result.returncode == 0, result.stdout
    assert "crypto-spot-hourly: UNVERIFIABLE — in_git: false" in result.stdout


def test_an_absent_artifact_is_unverifiable_without_the_catalog(tmp_path: Path) -> None:
    """Root reachable, catalog not: an absent artifact must not become a hard failure."""
    repo = _sandbox(tmp_path, f"crypto-spot-hourly: {GOOD_SHA}\n")
    env = _with_stub_datasets(repo, {})
    (repo / "catalog.yaml").unlink()
    result = _run_with(repo, env)
    assert result.returncode == 0, result.stdout
    assert "UNVERIFIABLE" in result.stdout
