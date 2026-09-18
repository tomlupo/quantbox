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


def _run(repo: Path, *, root: str | None = None, args: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["QUANTBOX_DATASETS_CATALOG"] = str(repo / "catalog.yaml")
    # A path that does not exist means "no root reachable" — deterministic whether or
    # not the machine running the tests happens to have a quantbox-datasets clone.
    env["QUANTBOX_DATASETS_ROOT"] = root if root is not None else str(repo / "no-such-root")
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "check_datasets_lock.py"), *args],
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
    # Whatever mode it chose, it must SAY which one, and the summary must agree with the
    # stage lines: a run that skipped a stage may not report it as checked, and vice
    # versa. Asserting only that the words appear would pass on a wholly skipped run.
    checked = result.stdout.rsplit("PASS — checked:", 1)[1].split(".")[0]
    for stage in ("names", "pins"):
        line = next(ln for ln in result.stdout.splitlines() if ln.strip().startswith(f"{stage}:"))
        assert ("SKIPPED" not in line) == (stage in checked), result.stdout


def test_good_lock_passes(tmp_path: Path) -> None:
    repo = _sandbox(tmp_path, f"crypto-spot-hourly: {GOOD_SHA}\n")
    result = _run(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS — checked: syntax, names." in result.stdout


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
    assert "maps to a nested block" in result.stdout


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


def test_finding_no_locks_is_a_failure(tmp_path: Path) -> None:
    """A check that read nothing must not exit like a clean one."""
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / "check_datasets_lock.py")
    (tmp_path / "catalog.yaml").write_text(CATALOG)
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "0 lock file(s)" in result.stdout


def test_a_skipped_stage_never_reads_as_a_full_pass(tmp_path: Path) -> None:
    """The summary names what ran, so a partial run cannot be mistaken for a clean one."""
    repo = _sandbox(tmp_path, f"crypto-spot-hourly: {GOOD_SHA}\n")
    (repo / "catalog.yaml").unlink()
    result = _run(repo)
    assert result.returncode == 0
    assert "NOT checked: names, pins" in result.stdout
    assert "All committed datasets.lock files pass" not in result.stdout


def test_require_catalog_fails_when_the_catalog_is_unreachable(tmp_path: Path) -> None:
    """CI fetches the catalog on purpose, so an unreachable one there is a broken check."""
    repo = _sandbox(tmp_path, f"crypto-spot-hourly: {GOOD_SHA}\n")
    (repo / "catalog.yaml").unlink()
    result = _run(repo, args=("--require-catalog",))
    assert result.returncode == 1
    assert "--require-catalog" in result.stdout


def test_require_catalog_still_passes_with_a_catalog(tmp_path: Path) -> None:
    repo = _sandbox(tmp_path, f"crypto-spot-hourly: {GOOD_SHA}\n")
    result = _run(repo, args=("--require-catalog",))
    assert result.returncode == 0, result.stdout


def test_discovery_reads_the_index_not_the_disk(tmp_path: Path) -> None:
    """Under a git hook GIT_INDEX_FILE/GIT_DIR are exported; stripped, git reads THIS tree.

    The sandbox is a real repo with one COMMITTED lock and one UNTRACKED broken lock: a
    discovery that honours the ambient GIT_* (or falls back to walking the disk) picks up
    the untracked file and fails, which is what binds the stripping itself.
    """
    repo = _sandbox(tmp_path, f"crypto-spot-hourly: {GOOD_SHA}\n")
    # A clean git env: run from a hook, the ambient GIT_* would point these at the
    # real repo, and its hooksPath would run this repo's pre-commit inside the sandbox.
    git_env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), "-c", "core.hooksPath=", "-c", "user.email=t@t", "-c", "user.name=t", *args],
            check=True,
            capture_output=True,
            text=True,
            env=git_env,
        ).stdout

    git("init", "-q")
    # Prove the sandbox owns this git BEFORE writing to an index: an escaped `add -A`
    # writes into whatever repo the ambient GIT_* names, which is not a test any more.
    assert Path(git("rev-parse", "--show-toplevel").strip()).resolve() == repo.resolve()
    git("add", "-A")
    git("commit", "-qm", "x")
    (repo / "research" / "untracked").mkdir()
    (repo / "research" / "untracked" / "datasets.lock").write_text("crypto-spot-hourly: not-a-sha\n")
    env = dict(os.environ)
    env["QUANTBOX_DATASETS_CATALOG"] = str(repo / "catalog.yaml")
    env["QUANTBOX_DATASETS_ROOT"] = str(repo / "no-such-root")
    env["GIT_DIR"] = str(REPO_ROOT / ".git")
    env["GIT_INDEX_FILE"] = str(REPO_ROOT / ".git" / "index")
    result = _run_with(repo, env)
    assert "1 lock file(s)" in result.stdout, result.stdout
    assert result.returncode == 0, result.stdout


def test_an_absent_in_git_true_artifact_is_a_pin_failure(tmp_path: Path) -> None:
    """A dataset the catalog says IS committed must resolve — absent, that is a bad pin."""
    repo = _sandbox(tmp_path, f"etf-daily: {GOOD_SHA}\n")
    result = _run_with(repo, _with_stub_datasets(repo, {}))
    assert result.returncode == 1, result.stdout
    assert "does not resolve" in result.stdout


def test_a_catalog_that_is_not_a_catalog_reads_as_no_catalog(tmp_path: Path) -> None:
    """CI FETCHES the catalog, so any HTTP 200 body must degrade, never traceback."""
    repo = _sandbox(tmp_path, f"crypto-spot-hourly: {GOOD_SHA}\n")
    (repo / "catalog.yaml").write_text("<html>not a catalog</html>\n")
    result = _run(repo)
    assert result.returncode == 0, result.stdout
    assert "names: SKIPPED" in result.stdout
    assert "Traceback" not in result.stderr
    assert _run(repo, args=("--require-catalog",)).returncode == 1
