"""Tests for scripts/check_pinned_shas.py — every quantbox SHA a lab line pins
must stay reachable from a branch or tag on origin (TOM-1340).

Each test builds a throwaway "origin" (bare repo) plus a clone that plays the
quantbox checkout, and a fake lab tree that pins SHAs the way the real labs do:
``quantbox.git@<sha>`` in a line's pyproject.toml and ``quantbox.git?rev=..#<sha>``
in its uv.lock.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_pinned_shas.py"
FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def _no_hook_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under the pre-push hook GIT_DIR etc. are exported, and would aim every git
    call here at the real quantbox repo. The script scrubs them itself; the test
    helpers rely on this.

    The box's own git config is shut out too, so every box runs the same git: a
    developer's ``init.defaultBranch=main`` once hid that CI's git (no such key,
    so ``master``) gave the bare origin a HEAD naming a branch that never exists."""
    for k in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def _git(cwd: Path, *args: str) -> str:
    env_args = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"]
    out = subprocess.run(["git", *env_args, *args], cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def _commit(repo: Path, msg: str) -> str:
    _git(repo, "commit", "--allow-empty", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    origin = tmp_path / "origin.git"
    # -b main: origin's HEAD must name the branch the tests push, or a --depth=1
    # clone (single-branch of HEAD) fetches nothing and is not even shallow.
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    work = tmp_path / "work"
    _git(tmp_path, "init", "-q", "-b", "main", str(work))
    _git(work, "remote", "add", "origin", str(origin))
    lab = tmp_path / "lab"
    lab.mkdir()
    return {"origin": origin, "work": work, "lab": lab, "tmp": tmp_path}


def _pin_line(lab: Path, line: str, sha: str, *, lock: bool = False) -> None:
    d = lab / "research" / line
    d.mkdir(parents=True, exist_ok=True)
    if lock:
        (d / "uv.lock").write_text(
            '[[package]]\nname = "quantbox"\n'
            f'source = {{ git = "https://github.com/tomlupo/quantbox.git?rev=v0.6.0#{sha}" }}\n'
        )
    else:
        (d / "pyproject.toml").write_text(
            "[project]\ndependencies = [\n"
            f'    "quantbox[full] @ git+https://github.com/tomlupo/quantbox.git@{sha}",\n'
            '    "quantbox-datasets @ git+https://github.com/tomlupo/quantbox-datasets.git@'
            'ffffffffffffffffffffffffffffffffffffffff",\n'
            "]\n"
        )


def _run(world: dict[str, Path], *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(world["work"]), *extra, str(world["lab"])],
        capture_output=True,
        text=True,
    )


def _orphan_on_origin(world: dict[str, Path]) -> str:
    """A commit origin still serves, but no branch or tag on origin contains."""
    work = world["work"]
    _git(work, "checkout", "-q", "-b", "doomed")
    sha = _commit(work, "orphan")
    _git(work, "push", "-q", "origin", "doomed")
    _git(work, "checkout", "-q", "main")
    _git(work, "push", "-q", "origin", "--delete", "doomed")
    _git(work, "branch", "-q", "-D", "doomed")
    return sha


def test_extract_reads_pyproject_and_lock_pins_and_ignores_others(world: dict[str, Path]) -> None:
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        import check_pinned_shas as mod
    finally:
        sys.path.pop(0)
    lab = world["lab"]
    _pin_line(lab, "a", "a" * 40)
    _pin_line(lab, "b", "b" * 40, lock=True)
    (lab / "pyproject.toml").write_text('deps = ["quantbox @ git+https://github.com/tomlupo/quantbox.git@main"]\n')
    pins = mod.extract_pins([lab])
    assert set(pins) == {"a" * 40, "b" * 40}
    assert "f" * 40 not in pins  # quantbox-datasets pin is not a quantbox pin


def test_branch_reachable_pin_passes(world: dict[str, Path]) -> None:
    sha = _commit(world["work"], "c1")
    _git(world["work"], "push", "-q", "origin", "main")
    _pin_line(world["lab"], "line", sha)
    res = _run(world)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "1 pinned SHA" in res.stdout


def test_script_ignores_hook_git_dir(world: dict[str, Path]) -> None:
    sha = _commit(world["work"], "c1")
    _git(world["work"], "push", "-q", "origin", "main")
    _pin_line(world["lab"], "line", sha)
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(world["work"]), str(world["lab"])],
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_DIR": str(world["tmp"] / "not-a-repo")},
    )
    assert res.returncode == 0, res.stdout + res.stderr


def test_orphan_pin_fails_then_passes_once_tagged_on_origin(world: dict[str, Path]) -> None:
    _commit(world["work"], "base")
    _git(world["work"], "push", "-q", "origin", "main")
    sha = _orphan_on_origin(world)
    _pin_line(world["lab"], "line", sha, lock=True)

    res = _run(world)
    assert res.returncode == 1, res.stdout + res.stderr
    assert sha in res.stdout and "UNREACHABLE" in res.stdout

    # a LOCAL tag is not protection — origin must carry it
    _git(world["work"], "fetch", "-q", "origin", sha)
    _git(world["work"], "tag", "-a", f"pin/{sha[:7]}", sha, "-m", "pin")
    assert _run(world).returncode == 1

    _git(world["work"], "push", "-q", "origin", f"pin/{sha[:7]}")
    res = _run(world)
    assert res.returncode == 0, res.stdout + res.stderr


def test_fake_sha_goes_red(world: dict[str, Path]) -> None:
    _commit(world["work"], "c1")
    _git(world["work"], "push", "-q", "origin", "main")
    _pin_line(world["lab"], "line", FAKE_SHA)
    res = _run(world)
    assert res.returncode == 1, res.stdout + res.stderr
    assert FAKE_SHA in res.stdout


def test_extra_sha_flag_is_checked(world: dict[str, Path]) -> None:
    sha = _commit(world["work"], "c1")
    _git(world["work"], "push", "-q", "origin", "main")
    _pin_line(world["lab"], "line", sha)
    assert _run(world, "--sha", FAKE_SHA).returncode == 1


def test_no_pins_found_is_cannot_check_not_green(world: dict[str, Path]) -> None:
    _commit(world["work"], "c1")
    _git(world["work"], "push", "-q", "origin", "main")
    res = _run(world)
    assert res.returncode == 2, res.stdout + res.stderr
    assert "no pinned" in res.stderr.lower()


def test_missing_lab_dir_is_cannot_check(world: dict[str, Path]) -> None:
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(world["work"]), str(world["tmp"] / "nope")],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2


def test_shallow_clone_is_cannot_check_not_unreachable(world: dict[str, Path]) -> None:
    """In a shallow clone (actions/checkout's default --depth=1) the ancestry walk
    stops at the shallow boundary, so a reachable pin would read UNREACHABLE.
    That is "could not look", not a verdict: exit 2, never 1 (PR #217 review)."""
    pinned = _commit(world["work"], "pinned")
    _commit(world["work"], "later")
    _git(world["work"], "push", "-q", "origin", "main")
    _pin_line(world["lab"], "line", pinned)

    shallow = world["tmp"] / "shallow"
    _git(world["tmp"], "clone", "-q", "--depth=1", f"file://{world['origin']}", str(shallow))
    assert _git(shallow, "rev-parse", "--is-shallow-repository") == "true"
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(shallow), str(world["lab"])],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2, res.stdout + res.stderr
    assert "shallow" in res.stderr.lower()
    assert "UNREACHABLE" not in res.stdout

    # once unshallowed the same pin is green — the check itself was right all along
    _git(shallow, "fetch", "-q", "--unshallow", "origin")
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(shallow), str(world["lab"])],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stdout + res.stderr
