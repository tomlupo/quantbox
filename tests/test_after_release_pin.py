"""TOM-1640: `scripts/after-release.sh` reads the quantbox-live pin from origin/main.

The script once read `pyproject.toml` from the quantbox-live working tree. On
forge that checkout sat on a feature branch and the script said "v0.7.0" while
`origin/main` (the branch prod's cron merges) pinned v0.10.0. These tests build
a quantbox repo and a quantbox-live repo, each with a bare `origin`, and run a
copy of the real script against them. The local live checkout pins a DIFFERENT
version than its origin/main, and its origin/main ref is stale until a fetch, so
a working-tree read or a missing fetch turns them red.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "after-release.sh"
TAG = "v0.11.0"


def _pyproject(pin: str) -> str:
    return (
        "[project]\n"
        'name = "quantbox-live"\n'
        "dependencies = [\n"
        f'    "quantbox[ccxt] @ git+https://github.com/tomlupo/quantbox.git@{pin}",\n'
        "]\n"
    )


def _env(home: Path) -> dict[str, str]:
    # No inherited GIT_* (TOM-1497): a hook's GIT_DIR would point these calls
    # at the real repository.
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }


def _git(env: dict[str, str], *args: str | Path) -> str:
    out = subprocess.run(["git", *map(str, args)], env=env, capture_output=True, text=True, check=False)
    assert out.returncode == 0, f"git {' '.join(map(str, args))}: {out.stdout}{out.stderr}"
    return out.stdout


def _commit(env: dict[str, str], repo: Path, name: str, text: str, message: str) -> None:
    (repo / name).write_text(text, encoding="utf-8")
    _git(env, "-C", repo, "add", "-A")
    _git(env, "-C", repo, "commit", "-q", "-m", message)


def _quantbox(tmp: Path, env: dict[str, str]) -> Path:
    """A quantbox clone whose TAG is on origin/main, with the real script copied in."""
    bare = tmp / "quantbox.git"
    _git(env, "init", "-q", "--bare", "-b", "main", bare)
    repo = tmp / "ws" / "quantbox"
    _git(env, "clone", "-q", bare, repo)
    _git(env, "-C", repo, "checkout", "-q", "-b", "main")
    (repo / "scripts").mkdir()
    shutil.copy2(SCRIPT, repo / "scripts" / "after-release.sh")
    _commit(env, repo, "README", "quantbox\n", "release")
    _git(env, "-C", repo, "tag", "-a", TAG, "-m", TAG)
    _git(env, "-C", repo, "push", "-q", "origin", "main", TAG)
    return repo


def _live(tmp: Path, env: dict[str, str], *, local_pin: str, origin_pin: str) -> Path:
    """A quantbox-live clone on a feature branch pinning `local_pin`.

    Its origin/main ref is STALE: the commit that pins `origin_pin` is pushed
    to the bare repo from a second clone, after this clone last fetched.
    """
    bare = tmp / "quantbox-live.git"
    _git(env, "init", "-q", "--bare", "-b", "main", bare)
    live = tmp / "ws" / "quantbox-live"
    _git(env, "clone", "-q", bare, live)
    _git(env, "-C", live, "checkout", "-q", "-b", "main")
    _commit(env, live, "pyproject.toml", _pyproject("v0.1.0"), "pin v0.1.0")
    _git(env, "-C", live, "push", "-q", "origin", "main")
    _git(env, "-C", live, "checkout", "-q", "-b", "fix/feature-branch")
    _commit(env, live, "pyproject.toml", _pyproject(local_pin), f"pin {local_pin}")

    other = tmp / "other-live"
    _git(env, "clone", "-q", bare, other)
    _commit(env, other, "pyproject.toml", _pyproject(origin_pin), f"pin {origin_pin}")
    _git(env, "-C", other, "push", "-q", "origin", "main")
    return live


def _run(repo: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(repo / "scripts" / "after-release.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_pin_is_read_from_fetched_origin_main_not_the_working_tree(tmp_path: Path) -> None:
    env = _env(tmp_path)
    repo = _quantbox(tmp_path, env)
    _live(tmp_path, env, local_pin="v0.7.0", origin_pin=TAG)

    result = _run(repo, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"[ok]   {TAG} is an ancestor of origin/main" in result.stdout, result.stdout
    assert f"quantbox-live origin/main already pins {TAG}" in result.stdout, result.stdout
    assert "v0.7.0" not in result.stdout, "read the working tree: " + result.stdout
    assert "v0.1.0" not in result.stdout, "read a stale origin/main without a fetch: " + result.stdout


def test_a_pin_behind_the_tag_on_origin_main_is_a_todo(tmp_path: Path) -> None:
    env = _env(tmp_path)
    repo = _quantbox(tmp_path, env)
    # The working tree already pins the tag; origin/main does not. A
    # working-tree read would say [ok] here.
    _live(tmp_path, env, local_pin=TAG, origin_pin="v0.10.0")

    result = _run(repo, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"[TODO] quantbox-live origin/main pins v0.10.0, not {TAG}" in result.stdout, result.stdout
    assert "[ok]   quantbox-live" not in result.stdout, result.stdout


def test_a_linked_worktree_finds_live_beside_the_main_checkout(tmp_path: Path) -> None:
    env = _env(tmp_path)
    repo = _quantbox(tmp_path, env)
    _live(tmp_path, env, local_pin="v0.7.0", origin_pin=TAG)
    worktree = repo / ".claude" / "worktrees" / "wt"
    _git(env, "-C", repo, "worktree", "add", "-q", "-b", "wt", worktree, "main")

    result = _run(worktree, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[skip]" not in result.stdout, "looked for quantbox-live beside the worktree: " + result.stdout
    assert f"quantbox-live origin/main already pins {TAG}" in result.stdout, result.stdout


def test_an_unreachable_live_origin_fails_loudly(tmp_path: Path) -> None:
    env = _env(tmp_path)
    repo = _quantbox(tmp_path, env)
    live = _live(tmp_path, env, local_pin="v0.7.0", origin_pin=TAG)
    _git(env, "-C", live, "remote", "set-url", "origin", tmp_path / "gone.git")

    result = _run(repo, env)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "[FAIL] cannot fetch quantbox-live origin/main" in result.stdout, result.stdout
    assert "already pins" not in result.stdout, result.stdout
