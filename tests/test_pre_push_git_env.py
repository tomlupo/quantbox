"""TOM-1497: the pre-push pytest hook hands the suite no `GIT_*` variable.

Git exports `GIT_DIR` (and friends) to its hooks. Pushed from a linked
worktree, `GIT_DIR` names `.git/worktrees/<name>`, and a test's own `git init`
in a tmp dir then rewrote THIS repository's config: `core.bare=true` on
2026-10-01. The hook entry unsets every `GIT_*` before pytest starts (the
shape of qute-plugins TOM-1152). This test runs that entry with a fake `uv`
that records its environment, so deleting the unset loop turns it red.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_ID = "pytest-not-slow"
# What git hands a hook. The paths do not exist, so a leak cannot damage anything.
LEAK = {
    "GIT_DIR": "/nonexistent/tom-1497/sentinel.git",
    "GIT_INDEX_FILE": "/nonexistent/tom-1497/index",
    "GIT_WORK_TREE": "/nonexistent/tom-1497",
}


def _entry() -> str:
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hooks = [h for repo in config["repos"] for h in repo.get("hooks", []) if h.get("id") == HOOK_ID]
    assert len(hooks) == 1, f"expected one `{HOOK_ID}` hook, found {len(hooks)}: re-point this test"
    return hooks[0]["entry"]


def test_pre_push_pytest_hook_hands_pytest_no_git_env(tmp_path: Path) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    dump = tmp_path / "uv-env.txt"
    fake_uv = bindir / "uv"
    fake_uv.write_text(f"#!/bin/sh\nenv > {shlex.quote(str(dump))}\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "HOME": str(tmp_path), **LEAK}

    subprocess.run(shlex.split(_entry()), cwd=tmp_path, env=env, check=True)

    assert dump.exists(), "the entry never reached `uv`: this test checked nothing"
    leaked = [line for line in dump.read_text(encoding="utf-8").splitlines() if line.startswith("GIT_")]
    assert leaked == [], f"the pre-push hook passed git's environment to pytest: {leaked}"
