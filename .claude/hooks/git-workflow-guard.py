#!/usr/bin/env python3
"""
Claude Code PreToolUse hook (matcher: Bash) — agent-side branch-workflow guard.

The deterministic equivalent of GitHub branch protection, for repos where
branch protection is unavailable (private repos on the Free plan). It BLOCKS
exactly one class of mistake: a direct `git commit` or `git push` to the
repo's PROTECTED branch when the workflow requires going through a PR (and
through the integration branch first, where one exists).

It does NOT lint/format (the git pre-commit hooks already do that) and it
does NOT touch normal feature-branch work. Conservative by design: it only
blocks the genuinely-wrong action, and it is trivially overridable.

Documented gaps (deliberate — these are explicit acts, not the accidental
footgun this guards, and catching them would risk false blocks):
  - `git checkout main && git commit ...` in one line — the hook reads the
    *current* branch, so a switch-then-act chain is not caught.
  - `git push --all` / `git push --mirror` from a non-protected branch —
    only caught when standing on the protected branch.

Config: `.claude/git-guard.json` in the repo root (see git-guard.json.example):
    {
      "protected_branch": "main",      # the branch you must never commit/push to directly
      "integration_branch": "dev",     # where feature PRs land first (null if feature -> PR -> main)
      "release_tool": "commitizen (/ship)"
    }
If the config file is absent, the hook is a no-op (allows everything) — so
dropping the hook into a repo without a config never blocks anything.

Override (e.g. Tom on his own machine, a deliberate hotfix):
    export GIT_GUARD_DISABLE=1
When set to a non-empty value the hook allows everything.

Contract (per Claude Code hook docs):
  - stdin: JSON with `tool_name` and `tool_input.command`
  - block: exit 0 with JSON
        {"hookSpecificOutput": {"hookEventName": "PreToolUse",
          "permissionDecision": "deny", "permissionDecisionReason": "...",
          "additionalContext": "..."}}
  - allow: exit 0 with no output (fall through to normal permission flow)
  - any internal error: exit 0, allow (fail-open — a guard bug must never
    wedge the agent's git access)
"""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


def _allow():
    """Emit nothing and exit — falls through to the normal permission flow."""
    sys.exit(0)


def _deny(reason: str, guidance: str):
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                    "additionalContext": guidance,
                }
            }
        )
    )
    sys.exit(0)


def _project_dir() -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd()))


def _load_config(project_dir: Path):
    cfg_path = project_dir / ".claude" / "git-guard.json"
    if not cfg_path.is_file():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return None
    if not cfg.get("protected_branch"):
        return None
    return cfg


def _current_branch(cwd: Path):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=5,
        )
        if out.returncode != 0:
            return None
        branch = out.stdout.strip()
        # Detached HEAD reports "HEAD" — not on a named branch, nothing to guard.
        return branch if branch and branch != "HEAD" else None
    except Exception:
        return None


# Split a shell line into the individual commands joined by &&, ||, ;, | (and
# newlines). Crude but sufficient: we only need to find git commit/push verbs.
_SEP = re.compile(r"&&|\|\||;|\n|\|")


def _segments(command: str):
    return [s.strip() for s in _SEP.split(command) if s.strip()]


def _tokens(segment: str):
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _git_subcommand(tokens):
    """Return (subcommand, args_after_it) if this segment is a git invocation."""
    if not tokens:
        return None, []
    # Skip leading inline env-var assignments (`FOO=1 git push ...`). Note an
    # inline `GIT_GUARD_DISABLE=1 git ...` does NOT override this hook — the hook
    # runs as a separate process and only sees exported env; use
    # `export GIT_GUARD_DISABLE=1` to override.
    while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
        tokens = tokens[1:]
    if not tokens or tokens[0] != "git":
        return None, []
    i = 0
    i = 1
    # Skip global git options like `-C path`, `-c key=val`, `--git-dir=...`.
    while i < len(tokens) and tokens[i].startswith("-"):
        if tokens[i] in ("-C", "-c", "--git-dir", "--work-tree"):
            i += 2
        else:
            i += 1
    if i >= len(tokens):
        return None, []
    return tokens[i], tokens[i + 1 :]


def _strip_ref(name: str) -> str:
    # Drop the force-push sigil (`+main`) and the fully-qualified prefix so
    # `+refs/heads/main` and `main` compare equal to the protected name.
    return name.lstrip("+").replace("refs/heads/", "")


def _push_targets(args):
    """
    Destination branch names for a `git push` invocation.
    Returns (targets, had_explicit_refspec).
    """
    positionals = [a for a in args if not a.startswith("-")]
    # First positional is the remote; the rest are refspecs.
    refspecs = positionals[1:] if len(positionals) >= 1 else []
    targets = []
    for ref in refspecs:
        # src:dst -> dst is the destination branch; bare ref -> same name.
        dst = ref.split(":", 1)[1] if ":" in ref else ref
        targets.append(_strip_ref(dst))
    return targets, len(refspecs) > 0


def main():
    if os.environ.get("GIT_GUARD_DISABLE"):
        _allow()

    try:
        hook_input = json.load(sys.stdin)
    except Exception:
        _allow()

    if hook_input.get("tool_name") != "Bash":
        _allow()

    command = (hook_input.get("tool_input") or {}).get("command", "")
    if not command or "git" not in command:
        _allow()

    project_dir = _project_dir()
    cfg = _load_config(project_dir)
    if cfg is None:
        _allow()

    protected = cfg["protected_branch"]
    integration = cfg.get("integration_branch")
    release_tool = cfg.get("release_tool")

    branch = _current_branch(project_dir)

    # The "right move" guidance, shared by both block paths.
    if integration:
        guidance = (
            f"'{protected}' is the protected branch in this repo. Land work via "
            f"a feature branch -> PR to '{integration}', then ship '{integration}' "
            f"-> '{protected}' through the release flow"
            + (f" ({release_tool})" if release_tool else "")
            + f". Create a feature branch (git switch -c feat/<slug>) off "
            f"'{integration}' and open a PR, instead of committing/pushing "
            f"'{protected}' directly. Deliberate exception (e.g. on your own "
            f"machine): export GIT_GUARD_DISABLE=1."
        )
    else:
        guidance = (
            f"'{protected}' is the protected branch in this repo. Land work via "
            f"a feature branch -> PR to '{protected}'. Create a feature branch "
            f"(git switch -c feat/<slug>) and open a PR instead of "
            f"committing/pushing '{protected}' directly. Deliberate exception "
            f"(e.g. on your own machine): export GIT_GUARD_DISABLE=1."
        )

    for seg in _segments(command):
        sub, args = _git_subcommand(_tokens(seg))
        if sub is None:
            continue

        if sub == "commit":
            if branch == protected:
                _deny(
                    f"Blocked: direct commit on protected branch '{protected}'.",
                    guidance,
                )

        elif sub == "push":
            targets, had_refspec = _push_targets(args)
            if protected in targets:
                _deny(
                    f"Blocked: push to protected branch '{protected}'.",
                    guidance,
                )
            # No explicit refspec -> pushes the current branch. If that's the
            # protected branch, it's a direct push to protected.
            if not had_refspec and branch == protected:
                _deny(
                    f"Blocked: push of current branch '{protected}' (the protected branch).",
                    guidance,
                )

    _allow()


if __name__ == "__main__":
    # Belt-and-suspenders fail-open: an unforeseen error must never wedge git.
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        sys.exit(0)
