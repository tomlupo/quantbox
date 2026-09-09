"""The vendored pre-push guard must keep refusing release-shaped pushes to `main`.

`.claude/hooks/pre-push-branch-guard` is a FORK of the qute-essentials template
with the template's "release exemption" deleted. That exemption lets a push whose
commits merely LOOK like releases land on a guarded branch without a PR — which
in this repo would be the only unreviewed route onto `main`, the branch
quantbox-live pins.

Re-vendoring the hook is a `cp`, and if the file goes missing the `.git/hooks`
dispatcher falls back to the plugin's CACHED template — exemption included. So
nothing about the divergence is self-enforcing, and this file is the enforcer.

Two design rules, both learned from review:

*   The release-shaped commit is BUILT in a throwaway repo, never mined from
    this repo's history. CI checks out at `fetch-depth: 1`, where
    `git log --grep='^bump: version'` finds nothing and `<tip>~1` does not
    resolve — a history-mining test skips (or passes for the wrong reason)
    exactly where it is most needed.
*   Nothing here skips. A missing guard file is the very scenario the module
    docstring names as dangerous, so it must be a FAILURE, not a skip.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD = REPO_ROOT / ".claude" / "hooks" / "pre-push-branch-guard"

#: Names that exist ONLY in the release exemption. Any of them reappearing means
#: the fork was overwritten by the upstream template.
EXEMPTION_SYMBOLS = (
    "_is_release_commit",
    "_range_is_only_release_commits",
    "_VERSION_ARTIFACTS",
)


#: Variables that would make this suite measure the SHELL instead of the guard.
#: `CLAUDE_GUARD_BRANCH_PUSH=0` is honoured before the config is even read, so
#: an exported one turns every "allow" assertion green for the wrong reason —
#: precisely the failure the guard's own docstring warns about (set it inline,
#: never export it). `QUTE_PRE_PUSH_GUARD_TRACE` only adds stderr noise.
_MUTING_VARS = frozenset({"CLAUDE_GUARD_BRANCH_PUSH", "QUTE_PRE_PUSH_GUARD_TRACE"})


def _clean_env() -> dict[str, str]:
    """`os.environ` minus every `GIT_*` variable and every muting override.

    This suite runs from the repo's own pre-push hook, and git exports `GIT_DIR`
    and `GIT_INDEX_FILE` to its hooks. Inherited, they point every `git` call in
    a `tmp_path` repo back at THIS repository — the fixture commit fails, and a
    guard invocation would judge the wrong repo while looking like it judged the
    fixture. Caught by the pre-push hook itself.
    """
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k not in _MUTING_VARS}


def _run(cwd: Path, stdin_text: str) -> subprocess.CompletedProcess:
    """Feed the guard one pre-push stdin line, judged from `cwd`."""
    return subprocess.run(
        [sys.executable, str(GUARD), "origin", "git@github.com:tomlupo/quantbox.git"],
        input=stdin_text,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=_clean_env(),
        timeout=60,
    )


def _git(cwd: Path, *args: str) -> str:
    env = {
        **_clean_env(),
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.strip()


@pytest.fixture
def guarded_repo(tmp_path: Path) -> Path:
    """A repo that opts into the guard and carries one release-shaped commit.

    The commit is what the upstream exemption is built to wave through: subject
    `bump: version …`, touching `pyproject.toml` and `CHANGELOG.md` and nothing
    else. Built here rather than mined from history so the test is identical in
    a shallow CI checkout.
    """
    repo = tmp_path / "guarded"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    (repo / ".qute").mkdir()
    (repo / ".qute" / "config.json").write_text(
        '{"git": {"protected_branch": "main", "integration_branch": "dev"}}',
        encoding="utf-8",
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "chore: base")

    (repo / "pyproject.toml").write_text('version = "1.1.0"\n', encoding="utf-8")
    (repo / "CHANGELOG.md").write_text("## 1.1.0\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "bump: version 1.0.0 → 1.1.0")
    return repo


def test_the_vendored_guard_is_present():
    """A missing hook is the dispatcher's cue to fall back to the TEMPLATE.

    So absence is a defect in this repo, never an environment to skip over.
    """
    assert GUARD.exists(), (
        f"{GUARD} is missing. The .git/hooks dispatcher then resolves the guard "
        "from the plugin's cached template — release exemption included."
    )


def test_a_release_shaped_push_to_main_is_refused(guarded_repo: Path):
    """The whole point of the fork: looking like a release earns no exemption."""
    tip = _git(guarded_repo, "rev-parse", "HEAD")
    parent = _git(guarded_repo, "rev-parse", "HEAD~1")

    result = _run(guarded_repo, f"refs/heads/main {tip} refs/heads/main {parent}\n")

    assert result.returncode == 1, (
        "a release-shaped push to `main` was ALLOWED — the release exemption is "
        "back, probably via a re-vendor from the upstream template.\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "REFUSED" in result.stderr


def test_the_guard_is_discriminating_not_broken_shut(guarded_repo: Path):
    """Positive control on both sides, so 'refuses everything' cannot pass."""
    refused = _run(guarded_repo, "refs/heads/main aaaa refs/heads/main bbbb\n")
    allowed = _run(guarded_repo, "refs/heads/feat/x aaaa refs/heads/feat/x bbbb\n")

    assert refused.returncode == 1, refused.stderr
    assert allowed.returncode == 0, allowed.stderr


def test_deleting_main_is_refused(guarded_repo: Path):
    result = _run(
        guarded_repo,
        "(delete) 0000000000000000000000000000000000000000 refs/heads/main bbbb\n",
    )
    assert result.returncode == 1, result.stderr


def test_a_repo_with_no_config_at_all_is_not_guarded(tmp_path: Path):
    """No config anywhere: the hook is a no-op, as it is in third-party clones."""
    repo = tmp_path / "plain"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")

    result = _run(repo, "refs/heads/main aaaa refs/heads/main bbbb\n")

    assert result.returncode == 0, result.stderr


def test_a_config_with_other_sections_but_no_git_is_not_guarded(tmp_path: Path):
    """Opt-in is the SECTION, not the file.

    A repo can carry `.qute/config.json` for `ship` or `review` alone and mean
    nothing by it about branch guarding, so the `git`-section lookup must fall
    through rather than treat the file's existence as consent.
    """
    repo = tmp_path / "othersections"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    (repo / ".qute").mkdir()
    (repo / ".qute" / "config.json").write_text('{"ship": {}, "review": {}}', encoding="utf-8")

    result = _run(repo, "refs/heads/main aaaa refs/heads/main bbbb\n")

    assert result.returncode == 0, result.stdout + result.stderr


def test_git_fields_at_the_top_level_refuse_rather_than_disarm(tmp_path: Path):
    """The forgotten `"git"` wrapper must fail CLOSED.

    Written flat, the fields configure nothing: the section lookup misses, the
    repo reads as "not opted in", and every push to `main` would be allowed.
    That is the "looks guarded, isn't" state, so it refuses instead.
    """
    repo = tmp_path / "misplaced"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    (repo / ".qute").mkdir()
    (repo / ".qute" / "config.json").write_text(
        '{"protected_branch": "main", "integration_branch": "dev"}',
        encoding="utf-8",
    )

    result = _run(repo, "refs/heads/main aaaa refs/heads/main bbbb\n")

    assert result.returncode == 1, result.stdout + result.stderr
    assert "TOP" in result.stderr


def test_a_stray_release_tool_key_alone_does_not_refuse(tmp_path: Path):
    """`release_tool` is guidance text with no security meaning.

    Refusing every push over a misplaced string would be a guard that costs
    more than it protects, so only the BRANCH fields trigger the refusal.
    """
    repo = tmp_path / "stray"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    (repo / ".qute").mkdir()
    (repo / ".qute" / "config.json").write_text('{"release_tool": "commitizen (/ship)"}', encoding="utf-8")

    result = _run(repo, "refs/heads/main aaaa refs/heads/main bbbb\n")

    assert result.returncode == 0, result.stdout + result.stderr


def test_a_non_object_git_section_refuses(tmp_path: Path):
    """A config that cannot be understood stops the push; it never fails open."""
    repo = tmp_path / "broken"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    (repo / ".qute").mkdir()
    (repo / ".qute" / "config.json").write_text('{"git": "dev"}', encoding="utf-8")

    result = _run(repo, "refs/heads/main aaaa refs/heads/main bbbb\n")

    assert result.returncode == 1, result.stdout + result.stderr


def test_malformed_json_refuses(tmp_path: Path):
    repo = tmp_path / "malformed"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    (repo / ".qute").mkdir()
    (repo / ".qute" / "config.json").write_text("{not json", encoding="utf-8")

    result = _run(repo, "refs/heads/main aaaa refs/heads/main bbbb\n")

    assert result.returncode == 1, result.stdout + result.stderr


def test_the_legacy_flat_file_is_still_honoured(tmp_path: Path):
    """Checkouts that have not folded yet must stay guarded."""
    repo = tmp_path / "legacy"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    (repo / ".claude").mkdir()
    (repo / ".claude" / "git-guard.json").write_text('{"protected_branch": "main"}', encoding="utf-8")

    result = _run(repo, "refs/heads/main aaaa refs/heads/main bbbb\n")

    assert result.returncode == 1, result.stdout + result.stderr


def test_exemption_symbols_are_absent_from_the_vendored_guard():
    source = GUARD.read_text(encoding="utf-8")
    present = [name for name in EXEMPTION_SYMBOLS if name in source]
    assert not present, (
        f"the release exemption is back in {GUARD}: {present}. This fork drops it "
        "deliberately — see the file's header docstring and `main()`."
    )


def test_the_refusal_names_the_config_surface_that_answered(tmp_path: Path):
    """`source_path` is fork-only, so assert the BEHAVIOUR, not the string.

    Grepping the source would pass a re-vendor that kept the name but dropped
    the field from the returned dict; the refusal text cannot.
    """
    consolidated = tmp_path / "consolidated"
    consolidated.mkdir()
    _git(consolidated, "init", "-q", "-b", "main", ".")
    (consolidated / ".qute").mkdir()
    (consolidated / ".qute" / "config.json").write_text('{"git": {"protected_branch": "main"}}', encoding="utf-8")

    legacy = tmp_path / "legacyonly"
    legacy.mkdir()
    _git(legacy, "init", "-q", "-b", "main", ".")
    (legacy / ".claude").mkdir()
    (legacy / ".claude" / "git-guard.json").write_text('{"protected_branch": "main"}', encoding="utf-8")

    line = "refs/heads/main aaaa refs/heads/main bbbb\n"
    from_consolidated = _run(consolidated, line)
    from_legacy = _run(legacy, line)

    assert ".qute/config.json" in from_consolidated.stderr, from_consolidated.stderr
    assert ".claude/git-guard.json" in from_legacy.stderr, from_legacy.stderr


def test_flat_keys_refuse_even_when_a_valid_legacy_file_exists(tmp_path: Path):
    """Precedence: written-with-intent beats a working fallback.

    `.qute/config.json` carrying the branch fields at the top level is someone
    configuring THIS guard and getting the wrapper wrong. Falling back to a
    valid `.claude/git-guard.json` would enforce a policy the repo's newest
    statement does not describe, silently. Pinned here so a later "helpful"
    fallback cannot reopen it.
    """
    repo = tmp_path / "both"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    (repo / ".qute").mkdir()
    (repo / ".qute" / "config.json").write_text('{"protected_branch": "main"}', encoding="utf-8")
    (repo / ".claude").mkdir()
    (repo / ".claude" / "git-guard.json").write_text('{"protected_branch": "main"}', encoding="utf-8")

    result = _run(repo, "refs/heads/main aaaa refs/heads/main bbbb\n")

    assert result.returncode == 1, result.stdout + result.stderr
    assert "TOP" in result.stderr


def test_stray_top_level_keys_beside_a_git_section_also_refuse(tmp_path: Path):
    """A top-level copy next to a real section is read by NOTHING.

    The section wins, so the guard may protect a different branch than the one
    named up top and the push the author believes is refused exits 0 in
    silence. Same "looks guarded, isn't" state as the forgotten wrapper.
    """
    repo = tmp_path / "duplicated"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    (repo / ".qute").mkdir()
    (repo / ".qute" / "config.json").write_text(
        '{"git": {}, "protected_branch": "release", "integration_branch": "dev"}',
        encoding="utf-8",
    )

    result = _run(repo, "refs/heads/release aaaa refs/heads/release bbbb\n")

    assert result.returncode == 1, result.stdout + result.stderr
    assert "TOP level" in result.stderr
