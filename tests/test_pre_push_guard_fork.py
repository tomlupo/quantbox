"""The vendored pre-push guard must keep refusing release-shaped pushes to `main`.

`.claude/hooks/pre-push-branch-guard` is a FORK of the qute-essentials template
with the template's "release exemption" deleted. That exemption lets a push whose
commits merely LOOK like releases land on a guarded branch without a PR — which
in this repo would be the only unreviewed route onto `main`, the branch
quantbox-live pins.

Re-vendoring the hook is a `cp` (and the dispatcher can also fall back to the
plugin's cached template), so nothing about the divergence is self-enforcing: a
refresh silently restores the exemption and every later release-titled push to
`main` starts passing. Until this file existed, the only thing standing in the
way was a comment asking people not to.

So this asserts the BEHAVIOUR (a release-shaped push to `main` is refused), not
just the absence of the symbols — the absence check is a second, cheaper net.
"""

from __future__ import annotations

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


requires_guard = pytest.mark.skipif(
    not GUARD.exists(),
    reason="vendored pre-push guard is absent from this checkout",
)


def _run(stdin_text: str) -> subprocess.CompletedProcess:
    """Feed the guard one pre-push stdin line, from the repo root."""
    return subprocess.run(
        [sys.executable, str(GUARD), "origin", "git@github.com:tomlupo/quantbox.git"],
        input=stdin_text,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )


@requires_guard
def test_release_shaped_push_to_main_is_still_refused():
    """The whole point of the fork: a release-looking push earns no exemption.

    The tip named here is the repo's real `bump: version …` commit, i.e. exactly
    the shape the upstream exemption waves through. It must be refused.
    """
    tip = subprocess.run(
        ["git", "log", "--format=%H", "-1", "--grep=^bump: version", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    ).stdout.strip()
    if not tip:
        pytest.skip("no `bump: version` commit reachable from HEAD in this checkout")

    result = _run(f"refs/heads/main {tip} refs/heads/main {tip}~1\n")

    assert result.returncode == 1, (
        "a release-shaped push to `main` was ALLOWED — the release exemption is "
        "back, probably via a re-vendor from the upstream template.\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "REFUSED" in result.stderr


@requires_guard
def test_ordinary_push_to_main_is_refused_and_a_feature_branch_is_not():
    """Positive control: the guard is live and discriminating, not refusing all.

    Without the second half, the first assert would also pass on a guard that is
    broken shut, which proves nothing about the exemption.
    """
    refused = _run("refs/heads/main aaaa refs/heads/main bbbb\n")
    allowed = _run("refs/heads/feat/x aaaa refs/heads/feat/x bbbb\n")

    assert refused.returncode == 1, refused.stderr
    assert allowed.returncode == 0, allowed.stderr


@requires_guard
def test_deleting_main_is_refused():
    result = _run("(delete) 0000000000000000000000000000000000000000 refs/heads/main bbbb\n")
    assert result.returncode == 1, result.stderr


@requires_guard
def test_exemption_symbols_are_absent_from_the_vendored_guard():
    source = GUARD.read_text(encoding="utf-8")
    present = [name for name in EXEMPTION_SYMBOLS if name in source]
    assert not present, (
        f"the release exemption is back in {GUARD}: {present}. This fork drops it "
        "deliberately — see the file's header docstring and `main()`."
    )
