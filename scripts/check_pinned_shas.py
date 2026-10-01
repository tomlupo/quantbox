"""Fail when a quantbox SHA that a lab line pins is not reachable on origin.

A research line in quantbox-lab / robo-lab / dm-evo-lab reproduces only while the
quantbox commit it pins still exists. GitHub keeps serving a commit for a while
after the last branch containing it is gone, but nothing obliges it to: an
unreachable commit can be garbage-collected, and the line stops reproducing.
TOM-1340 found three lines pinned to such an orphan (83651b8) and protected it
with an annotated ``pin/<sha7>`` tag. This script is the check that keeps it so.

What is a pin: in every ``pyproject.toml`` under the given lab trees, the SHA in
``.../tomlupo/quantbox.git@<40-hex>``; in every ``uv.lock``, the resolved SHA in
``.../tomlupo/quantbox.git?...#<40-hex>`` (so a pin by tag or branch is checked
at the commit it actually resolved to). ``quantbox-datasets`` and other repos
are not quantbox pins.

What is reachable: an ancestor of a branch or tag that ORIGIN advertises
(``git ls-remote``). A tag that exists only in your local clone protects nothing
and does not count. A pinned SHA origin cannot serve at all is unreachable too.

Run it against checkouts at the labs' origin tips; a stale local checkout checks
stale pins.

USAGE:
    uv run python scripts/check_pinned_shas.py [--repo PATH] [--remote NAME]
                                               [--sha SHA ...] LAB_DIR [LAB_DIR ...]

EXIT:
    0  every pin found is reachable (the count is printed — read it)
    1  at least one pin is UNREACHABLE (each one is named, with who pins it)
    2  could not check: a lab dir is missing, no pins were found, the quantbox
       clone is shallow (ancestry is cut at the boundary), or git failed
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PYPROJECT_PIN = re.compile(r"github\.com/tomlupo/quantbox\.git@([0-9a-f]{40})\b")
LOCK_PIN = re.compile(r"github\.com/tomlupo/quantbox\.git[^#\"\s]*#([0-9a-f]{40})\b")
FULL_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
SKIP_DIRS = {".git", ".venv", "node_modules"}
# Run from a git hook, these aim every `git -C <repo>` at the hook's repo instead.
REPO_LOCATING_ENV = {
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_PREFIX",
}
GIT_ENV = {k: v for k, v in os.environ.items() if k not in REPO_LOCATING_ENV}


class CannotCheck(Exception):
    pass


def _files(root: Path, name: str):
    for p in root.rglob(name):
        if not SKIP_DIRS.intersection(p.relative_to(root).parts):
            yield p


def extract_pins(lab_dirs: list[Path]) -> dict[str, list[str]]:
    """Map each pinned quantbox SHA to the files that pin it."""
    pins: dict[str, list[str]] = defaultdict(list)
    for lab in lab_dirs:
        for name, pattern in (("pyproject.toml", PYPROJECT_PIN), ("uv.lock", LOCK_PIN)):
            for f in _files(lab, name):
                for sha in pattern.findall(f.read_text(errors="replace")):
                    pins[sha].append(str(f))
    return dict(pins)


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    res = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=GIT_ENV)
    if check and res.returncode != 0:
        raise CannotCheck(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res


def refuse_shallow(repo: Path) -> None:
    """A shallow clone cuts history at its boundary, so ``merge-base --is-ancestor``
    says "no" for a pin below it even when origin's branch contains it. That is a
    blind check, not a verdict: refuse rather than report UNREACHABLE."""
    out = _git(repo, "rev-parse", "--is-shallow-repository").stdout.strip()
    if out == "true":
        raise CannotCheck(
            f"{repo} is a shallow clone; ancestry stops at the shallow boundary. "
            "Run `git fetch --unshallow` (or check out with fetch-depth: 0) and retry"
        )
    if out != "false":
        raise CannotCheck(f"cannot tell whether {repo} is shallow: {out!r}")


def origin_ref_tips(repo: Path, remote: str) -> dict[str, str]:
    """Commit each branch and tag on the remote points at (tags peeled)."""
    out = _git(repo, "ls-remote", "--heads", "--tags", remote).stdout
    tips: dict[str, str] = {}
    for line in out.splitlines():
        sha, ref = line.split("\t")
        if ref.endswith("^{}"):
            tips[ref[:-3]] = sha  # peeled annotated tag wins over the tag object
        else:
            tips.setdefault(ref, sha)
    if not tips:
        raise CannotCheck(f"remote {remote!r} advertises no branches or tags")
    return tips


def _have(repo: Path, sha: str) -> bool:
    return _git(repo, "cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode == 0


def reachable_from(repo: Path, remote: str, sha: str, tips: dict[str, str]) -> list[str] | None:
    """Origin refs containing ``sha``; [] if none; None if origin cannot serve it."""
    if not _have(repo, sha):
        _git(repo, "fetch", "-q", remote, sha, check=False)
        if not _have(repo, sha):
            return None
    missing = sorted({t for t in tips.values() if not _have(repo, t)})
    if missing:
        _git(repo, "fetch", "-q", "--tags", remote)
        _git(repo, "fetch", "-q", remote, *[t for t in missing if not _have(repo, t)], check=False)
    hits = []
    for ref, tip in tips.items():
        if not _have(repo, tip):
            raise CannotCheck(f"cannot read {ref} ({tip[:7]}) from {remote}")
        if _git(repo, "merge-base", "--is-ancestor", sha, tip, check=False).returncode == 0:
            hits.append(ref)
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("labs", nargs="+", type=Path, help="lab checkout(s) to scan for pins")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1], help="quantbox clone")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--sha", action="append", default=[], help="also check this full SHA")
    args = ap.parse_args(argv)

    try:
        for lab in args.labs:
            if not lab.is_dir():
                raise CannotCheck(f"lab dir not found: {lab}")
        for s in args.sha:
            if not FULL_SHA.match(s):
                raise CannotCheck(f"--sha wants a full 40-hex SHA, got {s!r}")
        pins = extract_pins(args.labs)
        for s in args.sha:
            pins.setdefault(s, []).append("--sha")
        if not pins:
            raise CannotCheck("no pinned quantbox SHA found under " + ", ".join(map(str, args.labs)))
        refuse_shallow(args.repo)
        tips = origin_ref_tips(args.repo, args.remote)
        bad = 0
        for sha in sorted(pins):
            refs = reachable_from(args.repo, args.remote, sha, tips)
            where = ", ".join(sorted(set(pins[sha])))
            if refs:
                # long-lived refs first: a feature branch is deleted after merge
                first = ("refs/heads/main", "refs/heads/dev")
                refs = sorted(refs, key=lambda r: (r not in first, not r.startswith("refs/tags/"), r))
                shown = ", ".join(refs[:3]) + (f" (+{len(refs) - 3})" if len(refs) > 3 else "")
                print(f"  ok           {sha[:7]}  via {shown}")
            else:
                bad += 1
                why = f"not served by {args.remote}" if refs is None else "no branch or tag on " + args.remote
                print(f"  UNREACHABLE  {sha}  ({why})  pinned by: {where}")
    except CannotCheck as e:
        print(f"check_pinned_shas: CANNOT CHECK — {e}", file=sys.stderr)
        return 2

    print(f"{len(pins)} pinned SHA(s) checked against {args.remote}: {len(pins) - bad} reachable, {bad} unreachable")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
