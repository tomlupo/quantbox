#!/usr/bin/env bash
# after-release — what "released" actually means for quantbox.
#
# In Python mode `/ship` finishes at "the tag is pushed", and the failure that
# hides is a tag that exists while prod never pulled. For this repo prod does
# not pull on its own account: `quantbox-live` PINS a tag, and its cron picks
# the pin up. So there is no deploy command to run here — there is a pin to
# move in another repo, and a verification to do before moving it.
#
# This script does the verification and prints the exact remaining steps. It
# changes nothing beyond `git fetch origin main` in both repos: every answer is
# read from origin/main, never from a local branch or working tree. It exits 1
# when a check could not run.
set -euo pipefail

cd "$(dirname "$0")/.."

# --match: this repo also carries prod-{slug}-* promotion tags
# (docs/playbooks/promote-a-methodology.md), which are NOT releases.
TAG="$(git describe --tags --abbrev=0 --match 'v[0-9]*' 2>/dev/null || true)"
if [ -z "$TAG" ]; then
    echo "after-release: no tags in this repo — nothing to verify." >&2
    exit 1
fi

# quantbox-live sits beside this repo in the workspace; override if it does not.
# "Beside" means beside the MAIN checkout: from a linked worktree, `..` is
# .claude/worktrees/, so resolve the parent through git's common dir.
MAIN_CHECKOUT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
LIVE_REPO="${QUANTBOX_LIVE_REPO:-$(dirname "$MAIN_CHECKOUT")/quantbox-live}"

# Exit 1 when a check could not run: "could not check" is not "clean".
STATUS=0

echo "quantbox ${TAG} is tagged."
echo

# 1. The tag must be an ancestor of the release branch, or anything pinning it
#    resolves code the release branch never contained (the v0.4.0 lesson).
#    Fetch first: a stale origin/main reads [WAIT] for a promotion that merged.
if ! git fetch --quiet origin main 2>/dev/null; then
    echo "  [FAIL] cannot fetch quantbox origin/main — the ancestry check reads a stale ref."
    STATUS=1
fi
if git merge-base --is-ancestor "$TAG" origin/main 2>/dev/null; then
    echo "  [ok]   ${TAG} is an ancestor of origin/main — safe to pin."
else
    echo "  [WAIT] ${TAG} is NOT an ancestor of origin/main."
    echo "         The promotion PR has not merged yet, or it was SQUASHED."
    echo "         Do not pin until this reads [ok]."
fi

# 2. Report the pin quantbox-live currently declares — on ITS origin/main, the
#    branch prod's cron merges. The local checkout's working tree may sit on any
#    branch: on forge it read v0.7.0 from a feature branch while origin/main
#    pinned v0.10.0 (TOM-1640).
if [ -d "${LIVE_REPO}/.git" ] || [ -f "${LIVE_REPO}/.git" ]; then
    if ! git -C "$LIVE_REPO" fetch --quiet origin main 2>/dev/null; then
        echo "  [FAIL] cannot fetch quantbox-live origin/main — pin not read."
        STATUS=1
    elif ! LIVE_PYPROJECT="$(git -C "$LIVE_REPO" show origin/main:pyproject.toml 2>/dev/null)"; then
        echo "  [FAIL] quantbox-live origin/main has no pyproject.toml — pin not read."
        STATUS=1
    else
        CURRENT_PIN="$(printf '%s\n' "$LIVE_PYPROJECT" | sed -n 's/.*quantbox\.git@\(v[0-9][^"]*\).*/\1/p' | head -1)"
        if [ -z "$CURRENT_PIN" ]; then
            echo "  [FAIL] no quantbox.git@v… pin in quantbox-live origin/main:pyproject.toml."
            STATUS=1
        elif [ "$CURRENT_PIN" = "$TAG" ]; then
            echo "  [ok]   quantbox-live origin/main already pins ${TAG}."
        else
            echo "  [TODO] quantbox-live origin/main pins ${CURRENT_PIN}, not ${TAG}."
        fi
    fi
else
    echo "  [skip] no quantbox-live git checkout at ${LIVE_REPO}"
    echo "         (set QUANTBOX_LIVE_REPO to point at it)"
fi

cat <<EOF

Remaining steps — there is NO deploy command; the pickup is pull-based:

  1. Verify the tag carries what you expect:
         git show ${TAG}:<file>
  2. In quantbox-live, bump the pin to @${TAG}, then \`uv lock && uv sync\`.
  3. Open a PR to quantbox-live \`main\` and merge it. \`main\` is protected and
     the repo is liveCapital, so this is human-gated by design.
  4. Nothing else. prod's cron runs scripts/run_daily.sh (06:00 UTC) and
     scripts/run_kraken.sh (06:15 UTC); run_daily.sh does
     \`git merge origin/main\` + \`uv sync\`, so it picks the new pin up on its
     next run. Confirm from the next daily report rather than assuming.
EOF

exit "$STATUS"
