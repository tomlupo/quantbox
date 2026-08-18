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
# changes nothing.
set -euo pipefail

cd "$(dirname "$0")/.."

TAG="$(git describe --tags --abbrev=0 2>/dev/null || true)"
if [ -z "$TAG" ]; then
    echo "after-release: no tags in this repo — nothing to verify." >&2
    exit 1
fi

# quantbox-live sits beside this repo in the workspace; override if it does not.
LIVE_REPO="${QUANTBOX_LIVE_REPO:-$(cd .. && pwd)/quantbox-live}"

echo "quantbox ${TAG} is tagged."
echo

# 1. The tag must be an ancestor of the release branch, or anything pinning it
#    resolves code the release branch never contained (the v0.4.0 lesson).
if git merge-base --is-ancestor "$TAG" origin/main 2>/dev/null; then
    echo "  [ok]   ${TAG} is an ancestor of origin/main — safe to pin."
else
    echo "  [WAIT] ${TAG} is NOT an ancestor of origin/main."
    echo "         The promotion PR has not merged yet, or it was SQUASHED."
    echo "         Do not pin until this reads [ok]."
fi

# 2. Report the pin quantbox-live currently declares.
if [ -f "${LIVE_REPO}/pyproject.toml" ]; then
    CURRENT_PIN="$(grep -o 'quantbox\.git@v[0-9][^"]*' "${LIVE_REPO}/pyproject.toml" | head -1 | sed 's/.*@//')"
    if [ "$CURRENT_PIN" = "$TAG" ]; then
        echo "  [ok]   quantbox-live already pins ${TAG}."
    else
        echo "  [TODO] quantbox-live pins ${CURRENT_PIN:-<unreadable>}, not ${TAG}."
    fi
else
    echo "  [skip] quantbox-live not found at ${LIVE_REPO}"
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
