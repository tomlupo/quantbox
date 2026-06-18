# Git Workflow

## Branch strategy

| Branch | Purpose | Merge target |
|--------|---------|--------------|
| `main` | Production-ready code — **protected** | — (deploy target) |
| `feat/{slug}` | One change | `main` via PR |

## The flow

1. Branch a `feat/{slug}` off `main`.
2. Commit there; open a PR to `main`.
3. Conventional-Commit prefixes on merge commits drive the semver bump (commitizen (/ship)).

**Never commit or push directly to `main`.** It is the protected branch;
all changes reach it through a PR. This is enforced agent-side by the
`git-workflow-guard.py` PreToolUse hook (see `.claude/git-guard.json`) — the
deterministic stand-in for GitHub branch protection, which is unavailable on
this repo's plan.

Trivial changes still go through a short-lived branch + PR; there is no
integration branch to commit to directly in this repo.

## Commits

Conventional Commits — `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `test:`.
Never `--no-verify`; never force-push to `main`; commit only when asked.

## Override

The guard hook blocks direct commit/push to `main`. For a deliberate
exception (e.g. on your own machine), `export GIT_GUARD_DISABLE=1` for that shell.
