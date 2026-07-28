# Git Workflow

## Branch strategy

| Branch | Purpose | Merge target |
|--------|---------|--------------|
| `main` | Release-only — **protected** | — (the tag + deploy target) |
| `dev` | Integration branch | `main`, at release time |
| `feat/{slug}` | One change | `dev` via PR |

## The flow

1. Branch a `feat/{slug}` off `dev`.
2. Commit there; open a PR to **`dev`** (`gh pr create --base dev`).
   Never PR a feature straight to `main`.
3. Conventional-Commit prefixes on merge commits drive the semver bump (commitizen (/ship)).
4. Release: `/ship` on `dev`, push `dev` **and the tag**, then PR `dev` → `main`
   and merge it with a **merge commit** (`gh pr merge --merge`), never `--squash`.
   Squashing rewrites the bump commit, so the commit `/ship` tagged is not an
   ancestor of `main` and the tag names code `main` does not contain — see
   `CLAUDE.md::Shipping cycle` for the full reasoning and the v0.4.0 incident.

**Never commit or push directly to `main`.** It is the protected branch;
all changes reach it through a PR. This is enforced agent-side by the
`git-workflow-guard.py` PreToolUse hook (see `.claude/git-guard.json`) — the
deterministic stand-in for GitHub branch protection, which is unavailable on
this repo's plan.

Trivial changes still go through a short-lived branch + PR to `dev`. `dev` IS
the integration branch — but land work on it via PR, not by committing to it
directly, so the review gate sees every change.

## Commits

Conventional Commits — `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `test:`.
Never `--no-verify`; never force-push to `main`; commit only when asked.

## Override

The guard hook blocks direct commit/push to `main`. For a deliberate
exception (e.g. on your own machine), `export GIT_GUARD_DISABLE=1` for that shell.
