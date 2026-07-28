# Multi-repo workflow

QuantBox is split across three repositories:

| Repo | Purpose | Visibility |
|---|---|---|
| **quantbox** | Library — strategies, plugins, protocols, core runtime | Public |
| **quantbox-live** | Trading deployment — configs, daily automation, reports | Private |
| **quantbox-lab** | Research workspace — backtests, notebooks, experiments | Private |

Both `quantbox-live` and `quantbox-lab` consume `quantbox` as a git dependency via `pyproject.toml`.

## Dependency pinning

### quantbox-live: pinned to tags on `main`

```toml
"quantbox[ccxt] @ git+https://github.com/tomlupo/quantbox.git@v0.1.0"
```

Tags are:
- **Readable** — `v0.1.0` vs `3f313fa`
- **Immutable** — can't be accidentally changed
- **Deliberate** — you choose exactly when to upgrade production

### quantbox-lab: tracks `dev` branch

```toml
"quantbox[full] @ git+https://github.com/tomlupo/quantbox.git@dev"
```

Lab always gets the latest development code. Refresh with:

```bash
uv sync --upgrade-package quantbox
```

## Branching model

```
dev (development)
  │
  │  develop, test, iterate
  │
  ▼
main (production-ready)  ← merge dev when ready
  │
  ├── v0.1.0  ← quantbox-live pins here
  ├── v0.1.1  ← next patch
  └── v0.2.0  ← next minor
```

- **`dev`** — all active development happens here
- **`main`** — only receives merges from `dev` when code is tested and ready
- **Tags** — cut by `/ship` on `dev` (annotated, semver), and reachable from
  `main` because the release PR is merged with a MERGE COMMIT, not squashed

## Promoting code to production

### 1. Tag a release in quantbox

```bash
cd ~/workspace/projects/quantbox
# /ship bumps version + CHANGELOG and cuts the ANNOTATED tag. Do not hand-roll
# `cz bump`: a raw bump makes a LIGHTWEIGHT tag, and `git push --follow-tags`
# silently declines to push those — the tag stays local until a consumer's
# `uv lock` cannot resolve it (2026-07-28).
/ship                                    # on dev

git push origin dev
git push origin v0.x.y                   # explicitly — do not rely on --tags
git ls-remote --tags origin v0.x.y       # VERIFY it landed

# Merge with a MERGE COMMIT, never --squash: squashing rewrites the bump so the
# tagged commit is not an ancestor of main, and the tag then names code main does
# not contain — while quantbox-live pins that tag.
gh pr create --base main --head dev --title "release: v0.x.y"
gh pr merge --merge
git checkout dev
```

### 2. Bump quantbox-live

Edit `pyproject.toml` in quantbox-live to reference the new tag:

```bash
cd ~/workspace/projects/quantbox-live
# Change @v0.1.0 to @v0.x.y in pyproject.toml
uv sync
```

Test, then commit and push:

```bash
uv run quantbox run --dry-run -c cookbook/configs/carver_hyperliquid.yaml
git add pyproject.toml uv.lock
git commit -m "Bump quantbox to v0.x.y"
git push
```

### 3. Lab auto-refreshes

No manual steps needed for quantbox-lab. Just run:

```bash
cd ~/workspace/projects/quantbox-lab
uv sync --upgrade-package quantbox
```

## Version numbering

Follow [semver](https://semver.org/):

| Change type | Version bump | Example |
|---|---|---|
| Bug fix, docs, internal refactor | Patch | `v0.1.0` → `v0.1.1` |
| New strategy, plugin, or feature | Minor | `v0.1.1` → `v0.2.0` |
| Breaking protocol/contract change | Major | `v0.2.0` → `v1.0.0` |

## Optional dependency extras

| Extra | Includes | Used by |
|---|---|---|
| `ccxt` | ccxt (Binance, Hyperliquid) | quantbox-live |
| `ibkr` | ib_insync | IBKR deployments |
| `binance` | python-binance | Legacy Binance adapter |
| `full` | All of the above | quantbox-lab |
