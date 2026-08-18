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
- **Tags** — annotated, semver, cut on `main` after the release PR merges

The release policy — who bumps, who tags, on which branch, and why the merge
method does not matter — is stated once, in
[`CLAUDE.md` → Shipping cycle](../../CLAUDE.md#shipping-cycle).
This page covers the cross-repo half only: pinning and promotion.

## Promoting code to production

### 1. Tag a release in quantbox

**`/ship` on `dev`. That is the whole command.** It bumps, changelogs, refreshes
`uv.lock`, commits, cuts the annotated tag, pushes and opens the promotion PR —
one act. Merge that PR **with a merge commit, never squash**.

The policy, the reasons and the failure it prevents live in ONE place:
[`CLAUDE.md` -> Shipping cycle](../../CLAUDE.md#shipping-cycle).
This page deliberately keeps no command inventory — the copy that used to live
here went stale across a `/ship` rewrite and told readers `/ship` does not tag and
that squashing is safe, both of which now produce the v0.4.0 incident it warns
about. A restated policy drifts; a link cannot.

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
