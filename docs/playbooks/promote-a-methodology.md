# Playbook — Promote a Methodology

Walks a methodology from `research` → `locked` → `production`. Read [architecture/lifecycle.md](../architecture/lifecycle.md) first.

---

## Pre-flight — is this ready to lock?

Before flipping `meta.status` to `locked`, confirm:

- [ ] Plugin has been live in `research` long enough that you trust it (≥ 1 week typical).
- [ ] EXPERIMENTS.md has at least one entry referencing it.
- [ ] Backtest validation has run; metrics meet thresholds you set in advance (don't move thresholds to fit results).
- [ ] Methodology spec doc exists at `docs/methodology/{slug}.md` (DRAFT status acceptable here).
- [ ] No known open issues that would invalidate locked-state guarantees.

If any are unclear, the answer is "not yet."

---

## Step A — Promote `research` → `locked`

### 1. Update the methodology spec

`docs/methodology/{slug}.md` — flip frontmatter to LOCKED:

```yaml
---
methodology: {slug}
version: vX.Y.0
status: LOCKED
date_locked: 2026-04-25
src_paths:
  - src/{project}/plugins/{kind}/{slug}.py
seeds:
  numpy: 42
revalidation:
  cadence: monthly
  baseline_metrics:
    sharpe_oos_min: 0.4
    drawdown_max: -0.18
  on_failure: alert_only
---
```

Required fields:

| Field | Why |
|---|---|
| `version` | Semver; `/promote` matches against `prod-{slug}-v{version}-...` tag |
| `date_locked` | ISO date; `/promote-lock` rejects if older than `spec_date_max_age_days` in promote config |
| `src_paths` | Cross-checked against subsystem registry — must match the plugin |
| `seeds` | Required for `--strict` mode |
| `revalidation` | Cron pulls these; required even if `on_failure: alert_only` |

### 2. Update the plugin's `meta.status`

```python
meta = PluginMeta(
    name="{namespace}.strategy.{slug}.v1",
    ...
    status="locked",       # was "research"
    ...
)
```

### 3. Update STATUS.md (if present)

In your project's `research/{study}/STATUS.md`, move the entry from "In Flight" to "Research Locked."

### 4. Run validation

```bash
quantbox validate -c config/{slug}.yaml
quantbox run -c config/{slug}.yaml
```

Verify metrics meet `baseline_metrics`. If they don't, *don't* lock yet. Fix the methodology or revisit thresholds.

### 5. Open feat branch + PR

```bash
git checkout -b feat/{slug}-lock
git add docs/methodology/{slug}.md src/{project}/plugins/{kind}/{slug}.py
git commit -m "feat({slug}): lock methodology vX.Y.0"
gh pr create --base dev
```

PR title format: `feat({slug}): lock vX.Y.0`. Body should reference the EXPERIMENTS.md entry and validation metrics.

### 6. Human review + merge

A second person (or future-you on a different day) reviews the lock. The reviewer's job: confirm methodology spec matches the code, confirm metrics aren't cherry-picked, confirm reproducibility pins are realistic.

After merge to `dev`, the methodology is `locked`.

---

## Step B — Promote `locked` → `production`

### 1. Verify locked-state preconditions

Run the gate checklist (mostly automated by `/promote {subsystem}`):

- [ ] Spec is LOCKED (frontmatter status).
- [ ] Plugin `meta.status="locked"`.
- [ ] EXPERIMENTS.md entry references the lock.
- [ ] Backtest run results stored under `output/{slug}/` with metrics.
- [ ] `--strict` validation passes.
- [ ] CI green on the merge commit.
- [ ] Reproducibility pins satisfied: `uv.lock` committed, dataset content-hashes recorded, seeds set.

### 2. Open dev → main PR

```bash
git checkout dev
git pull
gh pr create --base main --head dev --title "promote: {slug} vX.Y.0 to production"
```

PR body should include:
- Reference to the lock PR.
- Validation metrics (sharpe, drawdown, etc.) vs current production.
- Breaking changes called out (polarity, schema, signal sign).
- Date of intended prod tag.

### 3. Merge to main

After human approval, merge to `main`.

### 4. Tag the release

```bash
git checkout main && git pull
TODAY=$(date -u +%Y%m%d)
git tag -a "prod-{slug}-vX.Y.0-${TODAY}" -m "Promote {slug} to production"
git push origin "prod-{slug}-vX.Y.0-${TODAY}"
```

Tag format: `prod-{subsystem}-vMAJOR.MINOR.PATCH-YYYYMMDD`.

### 5. Update `meta.status`

```python
meta = PluginMeta(..., status="production", ...)
```

Commit on a small `chore({slug}): mark production status` branch and merge directly to `main` (it's a single-line change post-tag).

### 6. Schedule revalidation

Confirm the monthly `quantbox-revalidate` cron picks up the new methodology. Check the next scheduled run includes it; if not, the registry is out of sync — fix before walking away.

### 7. Update lifecycle dashboard

`quantbox lifecycle status` should show the methodology in `production`. If `inbox/reports/lifecycle/status-latest.md` is stale, force a refresh.

---

## Rollback

If a production methodology fails revalidation or shows live drift:

### Soft rollback (auto-demote)

If `revalidation.on_failure: auto_demote_to_research` was set, the methodology is automatically demoted. Publishers stop, scheduled runs continue but post warnings to `#quant`.

### Hard rollback (manual)

```bash
# Find prior good version
git tag --list 'prod-{slug}-*' | sort -V | tail -3
# Reset to prior version
PRIOR_TAG=prod-{slug}-vX.Y.0-YYYYMMDD
# Re-tag head as a rollback so prod points at the prior tree
git tag -a "rollback-{slug}-${TODAY}" -m "Rollback to ${PRIOR_TAG}" "${PRIOR_TAG}^{}"
```

Don't delete the failed tag — keep it for the audit trail. Update STATUS.md to show the rollback.

---

## When to bump major vs minor on lock

| Change vs prior locked version | Bump |
|---|---|
| Bug fix only | patch (`v4.5.0` → `v4.5.1`) |
| New parameter with default, no behavior change at default | minor (`v4.5.x` → `v4.6.0`) |
| Methodology change, polarity flip, schema change | major (`v4.x.x` → `v5.0.0`) |

If you bump major, the prior production version stays available — don't delete its tag or its bundle.

---

## Common mistakes

| Mistake | Fix |
|---|---|
| Locking a methodology to fit data already seen | Set thresholds in advance; don't move them after seeing results |
| Skipping the EXPERIMENTS.md entry | Required at every state — audit trail is non-optional |
| Tagging without `uv.lock` committed | Reproducibility broken; rebuild the tag |
| Changing seeds between lock and production | Seeds are part of the methodology; changing them = new version |
| Auto-promoting (LLM running `/promote` unsupervised) | Promotion is human-gated; LLMs propose, humans flip |
| Dropping the prior production version's bundle | Reproducibility relies on it; deprecate, don't delete |

---

## See also

- [architecture/lifecycle.md](../architecture/lifecycle.md) — full state machine and reproducibility rules.
- [add-a-plugin.md](add-a-plugin.md) — how the plugin originated.
- [add-a-skill.md](add-a-skill.md) — `quantbox-promote` skill that drives this from chat.
