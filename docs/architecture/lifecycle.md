# Methodology Lifecycle

How a research idea becomes a locked methodology and then a production tag, with reproducibility guarantees and a way to walk it back. Three states by default; deprecation/retirement come later.

---

## States

| State | Marker | Who flips it | What's required |
|---|---|---|---|
| `research` | scratch-plugin OR entry-pointed plugin with `meta.status="research"` and no LOCKED spec | LLM, freely | EXPERIMENTS.md entry referencing the plugin + at least one validated run |
| `locked` | spec frontmatter `status: LOCKED` + `meta.status="locked"` | human via `/promote-lock` | Backtest + OOS metrics meeting config thresholds; peer review note in spec |
| `production` | git tag `prod-{subsystem}-vX.Y.Z-YYYYMMDD` | human via `/promote` | Locked + reproducibility pins satisfied + CI green |

Optional/later: `scratch`, `shadow`, `deprecated`, `retired`. Not in v1 — add when needed.

---

## State transitions

```
  hypothesis (idea, not coded)
       │  EXPERIMENTS.md entry + scratch-plugin scaffolded
       ▼
   research  ←───────────────────┐
       │                          │ revalidation fail (auto-demote)
       │  /promote-lock           │
       ▼                          │
    locked  ──────────────────────┤
       │                          │ revalidation fail (auto-demote)
       │  /promote (creates tag)  │
       ▼                          │
  production ─────────────────────┘
```

A failed revalidation in `locked` or `production` automatically demotes back to `research` (or alerts only — configurable per methodology).

---

## What goes with each state

| Artifact | research | locked | production |
|---|---|---|---|
| Plugin code | ✅ — at `research/{study}/` or `src/{project}/plugins/` | ✅ — at `src/{project}/plugins/` | ✅ — pinned at tag |
| Methodology spec | optional — DRAFT | required — LOCKED frontmatter | required — LOCKED frontmatter |
| EXPERIMENTS.md entry | required | required | required (with prod-tag reference) |
| Backtest validation | recommended | required | required, passing thresholds |
| Reproducibility pins (uv.lock + dataset hashes + seeds) | not required | recommended | required (`--strict` enforces) |
| `--strict` mode | refused | acceptable | required |
| Git tag | none | none | `prod-{subsystem}-vX.Y.Z-YYYYMMDD` |

---

## Reproducibility — the three pins

Production methodologies must pin all three. Locked methodologies should. Research can skip.

### 1. `uv.lock`

Committed at every prod tag. Dependency graph reproducible bit-for-bit a year later.

### 2. Dataset content-hashes

The run manifest records `sha256(parquet_bytes)` for every input dataset. A year later, you can reconstruct the exact data slice that was used (via DVC or layered storage).

```json
// artifacts/{run_id}/lineage.json
{
  "datasets": [
    {"name": "fund_prices", "path": "data/published/fund_prices.parquet",
     "sha256": "abc123...", "as_of": "2026-04-25"},
    {"name": "izfia_assets", "path": "data/published/izfia_assets.parquet",
     "sha256": "def456...", "as_of": "2026-04-25"}
  ],
  "plugin_hashes": {
    "dm_evo.strategy.fund_scoring.v1": "ghi789..."
  }
}
```

`--strict` mode refuses to run if any required dataset can't be content-hashed.

### 3. Random seeds

ML training, bootstrap CIs, optimizer init — all need explicit seeds. Locked methodologies declare them in `params.yaml`:

```yaml
seeds:
  numpy: 42
  sklearn: 42
  bootstrap: 7
```

`--strict` mode raises if a locked methodology runs with seeds unset.

---

## Promotion paths

### From `research` to `locked`

`/promote-lock {plugin_name}`:

1. Verify EXPERIMENTS.md entry exists.
2. Verify spec frontmatter has `status: LOCKED` and required fields.
3. Run validation against locked-state thresholds.
4. Flip `meta.status` to `locked`.
5. Commit on a `feat/{subsystem}-lock-{slug}` branch.
6. Open PR to `dev` for human review.

### From `locked` to `production`

`/promote {subsystem}`:

1. Verify all reproducibility pins satisfied.
2. Run `--strict` validation.
3. Create tag `prod-{subsystem}-vX.Y.Z-YYYYMMDD` on the merge commit.
4. Update STATUS.md (or registry) with the new prod version.
5. Schedule the methodology for the monthly revalidation cron.

### From `research` (scratch-plugin) to `research` (project plugin)

When a scratch-plugin in `research/{study}/` proves out, promote it to a project entry-pointed plugin:

1. Move file to `src/{project}/plugins/{kind}/{slug}.py`.
2. Add entry-point in `pyproject.toml`.
3. Update `meta.status` (stays `research` until LOCKED).
4. Smoke test via `quantbox plugins list` → name appears.
5. Optional: open candidate PR to upstream (quantbox itself) if cross-project value.

---

## Revalidation cadence

Locked methodologies declare their revalidation policy:

```yaml
# in spec frontmatter
revalidation:
  cadence: monthly
  baseline_metrics:
    sharpe_oos_min: 0.4
    drawdown_max: -0.18
  on_failure: alert_only   # or: auto_demote_to_research
```

A monthly `quantbox-revalidate` cron job:

1. Reads the registry — every `locked` and `production` plugin.
2. Re-runs validation against fresh data.
3. Compares to declared baseline metrics.
4. Posts a Discord summary to `#quant`.
5. Auto-demotes (if configured) any methodology that drifted past thresholds.

This catches "the data moved underneath us" before it bites in production.

---

## Spec frontmatter — required fields for `locked`

```yaml
---
methodology: tactical-signals          # subsystem slug
version: v4.5.0                         # semver, matches prod-{subsystem}-v4.5.0-... tag
status: LOCKED                          # DRAFT | LOCKED | SUPERSEDED
date_locked: 2026-04-23                 # ISO date
supersedes: v3.0.0                      # previous version (optional)
src_paths:                              # which code this spec describes
  - src/dm_evo/plugins/strategy/tactical_signals.py
seeds:                                  # required for `--strict`
  numpy: 42
revalidation:
  cadence: monthly
  baseline_metrics: {...}
  on_failure: auto_demote_to_research
sections:                               # optional partial-lock
  - id: "§3.5"
    status: LOCKED
    date: 2026-04-23
---
```

`/promote` validates these fields and cross-checks against `promote.config.yaml` (per-subsystem gating).

---

## STATUS dashboard

`quantbox lifecycle status` reads `meta.status` from every registered plugin and renders:

```
Methodology         Status        Last revalidation   Drift
──────────────────────────────────────────────────────────────
dm_evo.taa.v4.8     production    2026-04-15         OK
dm_evo.fundsel.v2.1 production    2026-04-15         OK
dm_evo.fundsel_ml   research      —                  —
quantlab.trend.v3   locked        2026-04-12         WARN (sharpe -8%)
```

Markdown view of the same is updated by the cron at `inbox/reports/lifecycle/status-latest.md`.

---

## Deprecation (later)

When a successor goes to `production`, the predecessor is marked `deprecated`:

- Plugin code stays (reproducibility for old runs).
- New runs warn "this methodology is deprecated, use {successor}."
- After grace period (≥6 months) and zero live consumers, it can be marked `retired`.

Not in v1.

---

## Anti-patterns

| Anti-pattern | Fix |
|---|---|
| Editing a `production` plugin in-place | Bump version, ship new plugin name; old version stays for reproducibility |
| Skipping the EXPERIMENTS.md entry | Required at every state; the audit trail is non-optional |
| Locked methodology running without seeds | `--strict` refuses; if you bypass strict, you're not actually locked |
| Re-running a year-old prod tag without `uv.lock` | If lockfile is stale, the run is not the same methodology |
| Auto-promoting from research to production | Always two-step: human flips lock, then human flips prod |

---

## See also

- [plugin-authoring.md](plugin-authoring.md) — `meta.status` field and the Protocol.
- [api-layers.md](api-layers.md) — `--strict` mode applies at L5.
- [skills.md](skills.md) — `quantbox-promote` skill drives state transitions.
