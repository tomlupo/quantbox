# Methodology

Each entry explains *why* a plugin / algorithm / scoring model works the way it does. Methodology docs pair with plugin code (`src/...`) and feed the lifecycle gates — the LOCKED frontmatter is enforced by `/promote-lock` and `/promote`.

For the lifecycle state machine and frontmatter contract, see [`architecture/lifecycle.md`](../architecture/lifecycle.md).
For the template, see [`_template.md`](_template.md).

---

## When to write a methodology doc

- Your plugin is being promoted from `research` → `locked`. The methodology spec is *required* for the lock.
- You want to record *why* a non-obvious design choice was made (signal polarity, lookback choice, gating thresholds, blend weights).
- A future maintainer would ask "why does this work?" and would need an authoritative answer.

## When NOT

- Plain Python utilities with obvious behavior — code + docstrings suffice.
- One-off research scripts that won't be promoted — record in EXPERIMENTS.md instead.
- Pure infrastructure (adapters, plugin scaffolding) — those are documented in `architecture/`.
- Pre-lock drafts beyond a sentence — write the spec when you're approaching lock; don't pre-spec speculative work.

## Frontmatter contract — required fields for `LOCKED`

| Field | Why |
|---|---|
| `methodology` | Subsystem slug; must match a key in `promote.config.yaml` |
| `version` | Semver; matches the eventual `prod-{slug}-v{version}-...` tag |
| `status` | `DRAFT` \| `LOCKED` \| `SUPERSEDED` |
| `date_locked` | ISO date the status flipped to LOCKED |
| `src_paths` | Code files this spec describes |
| `seeds` | Required for `--strict` mode reproducibility |
| `revalidation` | Cadence + baseline metrics + on_failure policy (the cron reads this) |

Optional `sections:` array enables partial-lock — promote `§3.5` without locking `§4`.

`/promote-lock` and `/promote` cross-check this frontmatter against `promote.config.yaml`. Stale dates and mismatched paths are rejected.

## Conventions

- One file per methodology version. When you bump major (breaking change), copy to `vN+1` and supersede; don't mutate.
- File name format: `{methodology-slug}-v{major}.md` (e.g., `tactical-signals-v4.md`).
- Always link to the EXPERIMENTS.md entry that justified the lock.
- Mathematical notation: KaTeX-compatible LaTeX in fenced blocks.
- Validation evidence is *part* of the spec, not a separate artifact — table of metrics with run IDs.

## Index

| Methodology | Status | Version | Last locked | Plugin |
|---|---|---|---|---|
| *(none yet — methodology docs land here as plugins are promoted to LOCKED)* | | | | |

## See also

- [`_template.md`](_template.md) — fill-in scaffold for new methodology specs.
- [`../architecture/lifecycle.md`](../architecture/lifecycle.md) — state machine and reproducibility pins.
- [`../playbooks/promote-a-methodology.md`](../playbooks/promote-a-methodology.md) — how a methodology gets locked and tagged.
