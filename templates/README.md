# Templates

Copy-paste scaffolds for a QuantBox-based research project. These live here as a
starting point — copy what you need into your own repo and own it there.

| Template | Copy to | Purpose |
|---|---|---|
| [`methodology.md`](methodology.md) | `docs/methodology/{slug}-v{N}.md` | Strategy spec required for promotion to LOCKED |
| [`dataset.md`](dataset.md) | `docs/datasets/{artifact}.md` | Prose semantics layer for an artifact type |
| [`runbook.md`](runbook.md) | `docs/runbooks/{operation}.md` | Operational procedure for a scheduled or manual task |
| [`decision.md`](decision.md) | `docs/decisions/DEC-NNNN-{slug}.md` | Architecture Decision Record (hard-to-reverse choices) |

## When to use each

**Methodology** — when a plugin is approaching promotion from `research` → `locked`.
The frontmatter is enforced by `/promote-lock`; the prose captures the *why*.

**Dataset** — when an artifact type has ≥2 downstream consumers and its semantics
(NULL conventions, freshness, meaning per column) need to be explicit and auditable.

**Runbook** — after the first real run of a scheduled or recovery operation, not before.
Write it once you know what actually happens, not speculatively.
