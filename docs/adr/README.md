# Architecture Decision Records (ADRs)

Decisions that shape QuantBox's identity. Each ADR captures one decision: what was decided, why, what alternatives were rejected, and what consequences flow from it.

---

## When to write an ADR

Write one when:

- The decision is hard to reverse later.
- Reasonable engineers could disagree, and you want to lock in *which* choice and *why*.
- The decision shapes downstream design (other ADRs will reference it).
- A future reader asking "why is it this way?" deserves a definitive answer.

Don't write one for:

- Reversible implementation choices (variable names, file layouts within a module).
- Trivial config tweaks.
- Things already documented in `architecture/`.

If unsure, ask: would I want to refer back to this in 12 months? If yes, ADR.

---

## Format

Each ADR is a numbered file: `NNNN-short-slug.md`. ADRs are immutable once accepted; supersede them with a new ADR rather than editing.

For the fill-in scaffold, use [`_template.md`](_template.md). It includes the required frontmatter, section structure (Context / Decision / Alternatives / Consequences / Notes), and the conventions for status changes.

---

## Index

| # | Title | Status | Date |
|---|---|---|---|
| [0001](0001-library-not-framework.md) | Composing framework, not competing framework | accepted | 2026-04-25 |
| [0002](0002-layered-api.md) | Layered API (L0–L5) | accepted | 2026-04-25 |
| [0003](0003-autoresearch-as-driver-not-runtime.md) | Autoresearch as L4 driver, not runtime mode | accepted | 2026-04-25 |

(Add new entries at the bottom; renumber-cleanup is forbidden — numbers are stable references.)

---

## Rules

1. **Never edit an accepted ADR.** Supersede it with a new one.
2. **Never reuse a number.** Even if an ADR is rejected before acceptance, that number is burned.
3. **Status changes are themselves dated.** Add a "Status changes" section if the status moves.
4. **Reference ADRs from architecture docs.** When a doc encodes an ADR's decision, link back: `(see ADR-0001)`.
5. **LLMs cannot self-author ADRs.** Drafts only — humans accept.

---

## See also

- [architecture/principles.md](../architecture/principles.md) — the doctrine encodes the accepted ADRs.
- [architecture/](../architecture/) — domain docs reference ADRs for "why."
