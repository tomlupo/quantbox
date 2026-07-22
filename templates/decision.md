<!--
When to write: decision is hard to reverse, reasonable engineers could disagree,
or it shapes downstream design. Don't write for reversible choices or things
already in architecture/.

Rules:
- Never edit an accepted decision — supersede with a new one.
- Never reuse a number — even rejected drafts burn the number.
- LLMs draft only — humans accept.
- File name: NNNN-short-slug.md (next number = highest existing + 1).
-->
---
adr: NNNN                                             # next available number
title: Short, decision-shaped title          # imperative voice, names the choice
status: proposed                              # proposed | accepted | superseded | deprecated
date: YYYY-MM-DD
supersedes:                                  # ADR number this replaces (if any)
superseded_by:                               # ADR number that replaces this (added later)
status_changes:                              # add a row each time status changes
  # - YYYY-MM-DD: <change description>
---

# ADR-NNNN: Short, decision-shaped title

## Context

What problem are we solving? What forces are at play? What constraints exist?

State the *facts* and the *tension*. Don't argue here — just describe.

## Decision

What did we decide? Stated affirmatively, not as a debate.

If the decision has multiple parts, list them; one sentence each.

## Alternatives considered

What else was on the table? Each alternative gets its own subsection.

### A. Alternative name

Description of the alternative.

**Rejected because:** the load-bearing reason.

### B. Alternative name

Description.

**Rejected because:** the load-bearing reason.

### C. Chosen alternative

Description.

**Accepted because:** why this won. Multiple bullets allowed.

## Consequences

### Intended

- What flows from this on purpose.

### Unintended (and accepted)

- Side effects we noticed and chose to accept.

### Anti-patterns this rules out

- ❌ Things this decision forbids.

## Notes

References, prior art, links to threads where this was debated, related ADRs (use links).

If this ADR is later superseded, add a `## Status changes` section above this one with the supersedes timeline.
