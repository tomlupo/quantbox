# Runbooks

Operational procedures for cron jobs, deploys, brokers, and recovery scenarios. Each runbook is **idempotent** (re-running a step doesn't double-execute) and **complete** (a tired operator at 03:00 can follow it without context).

For the template, see [`_template.md`](_template.md).

---

## When to write a runbook

- A scheduled task (`agent-cron`, systemd timer, cron) needs an operator-readable description.
- A deploy sequence has more than 2 steps.
- A broker setup involves API keys + rotation discipline.
- A recovery scenario has been encountered (don't write speculative runbooks for things that haven't broken — write them after the first incident).

## When NOT

- One-line commands documented in CLAUDE.md or `reference/`.
- Generic CLI usage — that's `guides/`.
- Code-level operations on plugins / methodology — that's `playbooks/`.
- Things that change weekly — runbooks should be stable; volatile procedures live in code.

## Required sections

| Section | Why required |
|---|---|
| When to invoke | Trigger conditions, not "general info" |
| Prerequisites | Operator can verify these before starting |
| Steps | Numbered, idempotent, copy-pasteable |
| Verification | How to confirm success |
| Rollback | What to do if a step fails or produces wrong state |
| Common failures | The most operator-relevant section at 03:00 |
| Escalation | Who to page when the runbook itself fails |
| Related runbooks | Optional but useful |

A runbook that lacks any of these is unfinished.

## Conventions

- Runbooks are organized by **operation**, not by service. `revalidate-locked-methodologies.md`, not `quantbox-revalidate-cron.md`.
- One file per operation. If two operations share 80% of steps, factor common bits into a third runbook and reference it.
- Steps are **commands**, not narrative. `curl -fsS $URL` beats "make a GET request to the URL."
- Every command output is described — what does success *look like*?
- Sensitive paths (env vars, API keys, secret locations) are referenced by *path* (`~/.cookbook/configs/...`), never inlined.

## Index

| Runbook | Operation | Cadence |
|---|---|---|
| *(none yet — runbooks land here as operations stabilize)* | | |

## Candidate runbooks (when ready to write)

These are the operational surfaces where a runbook will earn its keep. Write each after the first real run, not before.

| Anticipated runbook | Trigger |
|---|---|
| `revalidate-locked-methodologies.md` | Monthly cron from `quantbox-revalidate` |
| `autoresearch-tick.md` | Daily cron per client config |
| `rotate-broker-credentials.md` | Quarterly or after suspected compromise |
| `recover-from-failed-promotion.md` | A `/promote` that left the registry in a half-state |
| `restore-from-prod-tag-rollback.md` | After a hard rollback to a prior `prod-{slug}-vX-...` tag |
| `bootstrap-new-client-loop.md` | First-time setup for a new client engagement |

## See also

- [`_template.md`](_template.md) — fill-in scaffold for new runbooks.
- [`../reference/`](../reference/) — for static config and credential locations runbooks reference.
- [`../playbooks/`](../playbooks/) — for code-level how-tos (different from operational runbooks).
