# Runbook — {operation name}

> One-line description of what this operation does.

---

## When to invoke

- **Triggered by**: cron schedule | manual | event | upstream signal
- **Owner**: `agent-cron iris` | `systemd` | `tom` | <other>
- **Channel for alerts**: `#channel-name`
- **Healthcheck**: `<healthchecks.io URL>` (if applicable)

## Prerequisites

Operator can verify each before starting:

- [ ] Environment variable `X` set (lives in `~/.cookbook/configs/path.env`, mode 600)
- [ ] Service `Y` running: `systemctl --user is-active Y`
- [ ] Healthcheck endpoint reachable: `curl -fsS $URL`
- [ ] Disk has > N GB free: `df -h /`

## Steps

Each step is idempotent unless explicitly noted. Re-running is safe.

1. **Verify pre-state** — `<command>` should return `<expected output>`.
2. **<Action description>** —
   ```bash
   <command>
   ```
   Expected output: `<exact expected text or pattern>`.
3. **<Action description>** — ...
4. **Confirm completion** — `<verification command>` returns `<expected>`.

If any step's actual output differs from expected, **stop**. Go to "Common failures" before proceeding.

## Verification

How to confirm the whole operation succeeded:

- [ ] `<check 1>` — e.g., latest run artifact exists at expected path
- [ ] `<check 2>` — e.g., process is running with expected PID
- [ ] Healthcheck pinged — `curl <healthchecks-url>` returns 200
- [ ] Discord post visible in `#channel`

## Rollback

If steps fail or produce wrong state:

1. **Halt scheduled runs** — `<command to disable cron>` (so the bad state isn't compounded).
2. **Restore prior state** — `<rollback command>`.
3. **Verify rollback** — `<verification>`.
4. **Notify** — post to `#channel-name` with a short summary of what was rolled back and why.

If rollback itself fails, escalate (see below) — do not improvise.

## Common failures and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `<error message>` | <root cause> | `<command>` |
| Step N hangs > 60s | <root cause> | `<action>` |
| Output mismatches expected | <root cause> | <action> |

Add a row here every time the runbook actually fails in a new way. The quality of this section is the quality of the runbook.

## Escalation

If the runbook fails to recover state:

1. Page <on-call / owner / tom>.
2. Reference incident in `postmortems/` (or create one).
3. Halt automation in this area until root cause identified.
4. Do not relitigate the runbook in chat — fix the runbook *after* the incident is closed.

## Related runbooks

- [`<other-runbook.md>`](other-runbook.md) — relationship.
