# Playbook — Add a Skill

Skills are the LLM-facing API. Foundation/capability/authoring skills live in `quantbox/skills/`; project-specific (domain) skills live in the project's `.claude/skills/`. Read [architecture/skills.md](../architecture/skills.md) first.

---

## Pre-flight

Decide the category:

| Category | Examples | Where it lives |
|---|---|---|
| Foundation | `quantbox-core`, `quantbox-config` | `quantbox/skills/` |
| Capability | `quantbox-backtest`, `quantbox-research` | `quantbox/skills/` |
| Authoring | `quantbox-strategy-author`, `quantbox-data-author` | `quantbox/skills/` |
| Generic quant | `qrd`, `paper-reading`, `investment-research` | `quantbox/skills/` |
| Domain | `dm-evo-fund-selection` | project's `.claude/skills/` |

If your skill is project-specific (advisory, regulatory, brand), it belongs in the project. Otherwise it lives in quantbox.

---

## Steps

### 1. Create the skill directory

```bash
mkdir -p quantbox/skills/{skill-name}
touch quantbox/skills/{skill-name}/SKILL.md
```

For skills with reference material or scripts:

```bash
mkdir -p quantbox/skills/{skill-name}/{references,scripts}
```

### 2. Author the frontmatter

Required:

```yaml
---
name: {skill-name}
description: One sentence + trigger phrases. e.g. "Use when the user asks to backtest a strategy. Triggers — 'backtest', 'test this idea', 'compare A vs B'."
default_layer: L1                    # default escalation level
escalation_rules:
  - to: L4
    when: "task requires logged experiment / EXPERIMENTS.md entry"
  - to: L5
    when: "production run / reproducibility pinning required"
requires_quantbox_min: "0.2.0"
---
```

Capability skills always declare `default_layer`. Authoring/foundation skills can omit it (they don't operate at the layer-choice abstraction).

Optional fields:

```yaml
allowed-tools: [Read, Write, Bash]   # restrict tool surface for this skill
disable-model-invocation: false      # set true to require manual /invocation
agent: Explore                        # subagent for context-fork mode
context: fork                         # isolated subagent execution
deprecated: false                     # mark with: true and add `successor: <name>`
```

### 3. Write the body

Recommended template for capability skills:

```markdown
# {skill-name}

## Pick the layer

| Task shape | Use | Code |
|---|---|---|
| "Quick check" | L0/L1 | [helper or vbt] |
| "Compare two ideas" | L2 | quantbox.compare |
| "Log this for record" | L4 | YAML + run_from_config |
| "Production" | L5 | quantbox run --strict |

Default to **L1**.

## L1 — fast path

[10–20 line example]

## L4 — logged experiment

[YAML example, with comments]
[reference recipes/{name}.yaml]

## When the existing plugin set is missing capability

Invoke `quantbox-{kind}-author`. Build a scratch-plugin in `research/{study}/`.
Return here once registered.

## Pitfalls

- PIT: don't index past asof
- Schema: `quantbox validate -c <config>` before run
- Status: scratch-plugins refused under `--strict`
```

For authoring skills, omit the layer table; instead structure as:

```markdown
# {skill-name}

## What this skill produces

A new plugin file at `research/{study}/{slug}.py` with `meta.status="research"`.

## Required user input

- Plugin kind (Strategy / Data / Feature / ...)
- Description in plain English
- Inputs/outputs (or "infer from intent")

## Output structure

[plugin scaffold template]

## Validation

[smoke test scaffold]
```

### 4. Reference recipes (don't restate them)

If a working YAML config already exists in `recipes/`, reference it:

```markdown
For a complete L4 config, see [recipes/regime_taa.yaml](../../recipes/regime_taa.yaml).
```

Single source of truth — recipes are the canonical examples; skills point at them.

### 5. Test the skill manually

The skill should pass two checks:

- **Trigger match**: ask Claude to do the task in natural language. The skill should activate without you naming it.
- **Output validity**: the YAML or code the LLM produces should pass `quantbox validate` or run cleanly.

Note any failure modes and update the skill body to address them.

### 6. (Optional) Add a script

If the skill produces non-trivial scaffolding (multi-file project setup, complex config generation), add a script:

```bash
quantbox/skills/{skill-name}/scripts/scaffold.py
```

Skills can call scripts via Bash. Keep scripts < 100 lines; if larger, refactor into `quantbox.cli` as a proper subcommand.

### 7. Document

Add a one-line entry in `quantbox/skills/README.md` (or `quantbox/docs/reference/skills.md`):

```
| {skill-name} | {default_layer} | {one-line description} |
```

### 8. Wire into templates (if applicable)

If foundation skills are auto-installed by `quantbox new`, update the install list in `quantbox/cli.py:new` so the skill ships with new projects.

---

## Validation checklist

- [ ] Frontmatter has `name`, `description`, `default_layer`, `escalation_rules`, `requires_quantbox_min`.
- [ ] First section of the body is the "Pick the layer" table (capability skills only).
- [ ] Examples reference `recipes/` instead of inlining full YAML.
- [ ] Trigger phrases in `description` are realistic (test by asking Claude to do the task).
- [ ] No code that drifts from current plugin contracts.
- [ ] Skill passes a manual smoke test (trigger + output validates/runs).

---

## Common mistakes

| Mistake | Fix |
|---|---|
| Skill defaults to L4/L5 | Reset `default_layer: L1`; escalate only when necessary |
| Skill writes Python by default | Configs first; code only when capability is missing |
| Skill restates a recipe | Reference `recipes/` |
| Skill describes a stale API | Bump `requires_quantbox_min`; test against current contracts |
| Skill is project-specific but lives in `quantbox/skills/` | Move to the project's `.claude/skills/` |
| Skill body is too long (> 300 lines) | Move detail to `references/` and link |

---

## See also

- [architecture/skills.md](../architecture/skills.md) — conventions and the capability-gap branch.
- [architecture/api-layers.md](../architecture/api-layers.md) — what `default_layer` means.
- [add-a-plugin.md](add-a-plugin.md) — what authoring skills generate.
