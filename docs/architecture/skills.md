# Skills

QuantBox skills are the LLM-facing API. They live in this repo (`skills/`) so they evolve in lockstep with the SDK they describe. Skills are how an LLM (or a human) goes from intent to either a config (preferred) or — when capability is missing — a new plugin.

---

## Where skills live

```
quantbox/
├── skills/
│   ├── quantbox-core/SKILL.md            # mental model
│   ├── quantbox-config/SKILL.md          # YAML grammar
│   ├── quantbox-research/SKILL.md        # capability skill — composes / authors
│   ├── quantbox-backtest/SKILL.md        # capability skill
│   ├── quantbox-strategy-author/SKILL.md # authoring skill
│   ├── quantbox-promote/SKILL.md         # lifecycle skill
│   ├── investment-research/SKILL.md      # generic quant research
│   ├── qrd/SKILL.md                      # quant research doc
│   └── ...
```

Project-specific skills (e.g., `dm-evo-fund-selection`) live in the *project's* `.claude/skills/`, not here. Quantbox owns generic + foundation; projects own their domain.

---

## Skill categories

| Category | Lives in | Purpose | Example |
|---|---|---|---|
| **Foundation** | `quantbox/skills/` | Teach LLM the contracts, CLI, config grammar | `quantbox-core`, `quantbox-config` |
| **Capability** | `quantbox/skills/` | Map task type → plugin composition + recipe | `quantbox-backtest`, `quantbox-research` |
| **Authoring** | `quantbox/skills/` | When the LLM extends quantbox itself | `quantbox-strategy-author`, `quantbox-data-author` |
| **Generic quant** | `quantbox/skills/` | Research methodology, reporting, market data | `qrd`, `paper-reading`, `investment-research` |
| **Domain** | each project's `.claude/skills/` | Project-specific intent → quantbox config | `dm-evo-fund-selection` |

---

## The two rules every skill follows

### Rule 1 — Skills produce configs first, code only when capability is missing

The default output of a capability skill is a YAML config that composes existing plugins. Code (a scratch-plugin) is the escape hatch when the existing plugin set doesn't cover the request.

This keeps skills declarative and resistant to API drift.

### Rule 2 — Lowest viable abstraction

Every capability skill teaches the LLM the [layer table](api-layers.md) and picks the lightest layer that solves the task. A 5-line vectorbt sanity check belongs at L0/L1, not in a YAML pipeline.

Skills with `default_layer: L1` are the most common. Skills that always produce L4 configs are usually doing too much ceremony.

---

## Frontmatter contract

Every skill declares its layer behavior in frontmatter:

```yaml
---
name: quantbox-backtest
description: Use when the user wants to run a backtest. Triggers — "backtest", "test this strategy", "compare A vs B against historical data."
default_layer: L1
escalation_rules:
  - to: L2
    when: "comparing two or more strategies on the same data"
  - to: L4
    when: "task requires logged experiment / EXPERIMENTS.md entry"
  - to: L5
    when: "production run / reproducibility pinning required"
requires_quantbox_min: "0.2.0"
---
```

| Field | Purpose |
|---|---|
| `name` | Unique skill identifier (matches directory name) |
| `description` | One sentence + trigger phrases for skill matching |
| `default_layer` | The layer the skill reaches for first |
| `escalation_rules` | When to escalate up the layer stack |
| `requires_quantbox_min` | Minimum quantbox version this skill describes |

Skills without the layer fields are considered incomplete and won't pass review.

---

## Skill body structure

Recommended template:

```markdown
# quantbox-backtest

## Pick the layer

| Task shape | Use | Code |
|---|---|---|
| "Does this idea even work?" | L0/L1 | vbt or qbt.run |
| "Compare A vs B" | L2 | quantbox.compare |
| "Log this for EXPERIMENTS.md" | L4 | YAML + run_from_config |
| "Production run" | L5 | quantbox run --strict |

Default to L1.

## L1 — fast path

[code example, 10–20 lines]

## L4 — logged experiment

[YAML example with comments]
[reference cookbook/configs/{name}.yaml]

## When the existing plugin set is missing capability

Invoke quantbox-strategy-author. Build a scratch-plugin in research/{study}/.
Return here once registered.

## Pitfalls

- PIT: don't index past asof
- Schema: validate before run
- Status: scratch-plugins refused under --strict
```

The "pick the layer" table is the **first thing** the skill teaches. Everything else is layer-specific guidance.

---

## The capability gap branch

Capability skills must check what's available before composing:

```
1. Parse user intent → identify required capabilities
2. quantbox plugins list --json | filter by capability tags
3. If all matched → compose YAML, validate, run (L4) or call helper (L1)
4. If missing:
   a. Classify gap level (param tweak / new plugin / new contract)
   b. If level ≤ 2: invoke quantbox-{strategy|data|feature}-author
      → scaffold scratch-plugin in research/{study}/
      → return to compose
   c. If level 3 (new Protocol needed): invoke quantbox-contract-rfc → halt
5. Run + record EXPERIMENTS.md entry
6. On success: invoke quantbox-promote (offers project / upstream / keep)
```

This is what stops skills from silently failing when the LLM asks for something quantbox doesn't yet support.

---

## Authoring skills

Authoring skills generate plugin code. They're invoked as subroutines by capability skills when a gap is detected.

Each authoring skill teaches:
- The relevant Protocol (StrategyPlugin, DataPlugin, FeaturePlugin)
- The required `meta = PluginMeta(...)` fields
- The `params_schema` shape
- The `meta.status="research"` default for new code
- The output schema the plugin must produce
- Where to place the file (`research/{study}/` for scratch, `src/{project}/plugins/` for promoted)

Authoring skills produce code, not configs. That's the exception to Rule 1.

---

## Versioning skills with quantbox

Skills version with the SDK. When quantbox bumps a contract:

- Skills describing affected APIs bump `requires_quantbox_min`.
- The skill body updates to match the new API.
- Skills that no longer apply (e.g., a capability got removed) are deprecated, not deleted — left in place with a `deprecated: true` field and a pointer to the successor.

The repo CI runs a smoke check that all skill examples (in fenced ```yaml or ```python blocks) parse against current contracts.

---

## Anti-patterns

| Anti-pattern | Fix |
|---|---|
| Skill writes Python by default | Should produce config; code only when capability missing |
| Skill defaults to L4/L5 for casual work | Reset `default_layer` to L1 |
| Skill describes a stale API | Bump `requires_quantbox_min`; update body |
| Skill duplicates content from another skill | Cross-reference, don't restate |
| Skill silently invents a new plugin type | Refuse; route to `quantbox-contract-rfc` |
| Skill body doesn't reference cookbook/configs/ or schemas/ | Add references — single source of truth |

---

## See also

- [api-layers.md](api-layers.md) — what each layer means.
- [plugin-authoring.md](plugin-authoring.md) — what authoring skills generate.
- [lifecycle.md](lifecycle.md) — what `meta.status` means at scaffold time.
- [playbooks/add-a-skill.md](../playbooks/add-a-skill.md) — step-by-step.
