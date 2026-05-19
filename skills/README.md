# QuantBox Skills

LLM-facing API. Skills produce YAML configs (or, when capability is missing, scaffold scratch-plugins). They live in this repo so they version with the SDK they describe.

> **Read [`docs/architecture/skills.md`](../docs/architecture/skills.md) before adding or modifying a skill.** It defines the frontmatter contract, the layer-default rule, and the capability-gap branch.

---

## Layout (target)

```
skills/
├── quantbox-core/SKILL.md              # mental model — contracts, registry, runner, ArtifactStore
├── quantbox-cookbook/configs/SKILL.md            # YAML grammar + validation
├── quantbox-cli/SKILL.md               # quantbox plugins/validate/run/new
├── quantbox-research/SKILL.md          # capability — checks/composes/authors
├── quantbox-backtest/SKILL.md          # capability — vectorbt at L1, pipeline at L4
├── quantbox-strategy-author/SKILL.md   # authoring — scaffolds StrategyPlugin
├── quantbox-data-author/SKILL.md       # authoring — scaffolds DataPlugin
├── quantbox-feature-author/SKILL.md    # authoring — scaffolds FeaturePlugin (catch-all)
├── quantbox-autoresearch/SKILL.md      # capability — LLM-driven continuous improvement loop (stub, blocked on driver)
├── quantbox-promote/SKILL.md           # lifecycle — research → locked → production
├── quantbox-revalidate/SKILL.md        # lifecycle — drift checks against locked baseline
│
├── investment-research/SKILL.md        # generic quant research methodology
├── qrd/SKILL.md                        # Quant Research Doc structure
├── paper-reading/SKILL.md              # extract method/data/results from papers
├── market-datasets/SKILL.md            # data source routing (Stooq, NBP, Yahoo, FRED, ...)
├── analizy-pl-data/SKILL.md            # Polish fund data
├── pipeline-docs/SKILL.md              # 4-doc pattern for pipelines
└── README.md                           # this file
```

The `quantbox-*` skills are foundation/capability/authoring/lifecycle — they describe quantbox itself.
The non-prefixed skills are generic quant skills that ship with quantbox templates.

Project-specific skills (e.g., `dm-evo-fund-selection`) live in the *project's* `.claude/skills/`, not here.

---

## Adding a skill

See [`docs/playbooks/add-a-skill.md`](../docs/playbooks/add-a-skill.md).

Required frontmatter for capability skills:

```yaml
---
name: skill-name
description: One sentence + trigger phrases
default_layer: L1
escalation_rules:
  - to: L4
    when: "logged experiment / EXPERIMENTS.md entry"
  - to: L5
    when: "production with reproducibility pinning"
requires_quantbox_min: "0.2.0"
---
```

---

## Skill categories

| Category | Examples | Lives where |
|---|---|---|
| Foundation | `quantbox-core`, `quantbox-config`, `quantbox-cli` | here |
| Capability | `quantbox-research`, `quantbox-backtest` | here |
| Authoring | `quantbox-strategy-author`, `quantbox-data-author` | here |
| Lifecycle | `quantbox-promote`, `quantbox-revalidate` | here |
| Generic quant | `qrd`, `paper-reading`, `investment-research`, `market-datasets` | here |
| Domain | `dm-evo-fund-selection` | the project's `.claude/skills/` |

---

## Status

This directory is a **mount point** at the moment. Skills will be authored as the corresponding architecture pieces ship:

| Skill | Blocked on |
|---|---|
| `quantbox-core`, `quantbox-config` | nothing — can be authored now |
| `quantbox-backtest` | L0/L1 surface (`quantbox.bt`, `adapters.vectorbt`) |
| `quantbox-research` | local-source plugin loader + capability-gap branch in runner |
| `quantbox-strategy-author` | `meta.status` field in `PluginMeta` |
| `quantbox-autoresearch` | `AutoResearchDriver`, `VariantProposerPlugin` Protocol, adapters for optuna + anthropic, EXPERIMENTS.jsonl format |
| `quantbox-promote` | lifecycle state transitions in registry |
| `quantbox-revalidate` | revalidation cron + spec frontmatter parser |
| Generic quant skills | (none — historical migration note removed) |

See [`docs/architecture/skills.md`](../docs/architecture/skills.md) for the full skill design and [`docs/playbooks/add-a-skill.md`](../docs/playbooks/add-a-skill.md) for the authoring flow.
