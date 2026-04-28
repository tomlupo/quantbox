# Architecture & Design

Read these in order. The first two are mandatory before modifying anything; the rest are needed when touching their domain.

| # | Document | Description |
|---|---|---|
| 1 | [**principles.md**](principles.md) | The doctrine. Composer-not-competitor, lowest-viable-abstraction, layered API, owned conventions, adapter-not-reimplementation. Read this first, every time. |
| 2 | [**api-layers.md**](api-layers.md) | The L0–L5 table. Operational rule for "which layer should this live at." Skill defaults reference this. |
| 3 | [**plugin-authoring.md**](plugin-authoring.md) | Plugin types, `meta.status`, registration paths, naming, testing. |
| 4 | [**adapters.md**](adapters.md) | The wrap-don't-rebuild rule with examples. When to add an adapter, when not. |
| 5 | [**skills.md**](skills.md) | LLM-facing API conventions. Frontmatter contract, capability-gap branch, authoring skills. |
| 6 | [**lifecycle.md**](lifecycle.md) | `meta.status` state machine, reproducibility pins, promotion, revalidation cadence. |
| 7 | [**templates.md**](templates.md) | Project bootstrap (`quantbox new`), the four templates, convergence model. |
| 8 | [**autoresearch.md**](autoresearch.md) | LLM-driven continuous improvement loops over strategies — `AutoResearchDriver` (L4). |
| 9 | [**pipeline-design.md**](pipeline-design.md) | Why imperative orchestration over declarative pipelines (existing). |

Once you've read principles + api-layers, the others are reference material — pull what's relevant.

For step-by-step how-tos, see [../playbooks/](../playbooks/).
For historical decisions, see [../adr/](../adr/).
