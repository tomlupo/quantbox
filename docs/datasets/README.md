# Datasets

Human-readable schema and semantics docs for each artifact type QuantBox produces or consumes. Pairs with the machine-readable JSON schemas in [`../../schemas/`](../../schemas/).

For the template, see [`_template.md`](_template.md).

---

## When to write a dataset doc

- A new artifact type is added to `schemas/` — every `*.schema.json` should have a paired prose doc.
- Existing artifact's semantics evolve in a way the JSON schema doesn't capture (NULL conventions, monotonicity guarantees, freshness contracts, business meaning).
- A pipeline's output is consumed by ≥2 downstream pipelines and the contract needs to be explicit and auditable.

## When NOT

- One-off research outputs (those go in `research/{study}/`, not here).
- Internal-to-a-plugin data structures (no external consumers).
- Re-stating what's already in the JSON schema — only add the *prose layer*. The JSON file is authoritative for shape; the prose is authoritative for meaning.

## What goes in the prose layer that the JSON can't express

| Concern | JSON schema | Prose doc |
|---|---|---|
| Column types | ✅ | — |
| Column required/optional | ✅ | — |
| Column **semantics** (what it *means*) | ❌ | ✅ |
| Source per column | ❌ | ✅ |
| Producer / consumer wiring | ❌ | ✅ |
| Freshness contract | ❌ | ✅ |
| NULL conventions, uniqueness, monotonicity | partial | ✅ |
| Loading examples | ❌ | ✅ |
| Known gotchas | ❌ | ✅ |

## Required sections

| Section | Purpose |
|---|---|
| What this is | One-paragraph human description |
| Schema | Columns, types, semantics, source per column |
| Coverage | Date range, frequency, freshness contract |
| Producer | Which plugin / pipeline emits this |
| Consumers | Which pipelines read this |
| Quality contract | NULL conventions, uniqueness, monotonicity |
| Loading examples | Python + DuckDB |
| Known issues | Edge cases, gotchas |

## Naming convention

File name = logical artifact name with hyphens: `fund-prices.md`, `strategy-weights.md`, `experiments-jsonl.md`.

The file lives next to its peers in this directory; cross-references use relative paths.

## Index

| Dataset | Producer | Schema |
|---|---|---|
| *(none yet — dataset docs land here as artifacts stabilize)* | | |

## Candidate datasets (when ready to write)

| Anticipated doc | Pairs with |
|---|---|
| `prices.md` | [`schemas/prices.schema.json`](../../schemas/prices.schema.json) |
| `strategy-weights.md` | [`schemas/strategy_weights.schema.json`](../../schemas/strategy_weights.schema.json) |
| `allocations.md` | [`schemas/allocations.schema.json`](../../schemas/allocations.schema.json) |
| `scores.md` | [`schemas/scores.schema.json`](../../schemas/scores.schema.json) |
| `rankings.md` | [`schemas/rankings.schema.json`](../../schemas/rankings.schema.json) |
| `orders.md`, `fills.md`, `targets.md`, `rebalancing.md` | trading artifacts |
| `experiments-jsonl.md` | autoresearch memory format |

## See also

- [`_template.md`](_template.md) — fill-in scaffold for new dataset docs.
- [`../../schemas/`](../../schemas/) — machine-readable JSON schemas.
- [`../architecture/plugin-authoring.md`](../architecture/plugin-authoring.md) — how plugins declare their output schema.
