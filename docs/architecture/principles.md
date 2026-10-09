# QuantBox Principles

**Read this first.** These are the load-bearing decisions. Every architecture doc, plugin, adapter, skill, and template downstream is shaped by them. Violate one and you're fighting the design.

---

## What QuantBox is

A **composing framework** — owned, opinionated, layered. Three things, in order of importance:

1. **Conventions** — your data layouts, your run-artifact shape, your lifecycle states, your skill API. The part nobody else can build for you.
2. **Adapters** — thin wrappers around best-of-breed libraries (vectorbt, riskfolio, lightgbm, ...). The wheel does the wheel's work. An adapter lives in core only when ≥2 consumers need the same bridge; single-consumer libraries are imported directly in the consuming repo.
3. **Skills + templates** — the LLM-facing interface and the project bootstrap shape. Coupled to the SDK in this repo so they never drift.

By the strict IoC sense (the runner calls *your* plugins), it is a framework. By the colloquial sense ("my framework for X"), it's also a framework. The question isn't whether to call it one — it's *what kind*. See [ADR-0001](../adr/0001-library-not-framework.md).

## What QuantBox composes — and what it doesn't reinvent

| Capability | The wheel | QuantBox's role |
|---|---|---|
| Backtest engine | vectorbt, rsims | the engine seam (`quantbox.engine`, [ADR-0008](../adr/0008-engine-seam.md)): one book function (`simulate`) owns the schedule, each adapter only executes it, the native object reachable; vectorbt re-exported in the research layer (`adapters.vectorbt`, `quantbox.bt`) |
| Portfolio optimization | riskfolio, PyPortfolioOpt, skfolio | adapter in core when ≥2 consumers need it |
| ML training | scikit-learn, lightgbm | import directly in plugin; adapter in core when ≥2 plugins need it |
| Experiment tracking / model registry | MLflow | import directly in quantbox-lab; no core adapter until ≥2 repos need same bridge |
| Data versioning | DVC | import directly in quantbox-lab; no core adapter until ≥2 repos need same bridge |
| Factor research | Qlib (optional) | import directly where needed |
| Live broker | ccxt, native broker APIs | adapter + plugin |

If a feature request implies reimplementing what one of these libraries already does, the answer is to use the library directly — and to add a core adapter only once the idiom recurs across ≥2 consumers.

---

## Five principles

### 1. Composer, not competitor

QuantBox is a framework — owned, opinionated, with IoC through the runner. The question isn't whether it's a framework; it's whether it's the *right kind* of framework.

Three failure modes to refuse:

- **Competing with the wheel** — reimplementing what vectorbt, MLflow, Qlib, riskfolio already do. Use them directly, or compose via adapters when the idiom recurs.
- **Monolithic surface** — forcing every caller through YAML and the runner. The Python API on the contracts prevents this ([api-layers.md](api-layers.md)).
- **Ceremony over conventions** — requiring plugin scaffolding for trivial work. Conventions inherit; ceremony gets bypassed.

Composing frameworks integrate; competing frameworks reinvent.

The runtime (`quantbox run -c config.yaml`) is *one* of multiple entry points — not the only way and not the default for casual use.

### 2. Lowest viable abstraction

A skill must not impose more ceremony than the task requires.

- "Does this idea even work?" → 5 lines of vectorbt
- "Compare A vs B" → one function call
- "Log this for the experiment record" → YAML config
- "Production run with reproducibility pins" → full pipeline

Every plugin is callable from Python and from YAML (see [api-layers.md](api-layers.md)). The skill picks the lighter entry style that does the job.

If a user has to reach for `import vbt` to bypass quantbox, the design has failed.

### 3. Audience layers

quantbox is one distribution in four layers: `core` < `plugins` < `research`, `trade`. A client install carries core and must not carry the research or trade libraries. A plugin is usable on its own — no "must be invoked via the runner."

The decision is [ADR-0010](../adr/0010-audience-layers-one-distribution.md); [api-layers.md](api-layers.md) says what each layer is for. The module map is the import-linter contracts in `pyproject.toml`.

### 4. Owned conventions

What quantbox uniquely provides, and won't delegate:

- `meta.status` lifecycle states (research / locked / production)
- Plugin contracts (`PluginMeta`, `RunResult`, `ArtifactStore`)
- Run artifact layout (`run_id`, `manifest`, `events.jsonl`, content-hashed lineage)
- LLM-facing skill API and `default_layer` doctrine
- Project templates (`quantbox new`)
- PIT discipline at the data boundary
- Schema validation (required-core + extensions)
- Status-aware strict mode (`run.strict`)

These are the moat. Everything else is somebody else's library.

### 5. Adapter, not reimplementation

When an external library covers a capability, you wrap it — never rebuild it.

The rule: an adapter is a thin pass-through plus optional convenience helpers. The underlying library is re-exported, not hidden.

```python
# Right
from quantbox.adapters.vectorbt import vbt  # re-export
from quantbox.bt import run  # convenience helper (research)

# Wrong
engine = VectorbtEngine(prices)  # an opaque wrapper class hiding vbt
```

If you find yourself writing logic that exists in the underlying library, stop. Use it. See [adapters.md](adapters.md) for the rule and examples.

---

## Anti-patterns to refuse

| Anti-pattern | Why it's wrong |
|---|---|
| **Forcing the runner** for one-off scripts | Violates lowest-viable-abstraction. The Python API serves casual use. |
| **Reimplementing vectorbt features** in `quantbox.plugins.backtesting` | Wheel already exists. Adapt, don't rebuild. |
| **Hiding the underlying library** behind opaque wrappers | Caller can't escape to the wheel when needed. Brittle. |
| **Rich domain types** in core (e.g., advisory profiles, regulated identifiers) | Domain belongs in projects, not in quantbox. |
| **Schema strictness** that rejects extra columns | `prefixItems` for required core; extras allowed. |
| **Plugin ceremony** for trivial work | Plugins are an *option*, not a requirement for a one-off script. |
| **Skill writes Python by default** | Skills produce configs first; code only when capability is missing. |
| **Implicit silent fallbacks** in PIT-sensitive code | Backtest correctness > convenience. Raise loudly. |
| **Frameworky abstractions** added "just in case" | Earn abstractions via use cases, not speculation. |

---

## Decision rule for new features

Before adding a capability to quantbox, answer:

1. **Does an external library already do this?** If yes → add an adapter, don't reimplement.
2. **Is this domain-specific?** If yes → it belongs in the project (dm-evo, quantlab), not in quantbox.
3. **Can this be a plain function, or does it need a full plugin?** Default to a function unless the runner adds value.
4. **Is the addition genuinely cross-project?** If only one consumer benefits, keep it in that consumer.

If answers point to "add to quantbox," then the next question is *which layer* (core, plugins, research, trade). Pick the lowest layer whose audience needs it: a client install carries only core and plugins.

---

## What to do when the principle and the request conflict

Rare but real. If a request seems to require violating a principle, that's a strong signal one of three things:

1. The principle has a missing nuance. Document it as an ADR.
2. The request is for the wrong layer (e.g., asking the runtime to do work that belongs in a script).
3. The request is for a domain-specific concern that should live in a project, not quantbox.

In all three cases, slow down. Don't paper over the principle with a special case. Fixes that violate the doctrine end up shaping the next twelve features in the wrong direction.

---

## Reading order for new contributors (or future you)

1. This file.
2. [api-layers.md](api-layers.md) — the audience layers and the two entry styles.
3. [adapters.md](adapters.md) — the wrap-don't-rebuild rule with examples.
4. [plugin-authoring.md](plugin-authoring.md) — how to add capability when adapters aren't enough.
5. [skills.md](skills.md) — LLM-facing API conventions.
6. [lifecycle.md](lifecycle.md) — status states, reproducibility, promotion.
7. [adr/](../adr/) — the decisions, why we are where we are.

After this, the [playbooks/](../playbooks/) directory has step-by-step how-tos for common modifications.
