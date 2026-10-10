---
adr: 0010
title: The surface is organised by audience — core, plugins, research, trade — in one distribution
status: accepted
date: 2026-10-09
supersedes: "ADR-0002 (Layered API, L0–L5)"
superseded_by:
amends: "Spec TOM-1325 decision D3 (three distributions in one uv workspace): superseded by one distribution with four import-linter layers."
status_changes:
  - 2026-10-09: accepted with TOM-1449 (4a of EPIC P4, TOM-1444) — Tom, 2026-10-09 (P4 grilling, round 2, Q1–Q3, Q6, Q7): one distribution, four layers, a lean base install with extras, the CLI in core, stays 0.x
---

# ADR-0010: The surface is organised by audience, in one distribution

## Context

[ADR-0002](0002-layered-api.md) organised the surface as a ladder of ceremony,
L0 to L5. Six months later the ladder does not describe the code:

- L1 promised `quantbox.opt` and `quantbox.score`. Neither was built.
  `quantbox.bt` is the only L1 module.
- L2 ("composable units") was never built.
- L3, L4 and L5 are not three layers of code. They are two ways to call the
  same plugins: from Python, or from YAML through the runner and the CLI.
- The ladder says nothing about who installs what. That is the question P4
  must answer: a client install (robo) must carry core and must not carry
  vectorbt, numba or ccxt.

The quantbox 1.0 spec (TOM-1325) answered the install question with decision
D3: three distributions in one uv workspace. The P4 cards measured the cost
(TOM-1451 § state of the code): 183 modules, about 57k LOC, and edges that
cross every proposed boundary. Three distributions would multiply the
release, pin and lockfile work by three for every consumer.

## Decision

1. **The axis is the AUDIENCE.** Three audiences use quantbox:
   - **client core**: a client install such as robo. It defines strategies and
     data on the contracts and runs them through the runner and the rsims
     engine. It reads metrics, inference and gates.
   - **research**: the lab. It adds the backtest pipeline and its variants,
     the sweep, arms, validation plugins, reports, simulation, the warehouse,
     the vectorbt engine and the `quantbox.bt` / `quantbox.adapters.vectorbt`
     re-exports.
   - **trade**: live. It adds brokers, the trading pipeline, rebalancing,
     reconciliation, order generation and publishers.
2. **Two entry styles, for every audience.** Neither is "lower" than the
   other.
   - The **Python API on the contracts**: instantiate a plugin and call it,
     or call the core modules (`quantbox.strategy_runner`,
     `quantbox.engine.simulate`, `quantbox.metrics`, `quantbox.inference`,
     `quantbox.gates`).
   - **YAML through the runner**: `quantbox.runner.run_from_config` or
     `quantbox run -c`. It writes the run manifest and the lineage.
     Strict mode (`run.strict: true`) is a property of the run, not a layer.
3. **One distribution. Spec decision D3 is superseded.** quantbox stays one
   package. The audiences become four layers inside it, in import order:
   `core` < `plugins` < `research`, `trade`. A layer imports only the layers
   below it. `research` and `trade` never import each other. import-linter
   layer contracts enforce this in CI.
   - **core**: contracts, registry, strategy runner, decision, frequency,
     metrics, inference, gates, the engine seam with the rsims engine,
     dataset and lock, store, run manifest, schemas, config loading,
     exceptions, the CLI.
   - **plugins**: builtin strategies, datasources, features, overlays,
     monitors, and the public `quantbox.universe` and `quantbox.market_cap`.
   - **research** and **trade**: as in decision 1.
   The module-by-module map is TOM-1451 § What to build. It is not copied here.
4. **A lean base install, with extras per audience.** The base install carries
   what core imports. `[data]` carries the data-source clients, `[research]`
   carries the research layer's libraries, `[trade]` carries the venue
   clients. `[vectorbt]`, `[full]` and the venue extras stay, and `[full]`
   includes the new ones. `pyproject.toml` is the owner of each list.
5. **The CLI is in core.** A command that needs a research or trade library
   loads it lazily. When the extra is missing, the command fails and names
   the extra, as `engine: vectorbt` does today (`MissingExtraError`).
6. **The public surface is declared.** A public module declares `__all__`.
   A name a consumer imports from a private module gets a public home. 4a
   adds three homes for the four names that consumers import today:
   - `quantbox.universe`: `select_universe`, `DEFAULT_STABLECOINS` (plugins).
   - `quantbox.market_cap`: `load_pit_market_cap`, the live
     `MarketCapProvider`, and one Hyperliquid k-prefix symbol mapping that both
     use (TOM-1419; plugins, `[data]`).
   - `quantbox.dataset.load_pinned_dataset` (core).
7. **The shim rule.** A name that moves keeps its old import path for one
   minor version. The old path resolves to the same object and emits a
   `DeprecationWarning` that names the new path, through
   `quantbox._deprecation.moved` (the helper of ADR-0009 decision 6). The
   contract step, 4d (TOM-1457), removes import shims only. YAML plugin-id
   aliases stay: a config written today keeps running.
   *Amended by TOM-1457 (4d, v0.13.0):* the import shims are removed. An old
   module path is a tombstone file and an old name in a live module is refused
   by its `__getattr__`; both raise `ImportError` naming the replacement, from
   one list, `quantbox._removed.REMOVED`. `quantbox._deprecation` is gone.
8. **This ADR decides; 4b builds.** The layers, the import-linter contracts,
   the edge cuts and the extras are built by 4b (TOM-1451). Until 4b lands,
   the layer map is the target, and the code on `dev` does not obey it yet.
9. **ADR-0001 needs no new amendment.** ADR-0008 decision 10 already amended
   it: book simulation sits behind the engine seam, and the native vectorbt
   object stays reachable in research. ADR-0001's pointer to ADR-0002 now
   leads here, through ADR-0002's `superseded_by`.

## Alternatives considered

### A. Keep the L0–L5 ladder

**Rejected because:** three of its six rungs describe no code (L1 beyond
`quantbox.bt`, L2) or the same code twice (L3/L4/L5). It cannot express the
client boundary, which is the reason P4 exists.

### B. Three distributions in one uv workspace (spec D3)

**Rejected because:** every consumer would pin and lock three packages that
release together, and the edges listed on TOM-1451 must be cut anyway. Layers
in one package give the same boundary with an exit code (import-linter) and
one version number.

### C. One package, no enforced layers

**Rejected because:** the client boundary would be a convention again. The
robo install tree carried vectorbt and numba because nothing failed when it
did.

## Consequences

- Releases stay 0.x. 4a and 4b ship together as v0.12.0; 4d ships in a later
  minor (Tom, 2026-10-09).
- `docs/architecture/api-layers.md` describes the audiences and the two entry
  styles. The L0–L5 labels survive only in the skill frontmatter field
  `default_layer`, defined in `docs/architecture/skills.md`; they no longer
  describe modules or packaging.
- `tests/test_docs_module_refs.py` keeps the docs honest: a doc that names a
  `quantbox.*` module or attribute that does not exist fails CI. ADRs are
  skipped as dated records, so this ADR and ADR-0002 may name planned modules.
- A consumer that imports a private path gets a warning for one minor, then
  an `ImportError` after 4d. The cross-repo list of such imports is on
  TOM-1449.
