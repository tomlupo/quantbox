# CLAUDE.md — Agent onboarding guide

> **Modifying QuantBox?** Read [`docs/architecture/principles.md`](docs/architecture/principles.md) and [`docs/architecture/api-layers.md`](docs/architecture/api-layers.md) first. Every architectural decision downstream is shaped by them. This file describes the *current* code; the architecture docs describe the *target design and the rules*. When they conflict, architecture wins.

## What is QuantBox?

A **template-driven SDK with adapters** for quant research and production. Three things, in order:

1. **Conventions** — data layouts, run-artifact shape, lifecycle states, skill API. The owned moat.
2. **Adapters** — thin wrappers around best-of-breed external libraries (vectorbt, riskfolio, ...). The wheel does the wheel's work. A core adapter is added only when ≥2 consumers need the same bridge; single-consumer libraries (mlflow, dvc) are imported directly in the downstream repo.
3. **Skills + templates** — LLM-facing interface and project bootstrap, coupled to the SDK in this repo.

The plugin runtime (`run_from_config`, CLI) is *one* of multiple entry points — see the [layered API](docs/architecture/api-layers.md) (L0–L5). Casual use defaults to L0/L1 (re-exports + convenience helpers). YAML pipelines are L4. Production is L5 with `--strict`.

QuantBox is a **composing framework** — owned and opinionated, but composing external libraries (vectorbt, MLflow, riskfolio, optionally Qlib) rather than competing with them on their turf. See [ADR-0001](docs/adr/0001-library-not-framework.md).

## Task source: Linear (team TOM, project quantbox)

**Linear is the task source** (qute-code-kit ADR-0004). Tasks, planning, and agent
assignment live in Linear — see [`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md)
for the binding (machine marker `qute-tracker: linear team=TOM`). qute `/task` and
`/repo-status` route there automatically. **GitHub Issues on `tomlupo/quantbox` are issue
*records* only** — bugs/defects/tech-debt attached to the code; an issue becomes work only
when a Linear task references it. Never pull work from the Issues list directly. **Never
Paperclip** — the fleet's Paperclip orchestrator is retired.

## qute runtime

This repo runs the standard qute regime (qute-code-kit ADR-0001..0004). Key skills:

- `/task` + `/repo-status` — honor [`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md) (Linear).
- `/decision` — records ADRs to [`docs/adr/`](docs/adr/) (`NNNN-title.md`).
- `/handoff` + `/pickup` — the continuity pair for pausing/resuming work.
- `/ship` — the release boundary (commitizen). Two stages: `/ship` bumps on
  `dev`, `/ship --tag` cuts the annotated `vX.Y.Z` tag on `main` after the
  release PR merges. See [`## Shipping cycle`](#shipping-cycle-two-stage-pr-mirrors-dm-evo)
  — the one place this repo states its release policy.
- Guards (secrets, audit, destructive-command, lakera/langfuse) stay active under all workflows.

Jimek dispatch + workflow policy is declared in [`conductor.yml`](conductor.yml).
Which skill when: qute-code-kit `docs/playbooks/skill-router.md`.

## Authoritative docs (read in order)

| # | Doc | When |
|---|---|---|
| 1 | [`docs/architecture/principles.md`](docs/architecture/principles.md) | Read first, every time. The doctrine. |
| 2 | [`docs/architecture/api-layers.md`](docs/architecture/api-layers.md) | The L0–L5 table. Operational rule for "which layer." |
| 3 | [`docs/architecture/plugin-authoring.md`](docs/architecture/plugin-authoring.md) | Plugin types, `meta.status`, registration, naming, testing. |
| 4 | [`docs/architecture/adapters.md`](docs/architecture/adapters.md) | Wrap-don't-rebuild rule. |
| 5 | [`docs/architecture/skills.md`](docs/architecture/skills.md) | LLM-facing API, frontmatter contract, capability-gap branch. |
| 6 | [`docs/architecture/lifecycle.md`](docs/architecture/lifecycle.md) | `meta.status` state machine, reproducibility, promotion. |
| 7 | [`templates/README.md`](templates/README.md) | Copy-paste scaffolds for methodology, dataset, runbook, and decision-record docs. Used by `quantbox new` and consumed by `/promote-lock`. |

For step-by-step modifications, see [`docs/playbooks/`](docs/playbooks/). For historical decisions, see [`docs/adr/`](docs/adr/).

## Project layout

```
src/quantbox/              ← installable library (uv add quantbox)
  contracts.py             Protocol definitions (start here)
  runner.py                Config → plugin instantiation → pipeline.run()
  registry.py              Plugin discovery (builtins + entry points)
  cli.py                   CLI entry point (quantbox command)
  store.py                 Artifact storage (Parquet + JSON)
  schemas.py               Runtime schema validation
  artifact_schemas/        JSON schemas for artifacts (bundled as package data)
  plugins/
    manifest.yaml          Default plugin profiles (bundled as package data)
    builtins.py            Plugin registration map
    strategies/            Strategy plugins (compute target weights)
    pipeline/              Pipeline plugins (orchestrate full runs)
    datasources/           Data plugins (OHLCV, market cap, funding rates)
    broker/                Broker plugins (paper + live execution)
    rebalancing/           Rebalancing plugins (weights → orders)
    risk/                  Risk plugins (pre-trade validation)
    publisher/             Publisher plugins (notifications)
    backtesting/           Backtest engines (vectorbt, rsims)
cookbook/
  configs/                 Example YAML pipeline configs (research, trading, paper, live)
  scripts/                 Runnable example scripts (quickstart, custom plugin, artifact inspection)
```

## Key commands

```bash
uv run quantbox plugins list               # list all plugins
uv run quantbox plugins list --json         # JSON output
uv run quantbox plugins info --name <id>    # plugin details
uv run quantbox validate -c <config>        # validate config
uv run quantbox run -c <config>             # run pipeline
uv run quantbox run --dry-run -c <config>   # dry run
uv run pytest -q                            # run tests
```

## Plugin architecture

- All plugins implement Protocols defined in `contracts.py`
- Plugins are `@dataclass` classes with a class-level `meta = PluginMeta(...)` attribute
- Registration: `plugins/builtins.py` builds `{meta.name: class}` dict
- Discovery: `registry.py:PluginRegistry.discover()` merges builtins + entry points
- Runner: `runner.py:run_from_config()` instantiates via `params_init`, calls `pipeline.run()`

## Plugin types and key methods

| Type | Protocol | Key method |
|---|---|---|
| Pipeline | `PipelinePlugin` | `run(mode, asof, params, data, store, broker, risk)` |
| Strategy | `StrategyPlugin` | `run(data, params)` → dict with `"weights"` (date × symbol) |
| Data | `DataPlugin` | `load_market_data(universe, asof, params) → Dict[str, DataFrame]` |
| Broker | `BrokerPlugin` | `execute_rebalancing(weights)`, `describe()` |
| Rebalancing | `RebalancingPlugin` | `rebalance(targets, positions, params)` |
| Risk | `RiskPlugin` | `check_targets(targets, params)`, `check_orders(orders, params)` |
| Publisher | `PublisherPlugin` | `publish(result, params)` |

## Data format

DataPlugin returns `Dict[str, pd.DataFrame]` of **wide-format** DataFrames:
- Index: date
- Columns: symbol names
- Keys:
  - `"prices"` (required) — close prices
  - `"volume"` — quote-currency dollar volume
  - `"high"` / `"low"` — daily high/low (needed for ATR-based strategies)
  - `"market_cap"` — monthly mcap snapshots (typically forward-filled to daily)
  - `"funding_rates"` — perp funding (futures datasets)
  - `"eligibility_mask"` — boolean wide DataFrame; top-N-by-mcap gate that
    strategies can consume via `data.get("eligibility_mask")` for PIT-correct
    daily universe rotation

  Optional keys are `setdefault`-ed to empty DataFrames by the engine, so
  strategies can always `data.get(key)` safely. Data plugins may emit
  additional non-canonical keys, but only the list above is guaranteed.

## Config structure

```yaml
run:
  mode: backtest|paper|live
  asof: "2026-02-06"
  pipeline: "pipeline.name.v1"

plugins:
  pipeline:
    name: "trade.full_pipeline.v1"
    params: { ... }
  strategies:
    - name: "strategy.crypto_trend.v1"
      weight: 1.0
      params: { ... }
  data:
    name: "binance.live_data.v1"
    params_init: { ... }
  broker:
    name: "hyperliquid.perps.v1"
    params_init: { ... }
  rebalancing:
    name: "rebalancing.futures.v1"
    params: { ... }
  risk:
    - name: "risk.trading_basic.v1"
      params: { ... }
```

## Environment variables

| Variable | Required for | Default |
|---|---|---|
| `API_KEY_BINANCE` | Binance spot/futures brokers | — |
| `API_SECRET_BINANCE` | Binance spot/futures brokers | — |
| `HYPERLIQUID_WALLET` | Hyperliquid perps broker | — |
| `HYPERLIQUID_PRIVATE_KEY` | Hyperliquid perps broker | — |
| `TELEGRAM_TOKEN` | Telegram publisher | — |
| `TELEGRAM_CHAT_ID` | Telegram publisher | — |
| `QUANTBOX_MANIFEST` | Custom manifest path | `plugins/manifest.yaml` |

None are needed for backtesting or paper trading with simulated brokers.
Copy `.env.example` to `.env` and fill in only what you need.

## Error handling

Quantbox uses custom exceptions (see `quantbox.exceptions`):

| Exception | When | Recovery |
|---|---|---|
| `ConfigValidationError` | YAML config fails validation | Check `.findings` list for details |
| `PluginNotFoundError` | Plugin name not in registry | Check `.available` for valid names |
| `PluginLoadError` | Entry point import failed | Check dependencies (`uv sync --extra full`) |
| `DataLoadError` | Data plugin can't fetch data | Check API keys, network, date range |
| `BrokerExecutionError` | Order placement failed | Check broker credentials and balances |

## Development rules

- Use `uv` as package manager, `uv run` to execute
- Don't use `requests` in core — use `urllib.request` or `httpx`
- `meta` is a class attribute, not an instance attribute
- Prefer additive changes and new plugin versions over breaking changes
- Don't rename existing entry-point IDs
- Add tests for new plugins or core behavior
- **Pipeline smoke is required for plugin PRs.** A plugin that passes
  unit tests but isn't registered in `builtins.py` / `manifest.yaml` is a
  silent production break. Run `uv run pytest -m pipeline_smoke` before
  marking a PR ready for review (CI runs it as a separate job too).
- See [`docs/architecture/principles.md`](docs/architecture/principles.md) for LLM-specific guidelines + anti-patterns

**For any architectural change, the rules in [`docs/architecture/principles.md`](docs/architecture/principles.md) take precedence over this file.** Anti-patterns to refuse, decision rules for new features, and the layer-choice doctrine all live there.

## Git workflow

| Branch | Purpose | Merge target |
|---|---|---|
| `main` | Release-only — **protected** | — (the tag + deploy target) |
| `dev` | Integration branch | `main`, at release time |
| `feat/{slug}` | One change | `dev` via PR |

Branch a `feat/{slug}` off `dev`, commit there, and open the PR to **`dev`**
(`gh pr create --base dev`) — never a feature straight to `main`. Release flow is
[Shipping cycle](#shipping-cycle-two-stage-pr-mirrors-dm-evo) below.

**Never commit or push directly to `main`.** Every change reaches it through a
PR; trivial fixes still get a short-lived branch. This is the deterministic
stand-in for GitHub branch protection, which this repo's plan does not offer.
**Land work on `dev` through a PR too** — that is the convention, not a guard
refusal: `dev` is deliberately unguarded locally (`integration_branch: null`),
so the cost of a direct push is a convention broken, not a hook fired.

Two guard layers enforce the `main` rule, both shipped by the **qute-essentials
plugin** — neither is a file this repo maintains. `.claude/git-guard.json` is the
opt-in: its *presence* arms both, and it carries only what differs from the house
defaults. `main` protected is the default, so the file names just
`integration_branch: null` (the deliberate deviation — house default would detect
`dev` and guard it) and `release_tool`.

- **`pre-push`** is the layer that holds. Git hands it the resolved refs, so it
  covers humans, scripts and agents alike. Install/verify it with
  `python3 "${CLAUDE_PLUGIN_ROOT}/scripts/install_pre_push_guard.py" --repo . --check`.
  Override one push with **`CLAUDE_GUARD_BRANCH_PUSH=0 git push …`**, which skips
  this check only; `git push --no-verify` also works but drops every other
  pre-push hook with it.
- **The `git-workflow` `PreToolUse` hook** is the speed bump in front of it: it
  sees Claude tool calls only, but it catches `git commit` (which never reaches
  `pre-push`) and explains the route before the command runs. Turn it off with
  **`/guard git-workflow off`**; that disarms only this layer.

A `.claude/hooks/git-workflow-guard.py` checked into this repo is a stale fork of
the plugin's guard — deleted in TOM-354, and it belongs deleted, not maintained.
(`GIT_GUARD_DISABLE=1`, which older revisions of this file advertised, was only
ever read by that fork and does nothing now.)

Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `test:`) —
the prefixes drive the semver bump. Never `--no-verify`, never force-push `main`,
and commit only when asked.

## Multi-repo setup

| Repo | Purpose | Branch/tag |
|---|---|---|
| quantbox (this) | Library | `dev` for development, `main` for releases |
| quantbox-live | Production trading | Pins to tags on `main` (e.g. `@v0.1.0`) |
| quantbox-lab | Research/backtesting | Pins `quantbox@main` (no `dev` branch) |

### Shipping cycle (two-stage PR, mirrors dm-evo)

**This section is the single statement of the release policy for this repo.**
Every other file that needs it links here rather than restating it — a restated
policy drifts, and this one did (three flows across seven files, TOM-354). The
machine-readable half lives in [`conductor.yml`](conductor.yml) (`release.branch`,
`baseBranch`) and must agree with what follows.

1. **Feature work → PR to `dev`** (`gh pr create --base dev`). `ci.yml` runs on
   `dev` PRs; the independent-reviewer gate runs on **`main`** PRs only — the
   expensive pass belongs at the merge gate, so a `dev` PR gets CI and nothing
   else. Merge feature PRs into `dev`, never straight to `main`.
2. **Release → `/ship` on `dev` (bump), then `/ship --tag` on `main` (tag).**

   a. On `dev`, run **`/ship`**. It bumps `pyproject.toml` + `CHANGELOG.md`,
      refreshes `uv.lock` into the same commit, and stops there — **no tag**.
      Never hand-roll `cz bump`: `/ship` is the single writer of a version or a
      tag. (`annotated_tag = true` in `[tool.commitizen]` is the second line of
      defence — a lightweight tag is one `git push --follow-tags` silently
      declines to push, so it stays local until something downstream cannot
      resolve it.)
   b. Push `dev`, then PR `dev` → `main` and merge it. **Squash or merge commit,
      either is fine** — see below.
   c. On `main`, after the merge, run **`/ship --tag`**. It asserts the tree is
      clean, the remote is reachable, the local branch matches its remote, and
      the version at the tip is the one being tagged; then it creates the
      annotated `vX.Y.Z` tag **and pushes it**.

   **Why the merge method no longer matters.** It used to: the tag was cut on
   `dev` *before* the merge, so a squash — which rewrites the bump into a new
   commit on `main` — left the tagged commit outside `main`'s ancestry, naming
   code `main` did not contain while `quantbox-live` pinned that tag. That is
   how `v0.4.0` was cut on 2026-07-28 missing a fix and had to be re-cut as
   `v0.4.1`. The old rule against `--squash` was the workaround. The fix
   replaced it: the tag is now created on `main` *after* the merge (qute-essentials
   v3.6.0, TOM-349), so it names a commit `main` contains **by construction**,
   whichever way the PR landed. `.github/workflows/release-tag-guard.yml` asserts
   exactly that on every pushed `v*` tag. **Do not reinstate a merge-commit
   mandate** — it would be ceremony guarding a hole that is already closed.

   Then bump the `quantbox @ …@vX.Y.Z` pin in `quantbox-live/pyproject.toml` and
   redeploy (the `sudo -u prod` step). Verify the tag contains what you expect
   before pinning to it — `git show vX.Y.Z:<file>`.

`main` is release-only; `dev` is the integration branch. Do NOT PR features to
`main` (the drift we corrected 2026-07-06 — dev had gone stale while everything
landed on main). `quantbox-lab` pins `quantbox@main`; `quantbox-live` pins a `main` tag.
