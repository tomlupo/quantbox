# CLAUDE.md — Agent onboarding guide

> **Modifying QuantBox?** Read [`docs/architecture/principles.md`](docs/architecture/principles.md) and [`docs/architecture/api-layers.md`](docs/architecture/api-layers.md) first. Every architectural decision downstream is shaped by them. This file describes the *current* code; the architecture docs describe the *target design and the rules*. When they conflict, architecture wins.

## What is QuantBox?

A **template-driven SDK with adapters** for quant research and production. Three things, in order:

1. **Conventions** — data layouts, run-artifact shape, lifecycle states, skill API. The owned moat.
2. **Adapters** — thin wrappers around best-of-breed external libraries (vectorbt, riskfolio, ...). The wheel does the wheel's work. A core adapter is added only when ≥2 consumers need the same bridge; single-consumer libraries (mlflow, dvc) are imported directly in the downstream repo.
3. **Skills + templates** — LLM-facing interface and project bootstrap, coupled to the SDK in this repo.

The plugin runtime (`run_from_config`, CLI) is *one* of multiple entry points — see the [layered API](docs/architecture/api-layers.md) (L0–L5). Casual use defaults to L0/L1 (re-exports + convenience helpers). YAML pipelines are L4. Production is L5 with `--strict`.

QuantBox is a **composing framework** — owned and opinionated, but composing external libraries (vectorbt, MLflow, riskfolio, optionally Qlib) rather than competing with them on their turf. See [ADR-0001](docs/adr/0001-library-not-framework.md).

## Task source: Linear

Linear is the task source (qute-code-kit ADR-0004). The binding — team, project,
and the machine marker skills read — lives in
[`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md) and is not
restated here; a copy in this file said "project quantbox" until 2026-08-18, and
no such project exists.

**GitHub Issues on `tomlupo/quantbox` are issue *records* only** — bugs and tech
debt attached to the code. An issue becomes work when a Linear task references
it; never pull work from the Issues list directly.

## qute runtime

This repo runs the standard qute regime (qute-code-kit ADR-0001..0004). Key skills:

- `/task` + `/repo-status` — honor [`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md) (Linear).
- `/decision` — records ADRs to [`docs/adr/`](docs/adr/) (`NNNN-title.md`).
- `/handoff` + `/pickup` — the continuity pair for pausing/resuming work.
- `/ship` — the release boundary (commitizen). ONE act on `dev`: bump,
  changelog, lockfile, commit and the annotated `vX.Y.Z` tag, pushed, then the
  promotion PR into `main`. See [`## Shipping cycle`](#shipping-cycle)
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
  parquet_io.py            Teardown-safe parquet reads (every pandas read routes here)
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

Branch off `dev`, PR to `dev` (`gh pr create --base dev`). **Never commit or push
directly to `main`** — every change reaches it through a PR. Features never go
straight to `main`: that drift was corrected 2026-07-06, and #152 slipped through
again on 2026-08-18, so it is worth actually checking.

The `main` rule is enforced by two qute-essentials guard layers armed by the
presence of `.claude/git-guard.json` — `pre-push` (the one that holds) and the
`git-workflow` PreToolUse hook. Neither is a file this repo maintains; `/guard`
documents and toggles them. This repo's only deviation is `integration_branch:
null`, so `dev` is deliberately unguarded locally.

Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `test:`) —
the prefixes drive the semver bump. Never `--no-verify`, never force-push `main`,
and commit only when asked.

## Multi-repo setup

| Repo | Purpose | Branch/tag |
|---|---|---|
| quantbox (this) | Library | `dev` for development, `main` for releases |
| quantbox-live | Production trading | Pins tags on `main` (e.g. `@v0.4.2`) |
| quantbox-lab | Research/backtesting | Pins `quantbox@main` (no `dev` branch) |

### Shipping cycle

**This section is the single statement of the release policy.** Every other file
links here rather than restating it — this policy has drifted twice now
(TOM-354; then again across three files by 2026-08-18). `conductor.yml`
(`release.branch`) is the machine-readable half and must agree.

1. **Feature work → PR to `dev`.** `ci.yml` runs on `dev` PRs; the
   independent-reviewer gate runs on **`main`** PRs only.
2. **Release → `/ship` on `dev`.** One act: bump, changelog, `uv.lock`, commit,
   annotated tag, push, promotion PR. There is no second command — `/ship --tag`
   was removed in qute-essentials v9.0.0 and is rejected by name.
3. **Merge the promotion PR with a MERGE COMMIT (`--merge`).** Not squash, not
   rebase. The tag is cut on `dev` *before* the promotion, so a squash rewrites
   the bump into a new sha and strands the tag outside `main`'s ancestry — the
   tag would name code `main` does not contain while `quantbox-live` pins it.
   That is the v0.4.0 incident (2026-07-28). `release-tag-guard.yml` job
   `release-tag-reachable` fires when `main` moves and fails exactly this case.
4. **Then bump the `quantbox@vX.Y.Z` pin in `quantbox-live`** and PR it to that
   repo's `main`. **There is no deploy command** — prod never pulls this repo;
   quantbox-live pins the tag and its cron picks it up (`run_daily.sh` does
   `git merge origin/main` + `uv sync`, 06:00 UTC). Run
   `./scripts/after-release.sh` to check the tag is reachable from `main`, see
   the pin quantbox-live currently declares, and print what remains.
