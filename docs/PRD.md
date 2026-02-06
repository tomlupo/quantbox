# PRD — QuantBox: Plugin-Based Quant Research & Trading

**Goal:** Build a modular, reproducible, automation-first system that turns research (ideas → allocations) into execution (signals → orders → fills) with minimal manual work.

**Designed for:**

- Python, local-first (Parquet + DuckDB)
- Deterministic math; LLM only for reasoning and analysis
- Easy research → paper/live transition via the trading bridge

---

## 1. Overview

### Problem

Typical quant workflow pain:

- Messy notebooks and hidden assumptions
- No reproducibility or clear lineage
- Manual runs and weak monitoring

**Result:** Slow iteration, hard debugging, fragile production.

### Solution

QuantBox is a **plugin-based Quant Research & Trading OS** with:

- **Structured runs** — config-driven; one command per pipeline
- **Tracked artifacts** — Parquet + JSON schemas; run manifest and events per run
- **Modular plugins** — data, pipeline, broker; extend via manifest and optional entry points
- **Research → trading bridge** — allocations from research → targets/orders/fills via a dedicated pipeline
- **Approval gate** — optional human sign-off before paper/live execution
- **LLM-friendly** — `validate`, `--dry-run`, JSON outputs, and artifact schemas for agents

---

## 2. Objectives

### Primary

- Reproducible runs from config (and seed where applicable)
- One-command research and one-command trading (with explicit chaining)
- Clear lineage: data → pipeline artifacts → (optional) broker execution
- Easy comparison of runs via artifact layout and run manifests

### Secondary

- Faster research iteration via plugin reuse
- Fewer silent data bugs via schema validation
- Interpretable decisions (approval gate, run manifests)
- Low ops overhead (local-first; optional brokers)

---

## 3. Non-Goals

**Not building:**

- HFT or real-time tick infrastructure
- Custom exchange gateways (use broker plugins: IBKR, Binance, sim)
- Black-box LLM trading — LLM for analysis, explanations, and tool use only

**Math and signal generation stay deterministic.**

---

## 4. Users

### Researcher (you)

- Runs research pipelines (e.g. fund selection)
- Adds or adjusts pipeline/data plugins
- Inspects artifacts and run history

### System / automation

- Runs daily or scheduled jobs (runner + configs)
- Produces allocations, then (optionally) runs trading pipeline
- Logs metrics and artifacts to `./artifacts/<run_id>/`

### AI assistant

- Uses `quantbox validate`, `quantbox run --dry-run`, `quantbox plugins list --json`
- Explains results, proposes config changes, audits anomalies
- No direct trading authority; human approval required for execution

---

## 5. System Architecture

### High-level

```
Data (Parquet) → DuckDB (data plugin) → Pipeline plugin → Artifacts (allocations / targets / orders / fills)
                                                                        ↓
                                              Optional: Broker plugin (sim / IBKR / Binance) → Fills, portfolio_daily
```

- **Runner** loads manifest, resolves plugins, runs pipeline (and broker when in paper/live).
- **Artifacts** are written under `artifacts/<run_id>/` with schemas in `schemas/*.schema.json`.
- **Approval gate** (when enabled) requires an approval file before broker execution.

---

## 6. Core Stack

| Role            | Technology        |
|-----------------|-------------------|
| Analytics/query | DuckDB over Parquet |
| Orchestration   | Config-driven runner (no Prefect/MLflow in core) |
| Plugins         | In-repo builtins + manifest; optional external entry points |
| LLM             | Analysis and tool use only; no direct trading |

---

## 7. Functional Requirements

### 7.1 Data layer

**Requirements:**

- Parquet-backed storage; fast local query
- Point-in-time correctness via `asof` in run config
- Reproducible snapshots (same data path + asof → same inputs)

**Current implementation:** `duckdb_parquet` data plugin (DuckDB over Parquet). Supports `prices_path`, optional `fx_path`.

**Tables / artifacts (examples):** `prices.parquet`, optional `fx.parquet` (date, pair, rate).

**Acceptance:**

- Load typical EOD universe in seconds
- No pandas loops required in core; vectorized/query-based

---

### 7.2 Pipeline layer

**Requirements:**

- Config-driven; pipeline chosen by name in run config
- Inputs/outputs defined by artifact contracts (see `contracts/ARTIFACTS.md` and `schemas/`)
- Reproducible from config + data + asof

**Current pipelines:**

- **Research:** `fund_selection.simple.v1` — universe + prices → scores/rankings → allocations
- **Trading bridge:** `trade.allocations_to_orders.v1` — allocations + prices + optional instrument_map/FX → targets, orders, fills, portfolio_daily

**Acceptance:**

- Rerun with same config and data → identical artifact set
- Pipeline chaining is explicit (research run → trading config points at allocations path or `allocations_ref: "latest:fund_selection.simple.v1"`)

---

### 7.3 Broker layer

**Requirements:**

- Pluggable; broker selected in config (paper or live)
- Simulator for research/backtest; stubs/live for IBKR and Binance

**Current brokers:**

- `sim.paper.v1` — paper simulator
- `ibkr.paper.stub.v1` / `ibkr.live.v1`
- `binance.paper.stub.v1` / `binance.live.v1`

**Acceptance:**

- Swap broker via config only; no code change in core

---

### 7.4 Artifact contracts

**Requirements:**

- All artifacts have a schema in `schemas/*.schema.json`
- Parquet where applicable; contracts documented in `contracts/ARTIFACTS.md`

**Current v1 artifacts:**

- Research: `universe`, `prices`, `scores`, `rankings`, `allocations`
- Trading: `targets`, `orders`, `fills`, `portfolio_daily`

**Acceptance:**

- `quantbox validate` and schema checks support correct artifact layout
- LLM/agents can discover schemas for validation and tool use

---

### 7.5 Approval gate

**Requirements:**

- When `approval_required: true`, pipeline writes orders but does not execute until approval file is present
- Approval file matches `orders_digest` (e.g. `./approvals/<orders_digest>.json`)

**Acceptance:**

- Without approval, `fills.parquet` remains empty in paper/live when approval is required
- With approval, rerun executes and fills are written

---

### 7.6 Automation and run tracking

**Requirements:**

- Single entry point: `quantbox run -c <config>`
- Optional `--dry-run` for validation without writing artifacts
- Per-run: `run_manifest.json`, `events.jsonl`, and artifact directory

**Acceptance:**

- Fully unattended possible (e.g. cron: research then trading with `allocations_ref: "latest:..."` and approval gate or read-only broker)

---

### 7.7 AI / LLM analyst

**Scope:**

- **Allowed:** Explain results, compare runs, suggest configs, summarize drawdowns, detect anomalies, call `quantbox` CLI with validate/dry-run
- **Forbidden:** Compute returns or signals directly; modify trades; bypass approval

**Acceptance:**

- Zero direct trading authority; execution only via runner and (when enabled) approval gate

---

## 8. Non-Functional Requirements

| Area            | Target |
|-----------------|--------|
| Performance     | Local laptop friendly; DuckDB for fast EOD analytics |
| Reliability     | Deterministic runs; versioned/config-driven data paths |
| Reproducibility | Config + asof + data path → identical artifacts |
| Maintainability | Modular plugins; no notebook dependency for production path |

---

## 9. Repository structure

```
quantbox/
├── configs/                    # Run configs (research, trading, broker-specific)
│   ├── instruments.yaml        # Instrument map (multiplier, lot size, FX, etc.)
│   ├── run_fund_selection.yaml
│   ├── run_trade_from_allocations.yaml
│   └── run_trade_from_allocations_*_live|paper.yaml
├── contracts/
│   └── ARTIFACTS.md            # Artifact contracts and schema references
├── docs/
│   ├── README.md               # This index
│   ├── PRD.md                  # This document
│   ├── specs/                  # Feature/spec docs (active development)
│   ├── guides/                 # How-to: chaining, trading bridge, approval
│   ├── reference/              # LLM ops, broker secrets
│   └── architecture/          # Design/contracts when added
├── packages/
│   └── quantbox-core/          # Core: registry, runner, store, CLI, builtin plugins
│       └── src/quantbox/
│           ├── plugins/        # data, pipeline, broker builtins
│           ├── cli.py, runner.py, store.py, registry.py, contracts.py, ...
│           └── ...
├── plugins/
│   ├── manifest.yaml           # Plugin list and profiles (research, trading)
│   └── manifest.schema.json
├── recipes/                    # How-to: add broker plugin, add pipeline plugin
├── schemas/                    # JSON schemas for artifacts
│   ├── allocations.schema.json
│   ├── prices.schema.json
│   ├── orders.schema.json
│   ├── fills.schema.json
│   ├── targets.schema.json
│   ├── portfolio_daily.schema.json
│   ├── rankings.schema.json
│   ├── scores.schema.json
│   └── universe.schema.json
├── scripts/
│   ├── make_sample_data.py     # Generate sample prices (and optional FX)
│   └── approve_orders.py       # Create approval file for a run
├── tests/
└── data/                       # Local data (e.g. data/curated/prices.parquet)
```

Artifacts are written to `./artifacts/<run_id>/` (configurable via `artifacts.root`).

---

## 10. MVP scope (current)

### In scope

- DuckDB/Parquet data plugin
- Fund-selection pipeline (research → allocations)
- Trading-bridge pipeline (allocations → targets/orders/fills/portfolio_daily)
- Sim + IBKR/Binance broker plugins (paper stubs and live)
- Plugin manifest and profiles
- Run manifest + events per run
- Approval gate
- Instrument map and optional FX
- `quantbox run`, `quantbox validate`, `quantbox plugins list` (including `--json`)

### Later / optional

- Prefect or other orchestrator for scheduling
- MLflow or similar for experiment tracking and leaderboards
- Advanced risk and optimization plugins
- Dashboards, Telegram bot, web UI
- Multi-agent orchestration

---

## 11. Success metrics

- Backtest/research run time and iteration speed improved vs ad-hoc notebooks
- Zero manual steps for recurring research → trading flow (with optional approval)
- Reproducible results from config and data
- Fewer data bugs via schemas and validate/dry-run
- LLM/agents can safely introspect and suggest changes without executing trades

---

## 12. Optional extensions

Future possibilities:

- Live trading connectors (additional brokers)
- Web dashboard for runs and artifacts
- Strategy marketplace or shared pipeline plugins
- Reinforcement learning or signal ensemble layers
- Tighter integration with experiment trackers (e.g. MLflow) and orchestrators (e.g. Prefect)

---

## References

- **Artifacts:** `contracts/ARTIFACTS.md`
- **Approval:** [guides/approval-gate.md](guides/approval-gate.md)
- **Trading bridge:** [guides/trading-bridge.md](guides/trading-bridge.md)
- **LLM usage:** [reference/llm-operations.md](reference/llm-operations.md), `CONTRIBUTING_LLM.md`
- **Chaining:** [guides/pipeline-chaining.md](guides/pipeline-chaining.md)
