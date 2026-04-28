# Autoresearch

LLM-driven continuous improvement loops over QuantBox strategies. Propose → run → evaluate → learn → iterate, with hard budget bounds and statistical gates that prevent the loop from p-hacking its way to false alpha.

This doc defines the **architecture**. For step-by-step usage see [playbooks/run-an-autoresearch-loop.md](../playbooks/run-an-autoresearch-loop.md).

---

## What this is for

You have a strategy (yours, or a client's). You want to keep improving it without hand-tuning every parameter. Autoresearch is the loop that does this:

1. Reads what's already been tried (EXPERIMENTS log).
2. Proposes the next variant — LLM-driven, algorithmic, or both.
3. Runs the variant through the full QuantBox pipeline (PIT-correct, schema-validated).
4. Evaluates against a statistical gate (walk-forward, deflated Sharpe, drawdown limits).
5. Updates the research log + LLM-summarized findings.
6. Decides: continue, halt, branch.
7. On exit: surfaces the best candidate to a human for promotion.

The loop is **autonomous within budget** and **human-gated for promotion**. It will never auto-promote to production. That is non-negotiable.

---

## What autoresearch is NOT

- Not a way to skip walk-forward testing — autoresearch *enforces* it.
- Not a substitute for human methodology design — the loop optimizes within a search space *you* defined.
- Not magic alpha generation — without statistical gates it will reliably p-hack.
- Not a new plugin runtime — it composes existing L4 pipelines.
- Not an LLM agent that writes plugin code — that's `quantbox-strategy-author`. Autoresearch *tunes* existing plugins (or composes their parameters into a search space).

---

## Five components, mapped to QuantBox

| Component | Maps to | Status |
|---|---|---|
| Experiment runner | `run_from_config` (L4) | exists |
| Variant generator | `VariantProposerPlugin` + adapters | new — see below |
| Evaluation gate | `ValidationPlugin` + `RiskPlugin` + autoresearch-specific gates | mostly exists |
| Research memory | `EXPERIMENTS.jsonl` (machine) + `EXPERIMENTS.md` (human) + `findings.md` (LLM summary) | jsonl is new |
| Loop driver | `AutoResearchDriver` (L4 driver, not in runner) | new |

The driver is **L4** — it sits *alongside* `run_from_config`, not above it. Each iteration is a normal pipeline run; the driver decides what runs next. See [ADR-0003](../adr/0003-autoresearch-as-driver-not-runtime.md) for why.

---

## VariantProposerPlugin (new Protocol)

The one new plugin type. Generates the next variant given the run history.

```python
from typing import Protocol
import pandas as pd

class VariantProposerPlugin(Protocol):
    meta: PluginMeta

    def propose(
        self,
        baseline: dict,                 # the baseline config
        history: pd.DataFrame,          # prior trials: params + metrics + decision
        search_space: dict,             # bounds per param
        params: dict,                   # proposer-specific params
    ) -> dict:                          # next variant config (delta from baseline)
        ...
```

Built-in proposers:

| Proposer | Adapter | When to use |
|---|---|---|
| `proposer.optuna.tpe.v1` | `quantbox.adapters.optuna` | Pure algorithmic, no LLM cost |
| `proposer.optuna.bayesian.v1` | `quantbox.adapters.optuna` | Sample-efficient on continuous params |
| `proposer.llm.anthropic.v1` | `quantbox.adapters.anthropic` | Reasoning across qualitative variants |
| `proposer.hybrid.v1` | both | LLM for branch decisions, Bayesian within branches |
| `proposer.random.v1` | none | Diversity injection (escape local optima) |

A hybrid proposer can wrap multiple sub-proposers and route based on heuristics (e.g., LLM proposes 1-in-N variants for diversity; Bayesian handles the rest).

---

## AutoResearchDriver

Lives at `quantbox.autoresearch.driver`. Top-level API:

```python
from quantbox.autoresearch import AutoResearchDriver, AutoResearchConfig

driver = AutoResearchDriver.from_config("configs/clients/X/autoresearch.yaml")
report = driver.run()
```

Exposes the same shape at L5:

```bash
quantbox autoresearch run -c configs/clients/X/autoresearch.yaml
quantbox autoresearch tick -c configs/clients/X/autoresearch.yaml   # one iteration
quantbox autoresearch status -c configs/clients/X/autoresearch.yaml # current state
```

`tick` is the cron-friendly mode — does N trials within tight budget, persists state, exits.

### Internal flow per iteration

```
1. Load EXPERIMENTS.jsonl
2. Load findings.md (LLM-readable summary)
3. Cap recent history to last N (keeps proposer context bounded)
4. Proposer.propose(baseline, history, search_space, params) → next variant
5. Materialize variant as a quantbox config
6. quantbox.validate(config) — refuses if schema/PIT invalid
7. run_from_config(config) → RunResult
8. Evaluation gate:
   - walk-forward sharpe ≥ threshold
   - deflated sharpe ≥ threshold
   - drawdown ≥ -threshold
   - constraints from goal section
9. Append to EXPERIMENTS.jsonl + EXPERIMENTS.md
10. (Periodically) LLM updates findings.md from recent trials
11. Cost tracker: deduct trial cost; halt if any budget breach
12. Convergence check: if best metric stagnant for K trials, halt
13. Return ResultRecord
```

Each iteration is fully reproducible — uses the standard ArtifactStore, run_id, lineage.

---

## Configuration shape

```yaml
autoresearch:
  baseline: configs/clients/X/strategy.yaml      # starting point
  search_space:                                  # what's tunable
    params.lookback_days: { range: [30, 252], step: 5 }
    params.vol_target:    { range: [0.08, 0.25], step: 0.01 }
    strategies[0].name:   { choices: [strategy.crypto_trend.v1,
                                       strategy.cross_asset_momentum.v1] }

  budget:
    max_trials: 50
    max_wall_clock: 4h
    max_llm_cost_usd: 25
    max_compute_cost_usd: 5
    halt_on_first_breach: true

  goal:
    optimize: sharpe_oos
    direction: maximize
    constraints:
      max_drawdown: -0.20
      min_trades_per_year: 12
      deflated_sharpe_min: 0.3

  proposer:
    name: proposer.hybrid.v1
    params:
      llm: anthropic.claude-sonnet-4
      search: optuna.tpe
      diversity_injection: 0.1                   # 10% random variants

  evaluation:
    walk_forward:
      splits: 5
      embargo_days: 5
    bootstrap_ci: 0.95

  memory:
    experiments_jsonl: research/clients/X/EXPERIMENTS.jsonl
    experiments_md:    research/clients/X/EXPERIMENTS.md
    findings_md:       research/clients/X/findings.md
    cap_recent_for_proposer: 20

  termination:
    converged_tolerance: 0.01
    consecutive_no_improve: 5

  delivery:
    discord_channel: "#client-X"
    promotion_candidate_min_lift: 0.15           # only surface if ≥15% better
    auto_summary_every_n_trials: 10
```

---

## Memory layout

### `EXPERIMENTS.jsonl` — machine-readable, append-only

One line per trial. The source of truth.

```json
{"trial_id": "T0042", "ts": "2026-04-25T14:33:21Z",
 "variant_params": {"lookback_days": 90, "vol_target": 0.18},
 "metrics": {"sharpe_oos": 1.23, "deflated_sharpe": 0.42, "drawdown": -0.16},
 "decision": "kept", "rank_among_history": 3,
 "run_id": "ar_T0042_2026-04-25T14-33", "cost_usd": 0.42,
 "lineage": {...}}
```

### `EXPERIMENTS.md` — human-readable narrative

Append-only. Per-trial entries with the same data plus a brief LLM-generated rationale.

### `findings.md` — LLM-summarized cumulative knowledge

Periodically regenerated by an LLM call summarizing recent trials. Examples:
- "Higher lookback (>120) helps in choppy regimes; hurts in trending"
- "vol_target above 0.20 consistently breaches drawdown gate"
- "crypto_trend.v1 dominates in this universe; cross_asset_momentum.v1 abandoned after 8 trials"

`findings.md` is **advisory**, not authoritative. The numbers live in `EXPERIMENTS.jsonl`. The LLM cannot fabricate metrics because it doesn't write them.

---

## Budget and cost tracking

A separate concern from the runner. `AutoResearchDriver` carries a `BudgetTracker`:

```python
@dataclass
class BudgetTracker:
    max_trials: int
    max_wall_clock_s: float
    max_llm_cost_usd: float
    max_compute_cost_usd: float

    used_trials: int = 0
    started_at: float = field(default_factory=time.time)
    used_llm_cost_usd: float = 0.0
    used_compute_cost_usd: float = 0.0

    def consume(self, *, trials: int = 0, llm_usd: float = 0,
                compute_usd: float = 0) -> None: ...

    def breach(self) -> str | None:
        """Returns a non-empty reason if any limit is breached."""
```

The driver halts on the first breach. Discord summary posts the breach reason.

For continuous mode (cron-driven), each tick has its *own* budget — the global per-month budget is enforced by a separate registry.

---

## Statistical gates (mandatory)

Without these, the loop is dangerous. They are not optional.

| Gate | Default | Why |
|---|---|---|
| Walk-forward splits ≥ 5 | required | One in-sample number is meaningless |
| Embargo period | 5 days | Prevent leakage between train/test |
| Deflated Sharpe (López de Prado) | required | Adjusts for multiple-testing inflation |
| Bootstrap CI 95% | required | Quantifies metric uncertainty |
| Drawdown constraint | per-config | Hard cap; rejects on breach |
| Min trade count | per-config | Reject methodologies that trade too rarely to evaluate |

The driver refuses to start if `evaluation:` is missing any of these. No "skip for speed" flag.

---

## Continuous mode (Iris + agent-cron)

For background continuous improvement, the driver exposes `tick`:

```bash
quantbox autoresearch tick -c configs/clients/X/autoresearch.yaml --max-trials 5
```

Wired into Iris's schedule via `agent-cron`:

```bash
# Daily, 06:30
agent-cron iris 'autoresearch tick: client-X strategy. Budget: 5 trials, $5 LLM, 30 min.
                 Post summary to #client-X if anything beats baseline by ≥15%.'
```

- Each tick is one iteration batch within tight budget.
- Loop persists across runs (EXPERIMENTS.jsonl is durable memory).
- Iris posts to `#client-X` only when meaningful (don't spam).
- Human-in-the-loop checkpoint required before any production promotion — the driver never flips `meta.status` past `research`.
- Cost capped per tick *and* per month.

This composes with existing scheduling — no new orchestration system.

---

## Lifecycle integration

When the loop converges or budget exhausts, the best variant is a **promotion candidate**:

1. Driver writes a candidate config under `research/clients/X/candidates/{trial_id}.yaml`.
2. `meta.status` of the proposed plugin variant stays `research`.
3. Discord summary includes a "Promote?" prompt.
4. Human runs `/promote-lock` (existing flow) on the candidate.
5. From there, normal lifecycle: research → locked → production.

The autoresearch loop **does not bypass the lifecycle**. It produces candidates; humans walk them through gates.

---

## What this lets you do

For each client:

```
research/clients/{client}/
├── strategy.yaml                    # current production methodology
├── autoresearch.yaml                # search space + budget + gates
├── EXPERIMENTS.jsonl                # all trials ever run
├── EXPERIMENTS.md                   # human view
├── findings.md                      # LLM cumulative summary
└── candidates/                      # promotion-ready variants
    ├── T0042.yaml
    └── ...
```

Per-client cron tick adds N trials per day within budget. Discord posts to `#client-{name}` when something interesting emerges. You review and promote.

This is what "improve client strategies continuously" looks like in practice.

---

## Anti-patterns to refuse

| Anti-pattern | Fix |
|---|---|
| Auto-promoting to production | Hard NO. Driver maxes at `research`; humans flip lock. |
| Skipping walk-forward "for speed" | Driver refuses to start without the gate. |
| Using vanilla Sharpe without deflation | Refused. Multiple-testing inflation is real. |
| Letting the LLM proposer write metrics into findings.md | LLM only writes prose; numbers come from EXPERIMENTS.jsonl. |
| Running unbounded budget | Driver requires explicit budget; no defaults that allow runaway. |
| Mixing autoresearch state across clients | One config = one client = one EXPERIMENTS.jsonl. |
| Treating findings.md as authoritative | It's advisory. Source of truth is the JSONL. |
| Reimplementing Optuna's search algorithms | Adapter only. See [adapters.md](adapters.md). |

---

## Composability with the rest of the architecture

This integrates without violating any principle:

- **Composer not competitor** — Optuna does search; Anthropic SDK does LLM calls; QuantBox does the loop + conventions + skill API. All adapters.
- **Lowest viable abstraction** — Driver is L4. Casual users don't see autoresearch unless they invoke it. Plain `run_from_config` is unaffected.
- **Layered API** — `quantbox.autoresearch` is itself an L1 namespace; you can call `from quantbox.autoresearch import propose` directly.
- **Owned conventions** — `EXPERIMENTS.jsonl` format, the `VariantProposerPlugin` Protocol, the budget tracker, the `tick` cron mode are quantbox-owned. The hard work is somebody else's library.
- **Adapter not reimplementation** — search algos: optuna; LLM: anthropic/openai SDKs; experiment tracking: optionally MLflow.

---

## See also

- [ADR-0003](../adr/0003-autoresearch-as-driver-not-runtime.md) — design decision for L4 driver placement.
- [api-layers.md](api-layers.md) — driver lives at L4, not embedded in runner.
- [plugin-authoring.md](plugin-authoring.md) — `VariantProposerPlugin` is a new plugin type.
- [lifecycle.md](lifecycle.md) — autoresearch produces candidates; promotion path is normal.
- [adapters.md](adapters.md) — Optuna, Anthropic, OpenAI as adapters.
- [playbooks/run-an-autoresearch-loop.md](../playbooks/run-an-autoresearch-loop.md) — step-by-step.
