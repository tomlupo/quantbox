---
name: quantbox-autoresearch
description: Use when the user wants to continuously improve a strategy via LLM-driven research loops. Triggers — "improve this strategy", "tune over time", "autoresearch", "search the parameter space", "make this better automatically", "continuous improvement".
default_layer: L4
escalation_rules:
  - to: L5
    when: "running as a scheduled cron job (use `quantbox autoresearch tick`)"
status: stub
requires_quantbox_min: "0.3.0"
---

# quantbox-autoresearch

> **Status: stub.** The `AutoResearchDriver` is not yet implemented. This skill describes the target shape so the LLM-facing API is locked in alongside the architecture. See [docs/architecture/autoresearch.md](../../docs/architecture/autoresearch.md) and [ADR-0003](../../docs/adr/0003-autoresearch-as-driver-not-runtime.md) for the design.

---

## Pick the layer

| Task shape | Use |
|---|---|
| "One-off improvement loop, run for a few hours" | L4 — `quantbox autoresearch run -c <config>` |
| "Continuous daily/weekly improvement, budget per tick" | L5 — `quantbox autoresearch tick -c <config>` from cron |
| "Test the loop wiring before committing budget" | `quantbox autoresearch run --dry-run --max-trials 1 -c <config>` |

Default: **L4** with a tight first-run budget (~50 trials, ≤$30 LLM cost). Escalate to L5 (cron) only after you've validated the loop config.

---

## What this skill produces

A YAML config (and the supporting `research/clients/{client}/` scaffolding) that defines:

- Baseline strategy to vary from
- Tunable parameters (search space)
- Budget (trials, wall clock, LLM cost, compute cost)
- Goal + statistical gates (walk-forward, deflated Sharpe, drawdown)
- Proposer (Optuna, LLM, or hybrid)
- Memory paths (EXPERIMENTS.jsonl, findings.md)
- Delivery (Discord channel, candidate threshold)

It does NOT produce a new plugin. The loop tunes existing plugins; it doesn't author them. For new plugin code, see `quantbox-strategy-author`.

---

## Fast path — typical config

```yaml
autoresearch:
  baseline: configs/clients/X/strategy.yaml
  search_space:
    params.lookback_days: { range: [30, 252], step: 5 }
    params.vol_target:    { range: [0.08, 0.25], step: 0.01 }

  budget:
    max_trials: 50
    max_wall_clock: 4h
    max_llm_cost_usd: 25
    max_compute_cost_usd: 5

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
      diversity_injection: 0.1

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
    promotion_candidate_min_lift: 0.15
    auto_summary_every_n_trials: 10
```

Validate, dry-run, then run:

```bash
quantbox autoresearch validate -c configs/clients/X/autoresearch.yaml
quantbox autoresearch run --dry-run --max-trials 1 -c configs/clients/X/autoresearch.yaml
quantbox autoresearch run -c configs/clients/X/autoresearch.yaml
```

For continuous mode (cron):

```bash
quantbox autoresearch tick -c configs/clients/X/autoresearch.yaml --max-trials 5
```

---

## Defaults the skill enforces

These are **mandatory**. Refuse to produce a config without them:

| Field | Default | Why |
|---|---|---|
| `evaluation.walk_forward.splits` | ≥ 5 | One in-sample number is meaningless |
| `evaluation.walk_forward.embargo_days` | ≥ 5 | Prevent train/test leakage |
| `evaluation.bootstrap_ci` | 0.95 | Quantify metric uncertainty |
| `goal.constraints.deflated_sharpe_min` | required | Adjust for multiple-testing inflation |
| `goal.constraints.max_drawdown` | required | Hard cap on tail risk |
| `budget.max_trials` | required | Bound the loop |
| `budget.max_wall_clock` | required | Bound time |
| `budget.max_llm_cost_usd` | required | Bound LLM spend |

If the user asks to skip any of these "for speed," refuse and explain why.

---

## When to escalate to authoring

If the user wants to vary something the existing plugins can't express (e.g., "try a regime-detection layer"), this skill is the wrong tool. Hand off to `quantbox-strategy-author` to scaffold the new capability first; *then* come back here to tune it.

The split: this skill **tunes**; `quantbox-strategy-author` **creates**.

---

## Pitfalls to warn about

- **In-sample p-hacking** — without the mandatory gates, the loop will reliably find spurious alpha. Refuse to skip them.
- **Cost runaway** — LLM proposers can rack up bills. The budget is a hard cap, but the user should also know roughly what 50 trials costs (~$30).
- **Concurrent loops on same memory** — one config = one client = one EXPERIMENTS.jsonl. Concurrent writes corrupt state.
- **Auto-promotion** — the loop will NEVER auto-promote to production. The skill must not suggest it. Best the loop produces is a `research`-status candidate.
- **findings.md as gospel** — it's LLM-summarized prose, advisory only. The truth is in EXPERIMENTS.jsonl.

---

## Result delivery

When the loop exits, it produces three artifacts:

1. **Best candidate** at `research/clients/{client}/candidates/{trial_id}.yaml` — a normal quantbox config you can run independently.
2. **Summary report** at `research/clients/{client}/autoresearch-report-{date}.md` — human-readable.
3. **Discord post** in the configured channel — metrics, chart, and a "Promote?" prompt.

The skill's job ends here. Promotion is human-driven via `/promote-lock` (existing flow). See `quantbox-promote`.

---

## See also

- [architecture/autoresearch.md](../../docs/architecture/autoresearch.md) — full design.
- [playbooks/run-an-autoresearch-loop.md](../../docs/playbooks/run-an-autoresearch-loop.md) — step-by-step.
- [ADR-0003](../../docs/adr/0003-autoresearch-as-driver-not-runtime.md) — why driver, not runtime mode.
- `quantbox-strategy-author` — when you need to author a new capability before tuning.
- `quantbox-promote` — what happens after a candidate is selected.
