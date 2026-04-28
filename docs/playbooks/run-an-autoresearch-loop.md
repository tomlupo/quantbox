# Playbook — Run an Autoresearch Loop

LLM-driven continuous improvement of an existing strategy within a defined search space, budget, and statistical gate. Read [architecture/autoresearch.md](../architecture/autoresearch.md) first.

---

## Pre-flight

Confirm:

- [ ] You have a working baseline strategy (existing plugin or config that runs cleanly).
- [ ] Walk-forward evaluation is working in your `ValidationPlugin` setup.
- [ ] You have an Anthropic or OpenAI API key configured (if using LLM proposer).
- [ ] You have a budget you can afford to lose (autoresearch *will* cost money).
- [ ] You're not running this against a live-trading config without `mode: backtest`.

If any are uncertain, fix them before starting. Autoresearch with broken validation is worse than no autoresearch.

---

## Steps

### 1. Pick a baseline

The baseline is the config the loop varies *from*. It must run end-to-end successfully on its own:

```bash
quantbox validate -c configs/clients/X/strategy.yaml
quantbox run -c configs/clients/X/strategy.yaml
```

If this fails, fix it first.

### 2. Define the search space

What's tunable? Think parameter ranges, strategy choices, optional features.

```yaml
search_space:
  params.lookback_days: { range: [30, 252], step: 5 }
  params.vol_target:    { range: [0.08, 0.25], step: 0.01 }
  strategies[0].name:   { choices: [strategy.crypto_trend.v1,
                                     strategy.cross_asset_momentum.v1] }
```

Rules:

- Keep the space tight on the first run — broad sweeps eat budget without learning.
- Don't include parameters you don't understand — the loop will exploit them in nonsensical ways.
- Categorical choices (strategy variants) cost more to explore than continuous params.

### 3. Set the budget

Always required. No defaults are permissive:

```yaml
budget:
  max_trials: 50
  max_wall_clock: 4h
  max_llm_cost_usd: 25
  max_compute_cost_usd: 5
  halt_on_first_breach: true
```

Order-of-magnitude rule: each trial costs ~$0.30–$2 in LLM (with a hybrid proposer) and pennies in compute (for a backtest). 50 trials is typically $30–$50.

### 4. Define the goal and gates

```yaml
goal:
  optimize: sharpe_oos
  direction: maximize
  constraints:
    max_drawdown: -0.20
    min_trades_per_year: 12
    deflated_sharpe_min: 0.3

evaluation:
  walk_forward:
    splits: 5
    embargo_days: 5
  bootstrap_ci: 0.95
```

The evaluation block is **mandatory**. The driver refuses to start without all of: walk-forward splits ≥ 5, embargo period, bootstrap CI, deflated-Sharpe gate. No skip flags.

### 5. Pick a proposer

For first runs:

| Goal | Proposer | Why |
|---|---|---|
| Maximize signal across continuous params | `proposer.optuna.tpe.v1` | Sample-efficient, no LLM cost |
| Explore qualitatively different ideas | `proposer.llm.anthropic.v1` | LLM reasoning across variants |
| Both at once | `proposer.hybrid.v1` | Default — LLM for branches, Bayesian within |

Start with `optuna.tpe` if you're unsure — cheapest, fastest, no LLM bills.

### 6. Configure memory and delivery

```yaml
memory:
  experiments_jsonl: research/clients/X/EXPERIMENTS.jsonl
  experiments_md:    research/clients/X/EXPERIMENTS.md
  findings_md:       research/clients/X/findings.md
  cap_recent_for_proposer: 20

delivery:
  discord_channel: "#client-X"
  promotion_candidate_min_lift: 0.15
  auto_summary_every_n_trials: 10
```

The `min_lift` filter prevents Discord spam — the loop only surfaces candidates that meaningfully beat baseline.

### 7. Validate and dry-run

```bash
quantbox autoresearch validate -c configs/clients/X/autoresearch.yaml
quantbox autoresearch run --dry-run -c configs/clients/X/autoresearch.yaml --max-trials 1
```

`--dry-run --max-trials 1` runs exactly one iteration end-to-end without committing to a multi-hour loop. Verifies the proposer, runner, evaluator, and memory all wire correctly.

### 8. Run

```bash
quantbox autoresearch run -c configs/clients/X/autoresearch.yaml
```

The loop persists state to `EXPERIMENTS.jsonl` after every trial — interrupting and resuming is safe.

For continuous (cron-based) operation, use `tick`:

```bash
quantbox autoresearch tick -c configs/clients/X/autoresearch.yaml --max-trials 5
```

Wire into Iris's schedule:

```bash
# In tom's crontab
30 6 * * * agent-cron iris 'autoresearch tick: client-X strategy. Budget: 5 trials, $5 LLM, 30 min. Post summary to #client-X if anything beats baseline by ≥15%.'
```

### 9. Monitor

```bash
quantbox autoresearch status -c configs/clients/X/autoresearch.yaml
```

Outputs:

```
Loop:        configs/clients/X/autoresearch.yaml
State:       running (tick 12 of 50)
Best:        T0042 — sharpe_oos 1.23 (+18% vs baseline 1.04)
Budget:      trials 12/50, llm $4.20/$25, compute $0.30/$5, wall 1h12m/4h
Convergence: 3/5 trials no-improve
Memory:      research/clients/X/EXPERIMENTS.jsonl (12 entries)
```

Discord posts auto-summaries every N trials and on candidate emergence.

### 10. On exit — review and decide

When the loop halts (budget, convergence, or `--max-trials`), it produces:

- A best variant (the winner) → `research/clients/X/candidates/{trial_id}.yaml`
- A summary report (Markdown) → `research/clients/X/autoresearch-report-{date}.md`
- A Discord post with metrics, chart, and a "Promote?" prompt

Review:

- Look at the metrics — is the lift real or noise? Check the bootstrap CI.
- Look at the lineage — what did the loop actually change? Sometimes the answer is "the loop overfit one regime" and the methodology should not be promoted.
- Look at `findings.md` — what did the LLM learn? Useful for human design intuition even if you don't promote.

If you decide to promote:

```bash
# This invokes the normal lifecycle promotion path
/promote-lock {plugin_name} --version {new_version}
```

See [promote-a-methodology.md](promote-a-methodology.md) for the rest.

If you decide *not* to promote, the trials are still in the log — they inform the next loop. Nothing is wasted.

---

## Common mistakes

| Mistake | Fix |
|---|---|
| Skipping the dry-run | Always do `--dry-run --max-trials 1` first; cheaper than a 4-hour loop that crashes at trial 2 |
| Setting search space too broad | Tight ranges learn faster; broaden after the loop converges in the tight space |
| Trusting in-sample Sharpe | The loop will p-hack; deflated-Sharpe + walk-forward are mandatory for a reason |
| Letting the budget be too generous | The first run should finish in 1–2 hours and cost <$30; you'll iterate on the loop config many times |
| Auto-promoting the winner | NEVER. The loop maxes at `research`; humans flip lock |
| Running multiple loops on the same EXPERIMENTS.jsonl | One config = one client = one log; concurrent writes corrupt state |
| Treating findings.md as authoritative | It's LLM-summarized prose; the truth is in EXPERIMENTS.jsonl |
| Skipping `quantbox autoresearch status` | Check often during the first few runs to catch budget creep |

---

## Iteration over time

Autoresearch becomes useful when you run it repeatedly:

- **Per-client cron tick** (daily, 5 trials, tight budget) accumulates knowledge slowly.
- **Bigger ad-hoc loops** (50+ trials, broader search) when a client wants a refresh.
- **Cross-client patterns** emerge when several clients show the same finding in their `findings.md`.

Over months, you build a body of empirical knowledge per client — auditable, reproducible, with full lineage. That's the long-term value, not any single loop's winner.

---

## See also

- [architecture/autoresearch.md](../architecture/autoresearch.md) — full design.
- [ADR-0003](../adr/0003-autoresearch-as-driver-not-runtime.md) — design decision.
- [promote-a-methodology.md](promote-a-methodology.md) — what happens after the loop produces a candidate.
- [add-a-plugin.md](add-a-plugin.md) — if you need to author a custom `VariantProposerPlugin`.
