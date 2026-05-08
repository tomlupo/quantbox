---
methodology: subsystem-slug                # matches promote.config.yaml key
version: v0.1.0                            # semver
status: DRAFT                              # DRAFT | LOCKED | SUPERSEDED
date_locked:                               # YYYY-MM-DD when status flipped to LOCKED
supersedes:                                # prior version, if any (e.g. v0.0.x)
superseded_by:                             # next version, when this becomes obsolete
src_paths:                                 # code files this spec describes
  - src/path/to/plugin.py
  - pipelines/path/to/orchestrator.py
seeds:                                     # required for --strict
  numpy: 42
  sklearn: 42
revalidation:
  cadence: monthly                         # monthly | quarterly | manual
  baseline_metrics:
    sharpe_oos_min: 0.4
    drawdown_max: -0.18
  on_failure: alert_only                   # alert_only | auto_demote_to_research
sections:                                  # optional — enables partial promotions
  - id: §3
    status: DRAFT
    date:
---

# {Methodology name}

> One-paragraph plain-English summary. What this methodology does and what problem it solves.

## Goal

What problem does this solve? What's the hypothesis being expressed? What does success look like?

## Inputs and outputs

| Side | Type | Source / sink |
|---|---|---|
| Input — universe | DataFrame[symbol, ...] | `{namespace}.data.universe.v1` |
| Input — prices | wide-format DataFrame | `{namespace}.data.prices.v1` |
| Output — scores | DataFrame[symbol, score, asof] | `scores` artifact |

## Algorithm

Step-by-step. Include formulas where they matter; KaTeX-compatible LaTeX is fine.

### §1. Step name

Plain-English description.

$$
\text{score}_i = w_m \cdot z_{m,i} + w_c \cdot z_{c,i} + w_q \cdot z_{q,i}
$$

Where $z_{m,i}$ is the z-score of momentum indicator within the scoring category, etc.

### §2. Next step

...

## Parameters and their effect

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `lookback_days` | 60 | [30, 252] | Larger → smoother but slower to react |
| `vol_target` | 0.15 | [0.05, 0.30] | Annualized target volatility for position sizing |

## Validation evidence

Reference the EXPERIMENTS.md entry and validation runs that justified the lock.

| Run | Period | Sharpe (OOS) | Deflated Sharpe | Max DD | Notes |
|---|---|---|---|---|---|
| `ar_T0042_2026-04-25T14-33` | 2020–2025 | 1.23 | 0.42 | -0.16 | baseline |
| `ar_T0067_2026-04-26T09-12` | 2020–2025 | 1.41 | 0.51 | -0.18 | post-tuning |

EXPERIMENTS log: [`research/{study}/EXPERIMENTS.md`](../../research/{study}/EXPERIMENTS.md)

## Known limitations

What this methodology does NOT handle. Edge cases. Regime sensitivities. Failure modes that are accepted (and documented) rather than fixed.

- Doesn't handle <case>; deferred to v0.2.
- Sensitive to <regime>; revalidation will demote if drift > X.
- Not validated for <asset class>; use only within <scope>.

## Lineage

- Supersedes: `v0.0.x`
- Branched from: [`research/{study}/EXPERIMENTS.md#YYYY-MM-DD-entry`](../../research/{study}/EXPERIMENTS.md)
- Related work: [paper title or link]

## See also

- Plugin: [`src/{path}/plugin.py`](../../src/{path}/plugin.py)
- EXPERIMENTS log: [`research/{study}/EXPERIMENTS.md`](../../research/{study}/EXPERIMENTS.md)
- Lifecycle: [`../architecture/lifecycle.md`](../architecture/lifecycle.md)
- Promotion playbook: [`../playbooks/promote-a-methodology.md`](../playbooks/promote-a-methodology.md)
