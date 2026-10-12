---
adr: 0011
title: Trade code that P5 writes or rewrites lives under quantbox.trade
status: accepted
date: 2026-10-10
supersedes:
superseded_by:
amends: "ADR-0010 decision 3, in sense only: the trade layer gains a package of its own, so a new trade module's path shows its layer."
status_changes:
  - 2026-10-10: accepted with TOM-1711 — Tom, 2026-10-10 (design review of P3+P4, then P5 grilling Q6/Q11): "quantbox.trade.* (new only)"; then "if design is better to be under trades please move"
---

# ADR-0011: Trade code that P5 writes or rewrites lives under quantbox.trade

## Context

ADR-0010 put the trade layer in one distribution, as a list of modules in the
`trade` import-linter contract. The list is scattered: `quantbox.plugins.broker`,
`plugins.rebalancing`, `plugins.pipeline.trading_pipeline`, `quantbox.portfolio_value`,
`quantbox.reconciliation`, and more. A reader cannot tell a module's layer from
its path. The design review of 2026-10-10 (https://claude.ai/artifact/Tv3rn1WDPA7mZrhNHvT9CS)
named this as the layer map's locality cost. P5 (TOM-1445) splits the 3,991-line
trading pipeline into new modules: a marked book, the reconciliation cycle, an
execution module, a notifier.

## Options considered

1. **Today's layout** (new modules beside the old ones, in `plugins/…` and the
   root). Rejected: every new module adds to the scatter, and the path keeps
   hiding the layer.
2. **Move all trade code now**, plugins included. Rejected: quantbox-live imports
   two broker classes by module path. A move of every plugin breaks it for no
   gain to P5. The YAML ids would stay, but the churn would hide P5's real
   changes in the shadow-run diffs.
3. **New and rewritten modules under `quantbox.trade`** (chosen).

## Decision

- Every new trade module goes under `quantbox.trade.*`.
- The two packages that P5 rewrites move there: `quantbox.reconciliation` →
  `quantbox.trade.reconciliation` (TOM-1455), and `quantbox.portfolio_value`
  becomes the marked book at `quantbox.trade.book` (TOM-1455).
- Their old paths are refused through `quantbox._removed`, which names the new
  path. quantbox-live changes its imports in the PR that bumps its pin.
- The trade plugins (`plugins/broker`, `plugins/rebalancing`, `plugins/risk`,
  `plugins/publisher`, `plugins/trading`, the trade pipelines) keep their
  module paths in P5. YAML plugin ids never change.
- `[tool.importlinter]` stays the one owner of the layer map:
  `quantbox.trade` and `quantbox.trade.**` join the `trade` contract.

## Consequences

- (+) A P5 module's layer can be read from its path.
- (+) quantbox-live's reconciliation imports get an exported home: today three
  of them come from `reconciliation.working_orders`, which the package does
  not export.
- (−) Accepted cost: for a while, trade code lives in two places (`quantbox.trade`
  and the plugin packages), so a reader must still check the contract.
- (−) A breaking import change for quantbox-live at the 5a pin bump.
- ⚑ Whether the trade plugins move after P5 is open. Decide it on its own card,
  only if a consumer gains from it.
