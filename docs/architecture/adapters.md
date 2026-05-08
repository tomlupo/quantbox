# Adapters

Adapters are how QuantBox composes external libraries instead of reimplementing them. This is the [adapter-not-reimplementation principle](principles.md#5-adapter-not-reimplementation) made operational.

---

## The rule

**A QuantBox adapter is a thin pass-through plus optional convenience helpers. The underlying library is re-exported, not hidden.**

If you find yourself writing logic that exists in the underlying library, stop. Use it.

---

## Structure

Each adapter lives at `packages/quantbox-core/src/quantbox/adapters/{lib}.py` (or `/{lib}/` for larger ones).

Minimum:

```python
# adapters/vectorbt.py
"""Adapter for vectorbt — re-exports + thin helpers.

Users should import vbt directly via:
    from quantbox.adapters.vectorbt import vbt
"""
import vectorbt as vbt

__all__ = ["vbt"]
```

That's a valid adapter. The pass-through is the contract. Anything else is bonus.

When you add convenience helpers, they live alongside:

```python
# adapters/vectorbt.py
import pandas as pd
import vectorbt as vbt

__all__ = ["vbt", "from_signals_with_costs"]

def from_signals_with_costs(prices, signals, *, fees=0.001, slippage=0.0005, freq="1D"):
    """Convenience: vbt.Portfolio.from_signals with sensible cost defaults."""
    return vbt.Portfolio.from_signals(
        close=prices,
        entries=signals > 0,
        exits=signals <= 0,
        fees=fees, slippage=slippage, freq=freq,
    )
```

---

## When to add an adapter

Add one when **all** are true:

1. The external library is going to be used by ≥2 plugins, skills, or downstream projects.
2. The library is a *default choice* for that capability in QuantBox (not an optional one of many).
3. The adapter doesn't impose ceremony beyond `import` — users can still drop to the underlying lib for any feature the adapter doesn't expose.

Don't add an adapter:

- For a library used in exactly one place (just `import` it there).
- For a one-off domain need (lives in the project).
- That hides the underlying API behind an opaque wrapper class.
- That requires the user to learn a new vocabulary when the underlying library already has one.

---

## What an adapter is allowed to do

| Allowed | Not allowed |
|---|---|
| Re-export the library namespace (`vbt`, `mlflow`) | Hide it behind a class hierarchy |
| Add convenience helpers for common idioms | Re-implement features that already exist |
| Accept QuantBox primitives (DataFrame, Series, dict) and pass through to the lib | Force users to learn a quantbox-specific type system |
| Translate between QuantBox conventions and the lib's conventions (e.g., wide ↔ long) | Translate silently in ways that lose information |
| Provide a Plugin wrapper (e.g., `BacktestPlugin` using vbt) at L3 | Make the plugin the *only* way to use the lib |

---

## Adapter inventory

| Adapter | Underlying library | Layer it serves | Status | Notes |
|---|---|---|---|---|
| `adapters.vectorbt` | vectorbt | L0/L1 (`quantbox.bt`) | ✅ shipped | Used by `bt.py`, backtest engine, strategy tests — ≥2 consumers |
| `adapters.riskfolio` | Riskfolio-Lib | L0/L1 (`quantbox.opt`) | deferred | Add when a second consumer beyond `portfolio_optimizer` needs it |
| `adapters.lightgbm` | lightgbm | L0/L1 (ML strategies) | deferred | `ml_strategy.py` imports it directly — add adapter when a second plugin needs it |
| `adapters.mlflow` | mlflow | experiment tracking, model registry | **not in core** | Single consumer (quantbox-lab); lab imports mlflow directly. Migrate here if ≥2 repos need the same `RunResult → mlflow` bridge |
| `adapters.dvc` | dvc | data versioning | **not in core** | Single consumer (quantbox-lab); lab imports dvc.api directly. Migrate here if ≥2 repos need the same data-versioning idiom |
| `adapters.qlib` | Microsoft Qlib | factor research | not in core | Optional; add if a Qlib-based plugin is ever built |

**Rule for "not in core":** the library is used in one downstream repo (quantbox-lab). That repo imports it directly. If a second repo (quantbox-live, quantbox-qute) needs the same bridge, extract to a core adapter then. Premature extraction adds ceremony without DRY benefit.

**Rule for "deferred":** the library touches quantbox-core but only one plugin currently uses it. Add the adapter file when a second plugin or downstream project needs the same idiom — not before.

---

## Walkthrough — adding the riskfolio adapter

1. **Create the file**: `packages/quantbox-core/src/quantbox/adapters/riskfolio.py`.
2. **Re-export**:
   ```python
   import riskfolio as rp
   __all__ = ["rp"]
   ```
3. **Add a convenience helper** if a common idiom emerges:
   ```python
   def max_sharpe(returns, *, risk_free_rate=0.02):
       port = rp.Portfolio(returns=returns)
       port.assets_stats(method_mu="hist", method_cov="hist")
       w = port.optimization(model="Classic", rm="MV", obj="Sharpe", rf=risk_free_rate, l=0)
       return w
   ```
4. **Wire the L1 surface** at `quantbox/opt.py`:
   ```python
   from .adapters.riskfolio import max_sharpe, rp
   __all__ = ["max_sharpe", "rp"]
   ```
5. **Add an extras entry** in `pyproject.toml`:
   ```toml
   [project.optional-dependencies]
   opt-riskfolio = ["riskfolio-lib>=4.0"]
   ```
6. **Test** the convenience helper with a tiny synthetic returns DataFrame. Don't test riskfolio itself — that's their job.
7. **Document** in `docs/reference/adapters.md` (one-line entry) and in this file's table.

That's the whole flow. No plugin needed at this stage. A plugin (`risk.opt.riskfolio.max_sharpe.v1`) can come later if a pipeline needs it at L4.

---

## When NOT to wrap

Examples of things that are tempting to wrap but shouldn't be:

- **Pandas/Numpy** — already universal; no wrapper adds value.
- **FastAPI** — domain-specific (dm-evo's app), not a quantbox concern.
- **Custom domain models** (advisory profiles, regulated identifiers) — domain belongs in projects.
- **A library you're going to use once** — just import it where you need it.

If you catch yourself adding a `quantbox.X` that exists in `pandas.X`, you've crossed the line.

---

## Versioning rule

When an underlying library has a breaking change:

- **Bump the adapter's minor version.** The pass-through still works (caller imports lib directly), but convenience helpers may break.
- **Bump the adapter's major version** if you remove a convenience helper or change its signature.
- The pass-through itself never breaks unless the underlying library is removed entirely (which is itself a major version event for QuantBox).

Adapters should be more conservative than plugins; downstream relies on them.

---

## Anti-patterns to refuse

| Anti-pattern | Fix |
|---|---|
| `class VectorbtEngine: ...` wrapping vbt opaquely | Re-export `vbt` directly; add helpers as functions |
| Custom DataFrame schema that vbt doesn't understand | Pass through standard wide-format prices; let vbt validate |
| Hiding vbt error messages behind generic exceptions | Let exceptions propagate; users debug against the real lib |
| Adapter requires importing 3+ external libs | That's a pipeline, not an adapter |
| Adapter has its own `Pipeline` / `Runner` / `Engine` class | Stop. That's the runner's job at L4. |

---

## See also

- [api-layers.md](api-layers.md) — how adapters compose into L0/L1.
- [principles.md](principles.md) — the doctrine.
- [playbooks/add-an-adapter.md](../playbooks/add-an-adapter.md) — step-by-step.
