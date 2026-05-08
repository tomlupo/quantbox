# Playbook — Add an Adapter

When QuantBox needs to compose a new external library at L0/L1. Read [architecture/adapters.md](../architecture/adapters.md) for the rule first.

---

## Pre-flight

Confirm:

- [ ] At least 2 plugins, skills, or projects will use this lib.
- [ ] The lib is a *default choice* for its capability (not one of many alternatives).
- [ ] The lib is maintained and unlikely to be replaced soon.

If any answer is "no," don't add an adapter — just `import` the lib where you need it.

---

## Steps

### 1. Create the adapter file

Single-file libs:

```bash
touch src/quantbox/adapters/{lib}.py
```

Larger ones get a directory:

```bash
mkdir src/quantbox/adapters/{lib}
touch src/quantbox/adapters/{lib}/__init__.py
```

### 2. Re-export

The minimum viable adapter is a re-export:

```python
# adapters/{lib}.py
"""Adapter for {lib} — re-exports + thin helpers."""
import {lib_module} as {lib_alias}

__all__ = ["{lib_alias}"]
```

Example for riskfolio:

```python
import riskfolio as rp
__all__ = ["rp"]
```

### 3. Add the optional dependency

In `pyproject.toml`:

```toml
[project.optional-dependencies]
opt-{lib} = ["{lib}>=X.Y"]
```

Add to `full` if appropriate:

```toml
full = [..., "{lib}>=X.Y"]
```

### 4. Add convenience helpers (only if a common idiom exists)

```python
# adapters/{lib}.py
import {lib_module} as {lib_alias}

__all__ = ["{lib_alias}", "convenience_helper"]

def convenience_helper(returns, *, risk_free_rate=0.02):
    """One-line description."""
    # delegate to the library; don't reimplement
    ...
```

Skip this step on first add. Wait for the second consumer to confirm what idiom is actually common.

### 5. Wire to L1 namespace (if applicable)

If the adapter serves a top-level capability (`quantbox.bt`, `quantbox.opt`, `quantbox.score`):

```python
# src/quantbox/opt.py
from .adapters.riskfolio import max_sharpe, rp
__all__ = ["max_sharpe", "rp"]
```

### 6. Test

`tests/adapters/test_{lib}.py`:

```python
def test_reexport_is_the_library():
    from quantbox.adapters.riskfolio import rp
    import riskfolio
    assert rp is riskfolio

def test_convenience_helper_runs():
    import pandas as pd
    from quantbox.adapters.riskfolio import max_sharpe
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, -0.01]})
    w = max_sharpe(returns)
    assert w is not None
```

Don't test the underlying library's behavior. That's their job.

### 7. Document

Add a one-line entry in:

- `docs/architecture/adapters.md` — table of existing adapters
- `CHANGELOG.md` — under "Added"

### 8. (Optional) Add an L3 plugin wrapper

If a pipeline at L4 will use this capability, add a plugin. Don't do this preemptively — wait for the use case. See [add-a-plugin.md](add-a-plugin.md).

---

## Validation

Before committing:

- [ ] `from quantbox.adapters.{lib} import {lib_alias}` works.
- [ ] Every helper has a test.
- [ ] No reimplementation of features that exist in the lib.
- [ ] No opaque wrapper class hides the lib's API.
- [ ] Optional dep declared in `pyproject.toml`.
- [ ] Adapter table in docs updated.

---

## Common mistakes

| Mistake | Fix |
|---|---|
| Wrapping the lib in a new class hierarchy | Use a function-style helper that delegates |
| Translating the lib's exception types to quantbox's | Let exceptions propagate |
| Adding the dep to base `dependencies` not `optional-dependencies` | Adapters are opt-in |
| Writing helpers that duplicate the lib's existing functions | Just re-export |
| Adding a helper for a single use case before a second emerges | Wait for the pattern to be real |

---

## See also

- [architecture/adapters.md](../architecture/adapters.md) — the rule.
- [architecture/api-layers.md](../architecture/api-layers.md) — where adapters live (L0/L1).
- [architecture/principles.md](../architecture/principles.md) — adapter-not-reimplementation.
