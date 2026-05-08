"""Adapters package — thin pass-throughs to external libraries.

QuantBox composes external libraries rather than reimplementing them. Each
adapter re-exports the underlying library namespace so users can drop down
to the wheel when needed:

    from quantbox.adapters.vectorbt import vbt
    pf = vbt.Portfolio.from_signals(prices, entries, exits)

Convenience helpers (when an idiom proves common across consumers) live next
to the re-export but never replace it. See ``docs/architecture/adapters.md``
for the rule.

Available adapters:
    vectorbt  — vectorbt re-export + helpers
    (more added as needed; see docs/architecture/adapters.md)
"""
