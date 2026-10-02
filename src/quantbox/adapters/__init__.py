"""Adapters package — thin pass-throughs to external libraries.

QuantBox composes external libraries rather than reimplementing them. Each
adapter re-exports the underlying library namespace so users can drop down
to the wheel when needed:

    from quantbox.adapters.vectorbt import vbt
    # vbt fills the bar it is handed: lag the signals one bar first
    # (next-bar is mandatory, docs/adr/0005-next-bar-is-mandatory.md).
    entries, exits = entries.shift(1, fill_value=False), exits.shift(1, fill_value=False)
    pf = vbt.Portfolio.from_signals(prices, entries, exits)

Convenience helpers (when an idiom proves common across consumers) live next
to the re-export but never replace it. See ``docs/architecture/adapters.md``
for the rule.

Available adapters:
    vectorbt  — vectorbt re-export + helpers
    (more added as needed; see docs/architecture/adapters.md)
"""
