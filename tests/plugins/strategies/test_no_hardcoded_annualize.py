"""CI guard: registered strategies must not hardcode their own annualize/trading_days default.

The pipeline (`backtest.pipeline.v1`) injects `_pipeline_annualize` into every
strategy's params, derived from the resolved `Frequency`. Any strategy that
hardcodes its own default (e.g. `annualize: float = 252.0`) will silently use
that value instead of the pipeline-injected one — which is the exact drift
this PR series is built to prevent.

This test enumerates every registered strategy class and asserts that any
dataclass field named `annualize` / `trading_days` / `days_per_year` /
`bars_per_year` / `annualization` defaults to `None`. Strategies that don't
need annualization don't have these fields and are unaffected.

If this test fails, the failing strategy needs to:
  1. Flip its annualization field default to `None`
  2. In `run()`, resolve via:
       pipeline_annualize = (params or {}).get("_pipeline_annualize")
       effective = self.<field> if self.<field> is not None else (pipeline_annualize or 252.0)
"""

from __future__ import annotations

import re
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from quantbox.plugins import strategies as _strategies_pkg

# Names of fields that the pipeline now injects via `_pipeline_annualize`.
# Any dataclass field with one of these names MUST default to None on a
# registered strategy plugin.
_INJECTED_FIELD_NAMES = frozenset(
    {
        "annualize",
        "annualization",
        "trading_days",
        "days_per_year",
        "bars_per_year",
    }
)


def _registered_strategy_classes() -> list[type]:
    """All public strategy classes exposed by quantbox.plugins.strategies.__all__."""
    out = []
    for name in _strategies_pkg.__all__:
        obj = getattr(_strategies_pkg, name)
        if isinstance(obj, type) and hasattr(obj, "meta") and is_dataclass(obj):
            out.append(obj)
    return out


@pytest.mark.parametrize(
    "strategy_cls",
    _registered_strategy_classes(),
    ids=lambda c: c.__name__,
)
def test_strategy_annualization_field_defaults_to_none(strategy_cls):
    """If a strategy declares an annualization field, its default must be None."""
    for f in fields(strategy_cls):
        if f.name in _INJECTED_FIELD_NAMES:
            assert f.default is None, (
                f"{strategy_cls.__name__}.{f.name} hardcodes default={f.default!r}; "
                f"set it to None and consume `_pipeline_annualize` from params in run(). "
                f"See cross_asset_momentum.py / vol_matched_buy_hold.py for the pattern."
            )


def test_at_least_one_strategy_has_an_annualization_field():
    """Sanity: the test infrastructure works (at least one strategy has the field)."""
    found = False
    for cls in _registered_strategy_classes():
        for f in fields(cls):
            if f.name in _INJECTED_FIELD_NAMES:
                found = True
                break
        if found:
            break
    assert found, (
        "No registered strategy has an annualization field — either every strategy "
        "uses hardcoded sqrt(252)/sqrt(365) calls (worse problem) or the field-name "
        "list is stale."
    )


# ---------------------------------------------------------------------------
# Stronger guard: no strategy *implementation* may contain literal
# np.sqrt(365) or np.sqrt(252). Catches the failure mode the dataclass-field
# guard above misses — annualization buried inside helper functions or
# inline expressions (issue #23).
# ---------------------------------------------------------------------------


_STRATEGIES_DIR = Path(_strategies_pkg.__file__).parent
_HARDCODED_SQRT_RE = re.compile(r"np\.sqrt\(\s*(252|365)\s*\)")
# Files exempt from the rule. Add explanations when adding entries.
_EXEMPT_FILES: set[str] = set()  # currently none — all three known offenders fixed in issue #23


@pytest.mark.parametrize(
    "strategy_file",
    sorted(p for p in _STRATEGIES_DIR.glob("*.py") if p.name != "__init__.py"),
    ids=lambda p: p.name,
)
def test_no_hardcoded_sqrt_252_or_365_in_strategies(strategy_file: Path):
    """Catch hardcoded ``np.sqrt(252)`` / ``np.sqrt(365)`` calls anywhere
    inside a strategy module — including helper functions outside the class.

    The pipeline-injection scheme requires ALL annualization (not just the
    dataclass-field-declared kind) to flow from `_pipeline_annualize`.
    Hardcoded sqrt calls bypass the injection silently.

    If this test fails, the offending helper should accept an `annualize`
    parameter (defaulting to 252.0) that the strategy's `run()` method
    threads through from `effective_annualize`.
    """
    if strategy_file.name in _EXEMPT_FILES:
        pytest.skip(f"{strategy_file.name} explicitly exempt")
    content = strategy_file.read_text()
    matches = _HARDCODED_SQRT_RE.findall(content)
    assert not matches, (
        f"{strategy_file.name} contains hardcoded np.sqrt({matches[0]}) — "
        f"strategies must accept the annualization factor via the "
        f"pipeline-injected `_pipeline_annualize` (see beglobal_strategy.py / "
        f"carver_trend.py / momentum_long_short.py for the helper-param pattern "
        f"introduced by issue #23)."
    )
