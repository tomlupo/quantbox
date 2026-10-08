"""The funding guard (TOM-1609): a funding series on an engine that does not charge it is refused.

An engine declares whether it charges the funding series it is handed
(``EngineAdapter.charges_funding``, docs/adr/0008). vectorbt does not. A perps
config with funding data on vectorbt used to run silently and leave the funding
cost out of the book; ``quantbox config explain`` showed ``funding.modelled:
false`` only as information.

Now ONE check, :func:`check_funding`, refuses that config. ``quantbox run``
calls it before any data is read (the data plugin's planned funding file) and
again on the frames a data plugin hands back (an API plugin plans no file).
``quantbox config explain`` calls it through the same
``BacktestPipeline.check_planned_data``, and ``quantbox validate`` reports what
explain refuses. The one escape is declared in the backtest pipeline's params::

    funding: {ignore: true, reason: "<why this book may leave funding out>"}

The reason is recorded in run@1 ``funding.ignored_reason``. On an engine that
charges funding the block is refused: it would not change the book.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .exceptions import ConfigValidationError

#: Finding code of the refusal; ``quantbox validate`` and the run carry it.
FUNDING_NOT_CHARGED = "funding_not_charged"

#: The declared escape, as a config spells it.
FUNDING_IGNORE = 'funding: {ignore: true, reason: "<why this book may leave funding out>"}'

#: ``funding:`` in ``backtest.pipeline.v1`` params.
FUNDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Leave a funding series out on an engine that does not charge it (TOM-1609): "
        "{ignore: true, reason: ...}. Without it, a run whose data carries funding on such an "
        "engine is refused. The reason is recorded in run_manifest.json funding.ignored_reason. "
        "Engine choice: docs/adr/0008."
    ),
    "properties": {
        "ignore": {"type": "boolean"},
        "reason": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}


@dataclass(frozen=True)
class FundingIgnore:
    """A declared ``funding: {ignore: true, reason}``."""

    reason: str

    def record(self) -> dict[str, str]:
        """The run@1 ``funding`` fields this declaration adds."""
        return {"ignored_reason": self.reason}


def resolve_funding(block: Any, *, charges_funding: bool, engine: str) -> FundingIgnore | None:
    """``funding:`` -> the declared ignore, or None. Raises ``ValueError`` on a malformed block."""
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError(f"funding must be a mapping like {FUNDING_IGNORE}, got {block!r}")
    unknown = sorted(set(block) - {"ignore", "reason"})
    if unknown:
        raise ValueError(f"funding: unknown key(s) {unknown}; the keys are 'ignore' and 'reason'")
    ignore = block.get("ignore", False)
    if not isinstance(ignore, bool):
        raise ValueError(f"funding.ignore must be true or false, got {ignore!r}")
    if not ignore:
        return None
    reason = block.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(
            f"funding.reason must be a non-empty string, got {reason!r}: leaving funding out "
            "overstates a perps book, and the reason is what the run manifest records for it."
        )
    if charges_funding:
        raise ValueError(
            f"funding.ignore is declared, but engine '{engine}' charges funding: the block would "
            "not change the book. Delete it."
        )
    return FundingIgnore(reason.strip())


def ignored_record(plan: dict[str, Any]) -> dict[str, str]:
    """The run@1 ``funding`` fields a plan's declared ignore adds: ``{}``, or ``{ignored_reason}``."""
    ignore = plan.get("funding_ignore")
    return ignore.record() if isinstance(ignore, FundingIgnore) else {}


def _charging_engines() -> list[str]:
    from .engine.registry import engine_names, get_engine

    return [n for n in engine_names() if get_engine(n, require_installed=False).charges_funding]


def check_funding(plan: dict[str, Any], source: str | None) -> None:
    """Refuse a funding series the planned engine would not charge.

    *plan* is ``BacktestPipeline.plan(params)``; *source* names where the funding
    series comes from (a file path, or the data plugin that returned it), None
    when the data carries none. Raises :class:`~quantbox.exceptions.ConfigValidationError`
    with one ``funding_not_charged`` finding.
    """
    if not source or plan.get("charges_funding") or plan.get("funding_ignore") is not None:
        return
    from .validate import ValidationFinding

    engine = plan["engine"]
    charging = " or ".join(f"engine: {n}" for n in _charging_engines())
    finding = ValidationFinding(
        "error",
        f"{FUNDING_NOT_CHARGED}: the data carries a funding series ({source}), but engine '{engine}' "
        f"does not charge funding: the backtest would leave the funding cost out and overstate a "
        f"perps book. Use {charging} (docs/adr/0008), or declare {FUNDING_IGNORE} in "
        "plugins.pipeline.params.",
        FUNDING_NOT_CHARGED,
        {"engine": engine, "source": source},
    )
    raise ConfigValidationError(finding.message, findings=[finding])
