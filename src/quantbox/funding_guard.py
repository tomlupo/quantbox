"""The funding guard (TOM-1609, TOM-1619): a perps book that would leave funding out is refused.

An engine declares whether it charges the funding series it is handed
(``EngineAdapter.charges_funding``, docs/adr/0008). vectorbt does not. A perps
config with funding data on vectorbt used to run silently and leave the funding
cost out of the book.

ONE check, :func:`check_funding`, refuses three things:

- a funding SERIES on an engine that does not charge it (``funding_not_charged``);
- a PERP MARKET on an engine that does not charge funding, with or without a series
  (``funding_not_charged``) — a data plugin may hand back no funding at all;
- a perp market with NO funding series on an engine that charges it
  (``funding_missing``): the engine would charge zero.

The market is the dataset's own declaration (its manifest ``market``), which a
data plugin answers through an optional ``planned_market()`` method
(:func:`planned_market`) before any data is read. :data:`PERP_MARKETS` lists the
values that mean a perpetual market.

Where the check runs: ``quantbox run`` calls it before any data is read (the
data plugin's market and planned funding file) and again on the frames a data
plugin hands back. ``quantbox config explain`` calls it through the same
``BacktestPipeline.check_planned_data``, and ``quantbox validate`` reports what
explain refuses. The engine seam, :func:`quantbox.engine.simulate`, calls it on
the series it is handed (:func:`check_series`), so ``backtest()``, ``optimize()``
and the sweep are refused too. The one escape, in the backtest pipeline's params
and as the ``funding=`` argument of each helper::

    funding: {ignore: true, reason: "<why this book may leave funding out>"}

The reason is recorded in run@1 ``funding.ignored_reason``. With a funding series
on an engine that charges it, the block is refused: it would not change the book.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .exceptions import ConfigValidationError

#: Finding code of the refusal; ``quantbox validate`` and the run carry it.
FUNDING_NOT_CHARGED = "funding_not_charged"

#: Finding code: a perp market with no funding series on an engine that charges funding.
FUNDING_MISSING = "funding_missing"

#: Finding code: an ignore declared where the engine charges the series it is handed.
FUNDING_IGNORE_UNUSED = "funding_ignore_unused"

#: Dataset ``market`` values that mean a perpetual (funding-bearing) market. quantbox-datasets
#: spells Binance USDM perpetuals ``futures`` (crypto-futures-daily), and its instrument
#: registry reads the three values as one.
PERP_MARKETS = frozenset({"perp", "perps", "futures"})

#: The declared escape, as a config spells it.
FUNDING_IGNORE = 'funding: {ignore: true, reason: "<why this book may leave funding out>"}'

#: ``funding:`` in ``backtest.pipeline.v1`` params.
FUNDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Leave funding out of a perps book (TOM-1609, TOM-1619): {ignore: true, reason: ...}. "
        "Without it, a run is refused when its data carries funding or its dataset is a perp market "
        "on an engine that does not charge funding, and when a perp dataset has no funding series on "
        "an engine that does. The reason is recorded in run_manifest.json funding.ignored_reason. "
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


def resolve_funding(block: Any) -> FundingIgnore | None:
    """``funding:`` -> the declared ignore, or None. Raises ``ValueError`` on a malformed block.

    Whether an ignore is honoured depends on the data (a funding series, a perp
    market), so :func:`check_funding` refuses one that would not change the book.
    """
    if block is None or isinstance(block, FundingIgnore):
        return block
    if not isinstance(block, Mapping):
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
    return FundingIgnore(reason.strip())


def ignored_record(plan: dict[str, Any]) -> dict[str, str]:
    """The run@1 ``funding`` fields a plan's declared ignore adds: ``{}``, or ``{ignored_reason}``."""
    ignore = plan.get("funding_ignore")
    return ignore.record() if isinstance(ignore, FundingIgnore) else {}


def _charging_engines() -> list[str]:
    from .engine.registry import engine_names, get_engine

    return [n for n in engine_names() if get_engine(n, require_installed=False).charges_funding]


def is_perp(market: str | None) -> bool:
    """Whether a dataset ``market`` is a perpetual (funding-bearing) market (:data:`PERP_MARKETS`)."""
    return isinstance(market, str) and market.strip().lower() in PERP_MARKETS


def planned_market(data: Any) -> str | None:
    """The market *data* will serve — its optional ``planned_market()``, read before any data — or None.

    A data plugin that reads a dataset answers with the dataset manifest's ``market``
    (``local_file_data`` by name; quantbox-datasets' ``dataset.curated.v1``). A plugin
    without the method, or one reading inline files, answers None: no market is known.
    """
    answer = getattr(data, "planned_market", None)
    if not callable(answer):
        return None
    market = answer()
    return str(market) if market else None


#: Where a pipeline config declares the escape.
IN_CONFIG = "in plugins.pipeline.params"
#: Where a helper (backtest(), optimize(), the sweep, simulate) takes it.
AS_ARGUMENT = "as the funding= argument"


def check_funding(
    plan: Mapping[str, Any],
    source: str | None,
    *,
    market: str | None = None,
    funding_known: bool = True,
    declare: str = IN_CONFIG,
) -> None:
    """Refuse a perps book the planned engine would get wrong on funding (see the module docstring).

    *plan* carries ``engine``, ``charges_funding`` and ``funding_ignore``
    (``BacktestPipeline.plan(params)``, or :func:`engine_plan`). *source* names
    where the funding series comes from (a file path, or the data plugin that
    returned it), None when the data carries none. *market* is the dataset's
    market (:func:`planned_market`). *funding_known* False means the series is
    not known yet (a data plugin that plans no file, before it loads), so a
    missing series is not refused here. *declare* says where the escape goes.
    Raises :class:`~quantbox.exceptions.ConfigValidationError` with one finding.
    """
    engine = plan["engine"]
    ignore = plan.get("funding_ignore")
    perp = is_perp(market)
    if plan.get("charges_funding"):
        if source and ignore is not None:
            _refuse(
                FUNDING_IGNORE_UNUSED,
                f"funding.ignore is declared, but engine '{engine}' charges funding and the data carries "
                f"a funding series ({source}): the block would not change the book. Delete it.",
                engine,
                source,
                market,
            )
        if perp and not source and funding_known and ignore is None:
            _refuse(
                FUNDING_MISSING,
                f"the dataset is a perp market (market: {market}) but carries no funding series: engine "
                f"'{engine}' charges funding and would charge zero, overstating a perps book. Add the "
                f"funding series to the dataset, or declare {FUNDING_IGNORE} {declare}.",
                engine,
                source,
                market,
            )
        return
    if ignore is not None or not (source or perp):
        return
    what = (
        f"the data carries a funding series ({source})"
        if source
        else f"the dataset is a perp market (market: {market})"
    )
    charging = " or ".join(f"engine: {n}" for n in _charging_engines())
    _refuse(
        FUNDING_NOT_CHARGED,
        f"{what}, but engine '{engine}' does not charge funding: the backtest would leave the funding "
        f"cost out and overstate a perps book. Use {charging} (docs/adr/0008), or declare "
        f"{FUNDING_IGNORE} {declare}.",
        engine,
        source,
        market,
    )


def _refuse(code: str, message: str, engine: str, source: str | None, market: str | None) -> None:
    from .validate import ValidationFinding

    finding = ValidationFinding(
        "error", f"{code}: {message}", code, {"engine": engine, "source": source, "market": market}
    )
    raise ConfigValidationError(finding.message, findings=[finding])


def engine_plan(engine: Any, block: Any) -> dict[str, Any]:
    """The fields :func:`check_funding` reads, for a helper door: *engine* (an adapter) and its ``funding=``."""
    return {"engine": engine.name, "charges_funding": engine.charges_funding, "funding_ignore": resolve_funding(block)}


def check_series(engine: Any, series: Any, block: Any, *, where: str = "") -> FundingIgnore | None:
    """The seam's check (:func:`quantbox.engine.simulate`): the series handed to *engine*, with ``funding=`` *block*.

    Every door that reaches the seam — the pipeline, ``backtest()``, ``optimize()``,
    the sweep — is checked here with :func:`check_funding`. Returns the resolved ignore.
    """
    plan = engine_plan(engine, block)
    has_series = series is not None and not getattr(series, "empty", True)
    check_funding(plan, f"{where}the funding series handed to the engine" if has_series else None, declare=AS_ARGUMENT)
    return plan["funding_ignore"]
