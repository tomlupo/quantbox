"""What a broker DECLARES it can do — the one reader of ``broker_capabilities`` (TOM-1459).

The trade path used to probe a broker with ``hasattr``/``getattr`` at every
site that wanted an optional method: funding, fees, the simulated cost model,
the paper-book price injection, ``get_equity``, ``fetch_order_result``. A probe
cannot tell "this venue cannot" from "this adapter misspelt the method", and a
private attribute (``_cumulative_fees``) became a public contract by accident.

Now a broker declares its capabilities in one class attribute,
``broker_capabilities``, typed by :class:`~quantbox.contracts.BrokerCapability`
(the contract and how a third-party broker declares one live there). The
pipeline asks :func:`supports`. Venue knowledge stays in the adapter.

The default is "not supported": a broker without the attribute declares
nothing, and every site takes the branch it took for a broker without the
method. :func:`mismatches` is the check that the declaration and the class
agree; ``quantbox plugins doctor`` and each broker's test run it.
"""

from __future__ import annotations

import logging
from typing import Any

from quantbox.contracts import BROKER_CAPABILITY_MEMBERS, BrokerCapability

logger = logging.getLogger(__name__)

__all__ = ["DECLARATION_ATTR", "declared", "mismatches", "supports"]

#: The class attribute a broker declares its capabilities in.
DECLARATION_ATTR = "broker_capabilities"


def declared(broker: Any) -> frozenset[BrokerCapability]:
    """The capabilities ``broker`` declares; empty when it declares none.

    Read from the instance, so a wrapper that forwards attributes (the
    quantbox-live shadow run's broker proxy) carries its broker's declaration.
    An entry that is not a :class:`BrokerCapability` (or its value) is logged
    and ignored: a declaration nobody can parse carries no more than no
    declaration, the same rule as an unparseable ``valuation_basis``.
    """
    if broker is None:
        return frozenset()
    raw = getattr(broker, DECLARATION_ATTR, None)
    if raw is None:
        return frozenset()
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        logger.error(
            "Broker %s declares %s=%r, which is not a set of BrokerCapability — treating it as declaring NOTHING.",
            type(broker).__name__,
            DECLARATION_ATTR,
            raw,
        )
        return frozenset()
    caps: set[BrokerCapability] = set()
    for item in raw:
        try:
            caps.add(BrokerCapability(item))
        except ValueError:
            logger.error(
                "Broker %s declares an unknown capability %r — ignored. Known: %s",
                type(broker).__name__,
                item,
                sorted(c.value for c in BrokerCapability),
            )
    return frozenset(caps)


def supports(broker: Any, capability: BrokerCapability) -> bool:
    """True when ``broker`` declares ``capability``. ``None`` supports nothing."""
    return capability in declared(broker)


def mismatches(broker_cls: type) -> list[str]:
    """Where a broker class's declaration and its members disagree; empty when they agree.

    Two directions, both reported:

    * declared, but a member the capability needs is missing — the pipeline
      would call it and fail;
    * every member present, but not declared — the pipeline will NOT use it,
      which is the behaviour change a third-party broker gets by not declaring.

    Checked on the class: a dataclass field with a default, a property and a
    method are all class attributes.
    """
    caps = declared(broker_cls)
    found: list[str] = []
    for cap, members in BROKER_CAPABILITY_MEMBERS.items():
        present = [m for m in members if hasattr(broker_cls, m)]
        if cap in caps and len(present) < len(members):
            missing = [m for m in members if m not in present]
            found.append(f"declares {cap.value} but has no {', '.join(missing)}")
        elif cap not in caps and len(present) == len(members):
            found.append(f"has {', '.join(members)} but does not declare {cap.value} — the pipeline will not use it")
    return found
