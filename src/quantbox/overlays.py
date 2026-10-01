"""The overlay stage: a chain of modifiers between a decision and its execution (ADR-0004).

A backtest decides weights (strategies, then aggregation), the overlay chain
modifies that decided book in config order, and only then do the venue / risk
transforms and the execution lag (:mod:`quantbox.execution`) turn it into the
traded book. Every overlay therefore inherits ONE execution timing: no overlay
shifts its own output, so two overlays (or two research lines) can never mix
fills on different bars.

This module owns the chain and its contract checks; the overlays themselves
are plugins (``plugins/overlays``, kind ``overlay``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

#: One link of the chain: the overlay plugin and the ``params`` its config block sets.
OverlayLink = tuple[Any, dict[str, Any]]


def overlay_record(chain: Sequence[OverlayLink]) -> list[dict[str, Any]]:
    """What a run records for *chain* (``name``, ``version``, ``params`` per overlay, in order).

    :func:`apply_overlays` returns exactly this, and ``quantbox config explain``
    reports it before a run, so the plan and the manifest's ``overlays`` agree.
    """
    record = []
    for plugin, params in chain:
        meta = getattr(plugin, "meta", None)
        record.append(
            {
                "name": getattr(meta, "name", type(plugin).__name__),
                "version": getattr(meta, "version", None),
                "params": dict(params or {}),
            }
        )
    return record


def apply_overlays(
    weights: pd.DataFrame,
    data: dict[str, Any],
    chain: Sequence[OverlayLink],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply ``chain`` to the decided ``weights`` in order.

    Returns the overlaid weights and the record of what was applied (``name``,
    ``version``, ``params`` per overlay, in order) — the block a run writes to
    its manifest. An overlay that returns anything but a frame on the input's
    index and columns is refused: a moved row is a shifted signal, and a
    dropped column a position that silently vanished.
    """
    for plugin, params in chain:
        meta = getattr(plugin, "meta", None)
        name = getattr(meta, "name", type(plugin).__name__)
        # Snapshot BEFORE the call: an overlay that mutates its input in place
        # (``drop(columns=..., inplace=True)``) would otherwise be compared with
        # itself and pass. Each overlay also gets its own copy of the book and of
        # the data mapping, so neither the caller's frame (saved as
        # ``base_weights_history``) nor the mapping the engine later prices from
        # can be changed behind the chain's back.
        index, columns = weights.index.copy(), weights.columns.copy()
        out = plugin.apply(weights.copy(), dict(data), dict(params or {}))
        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"overlay {name!r} returned {type(out).__name__}, not a DataFrame")
        if not out.index.equals(index) or not out.columns.equals(columns):
            raise ValueError(
                f"overlay {name!r} changed the weights' index or columns; an overlay modifies row t in place "
                "and never shifts — the execution lag is applied once, after the whole chain"
            )
        weights = out
    return weights, overlay_record(chain)
