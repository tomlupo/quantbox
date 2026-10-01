"""Overlay plugins: modifiers chained between a base strategy's decision and execution (ADR-0004)."""

from .corr_gross_cap import CorrGrossCapOverlay
from .regime_reweight import RegimeReweightOverlay
from .reversal_derisk import ReversalDeriskOverlay

__all__ = [
    "CorrGrossCapOverlay",
    "RegimeReweightOverlay",
    "ReversalDeriskOverlay",
]
