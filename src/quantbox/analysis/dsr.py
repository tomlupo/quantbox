"""Deprecated path: the Deflated Sharpe Ratio lives in :mod:`quantbox.inference` (TOM-1618).

Every name this module exported (``deflated_sharpe_ratio``,
``deflated_sharpe_ratio_from_returns``, ``expected_max_sr``,
``sr_estimator_std``, ``DSRResult``, ``DEGENERATE_RTOL``, ``EULER_MASCHERONI``)
resolves to the same object in :mod:`quantbox.inference`, with a
``DeprecationWarning``.
"""

from quantbox._deprecation import moved

__getattr__ = moved(__name__, "quantbox.inference")
