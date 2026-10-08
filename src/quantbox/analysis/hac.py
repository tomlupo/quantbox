"""Deprecated path: Newey-West / HAC lives in :mod:`quantbox.inference` (TOM-1618).

Every name this module exported (``newey_west_tstat``, ``newey_west_auto_lags``,
``factor_regression``, ``require_finite``) resolves to the same object in
:mod:`quantbox.inference`, with a ``DeprecationWarning``.
"""

from quantbox._deprecation import moved

__getattr__ = moved(__name__, "quantbox.inference")
