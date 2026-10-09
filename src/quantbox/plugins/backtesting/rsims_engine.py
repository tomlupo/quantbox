"""Deprecated: the rsims simulator moved to :mod:`quantbox.engine.rsims_sim` (TOM-1451).

It is core code behind the engine seam (ADR-0010), so it lives in the engine
package now. Every name resolves to the SAME object in
:mod:`quantbox.engine.rsims_sim` and emits a ``DeprecationWarning`` that names
the new path. This shim stays for one minor version.
"""

from quantbox._deprecation import moved

__getattr__ = moved("quantbox.plugins.backtesting.rsims_engine", "quantbox.engine.rsims_sim", card="TOM-1451")
