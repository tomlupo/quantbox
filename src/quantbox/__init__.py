"""quantbox: one distribution in four layers, core < plugins < research, trade (ADR-0010).

``__all__`` lists core submodules only, so ``from quantbox import *`` imports
no plugin, research or trade code (TOM-1451). ``quantbox.bt``,
``quantbox.adapters`` and ``quantbox.plugins`` stay importable by name.
"""

__all__ = [
    "contracts",
    "exceptions",
    "features",
    "indicators",
    "parquet_io",
    "performance",
    "registry",
    "runner",
    "schemas",
    "store",
]
__version__ = "0.11.0"
