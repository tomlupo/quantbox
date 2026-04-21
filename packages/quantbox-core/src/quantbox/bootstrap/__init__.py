"""Bootstrap helpers — host-agnostic config loading + logging patterns.

These are the small pieces a host project (robo, dm-evo, ad-hoc research)
keeps re-implementing: a yaml + optional-profile-override loader, and a
stdlib logging factory. Quantbox owns the *pattern*; each host brings its
own schema class (pydantic, dataclass, TypedDict, anything with a
``model_validate`` method) and its own log-dir convention.

No dependency on pydantic — ``load_config`` just calls
``schema_cls.model_validate(data)`` duck-typed.
"""

from quantbox.bootstrap.config import load_config, load_yaml_merged
from quantbox.bootstrap.logging import configure_logging

__all__ = ["configure_logging", "load_config", "load_yaml_merged"]
