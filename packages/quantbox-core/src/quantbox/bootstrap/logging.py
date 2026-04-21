"""Stdlib logging factory — host-agnostic, no stack inspection.

A single entry point, :func:`configure_logging`, sets up the root
logger with a consistent file+console handler pair. Callers pass
explicit names — no ``inspect.stack()`` magic.

Design:

* Root-logger-owned handlers only. Plugins / libraries use
  ``logging.getLogger(__name__)`` and rely on propagation.
* Optional per-run file (``log_dir/<per_run_file>_<ts>.log``) for the
  "one file per process invocation" pattern robo uses.
* Callers pass a ``log_format`` string if they want embedded host
  metadata (hostname / user / ip); default format is the stdlib
  standard.
* Idempotent: repeated calls clear prior handlers so the active
  config is whatever the most-recent call set — useful in REPLs
  and test fixtures.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logging(
    log_dir: str | Path,
    level: str = "INFO",
    per_run_file: str | None = None,
    log_format: str | None = None,
    console: bool = True,
    file_level: str = "DEBUG",
) -> logging.Logger:
    """Configure the root logger; return it.

    Args:
        log_dir: Directory for log files. Created if missing.
        level: Console / overall log level (str, e.g. ``"INFO"``).
        per_run_file: Basename for the per-run log file. When set,
            creates ``log_dir/<per_run_file>_<YYYYMMDD-HHMMSS>.log``
            at ``file_level``. When ``None``, no file handler is
            attached.
        log_format: Format string passed to ``logging.Formatter``.
            Defaults to a simple ``asctime - name - levelname - message``.
            Hosts embedding hostname / user / ip should pass a
            pre-built format string here.
        console: Whether to attach a stderr ``StreamHandler``.
        file_level: Level for the file handler (defaults ``DEBUG``
            so the file captures everything even when the console
            is quieter).

    Returns:
        The configured root logger.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Clear existing handlers so repeat calls don't duplicate output.
    root.handlers.clear()

    fmt = logging.Formatter(log_format or DEFAULT_FORMAT)

    if per_run_file:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = log_dir / f"{per_run_file}_{ts}.log"
        fh = logging.FileHandler(log_file, mode="a", delay=True, encoding="utf-8")
        fh.setLevel(_level_as_int(file_level))
        fh.setFormatter(fmt)
        root.addHandler(fh)

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(_level_as_int(level))
        ch.setFormatter(fmt)
        root.addHandler(ch)

    return root


def _level_as_int(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)
