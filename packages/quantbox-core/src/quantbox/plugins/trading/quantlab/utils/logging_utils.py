# logging_utils.py
from __future__ import annotations

import inspect
import logging
import os
import re
import socket
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Pattern, List
import functools
import time
import getpass
import pandas as pd
from .paths import PROJECT_ROOT

# =========================
# Globals & config
# =========================
_LOGGING_CONFIGURED = False
_FILE_HANDLER_ATTACHED = False
_LOG_FILE_PATH: Optional[Path] = None

# System metadata (safe lookups)
hostname = socket.gethostname()
try:
    ip_address = socket.gethostbyname(hostname)
except Exception:
    ip_address = "unknown"
username = getpass.getuser()

# Formats
LOG_FORMAT = (
    f"%(asctime)s - {hostname} - {ip_address} - {username} - "
    "%(name)s - %(levelname)s - %(message)s"
)
CONSOLE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Output dir
LOG_DIR = Path(PROJECT_ROOT, "output/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# External families that should be verbose (console + file)
EXTERNAL_VERBOSE_FAMILIES: tuple[str, ...] = tuple(
    s.strip() for s in os.getenv("LOG_VERBOSE_EXTERNALS", "luigi").split(",") if s.strip()
)

# =========================
# Helpers
# =========================
def _detect_package_root() -> str:
    env_name = os.getenv("LOG_PACKAGE")
    if env_name:
        return env_name.split(".")[0]
    main = logging.__dict__.get("__main__") or __import__("sys").modules.get("__main__")
    if main:
        pkg = getattr(main, "__package__", None)
        if pkg:
            return pkg.split(".")[0]
        f = getattr(main, "__file__", None)
        if f:
            return Path(f).resolve().parent.name
    return "app"

def _make_session_logfile(prefix: str, ts: Optional[str] = None, log_dir: Path = LOG_DIR) -> Path:
    ts = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{prefix}_{ts}.log"

def get_log_file_path() -> Optional[Path]:
    return _LOG_FILE_PATH

def get_caller_script_name() -> str:
    stack = inspect.stack()
    caller_frame = stack[2] if len(stack) > 2 else stack[-1]
    return os.path.basename(caller_frame.filename)

def _ensure_console(logger: logging.Logger, fmt: str = CONSOLE_FORMAT) -> None:
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter(fmt))
        logger.addHandler(ch)

# =========================
# Configure (console only)
# =========================
def _configure_base_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    # Root: externals quiet by default
    root = logging.getLogger()
    root.setLevel(logging.CRITICAL)

    # Internal package → console
    pkg = _detect_package_root()
    internal = logging.getLogger(pkg)
    if internal.level == logging.NOTSET:
        internal.setLevel(logging.DEBUG)
    _ensure_console(internal)

    # Whitelisted externals → console
    for fam in EXTERNAL_VERBOSE_FAMILIES:
        if not fam:
            continue
        ext = logging.getLogger(fam)
        if ext.level == logging.NOTSET:
            ext.setLevel(logging.DEBUG)
        _ensure_console(ext)

    _LOGGING_CONFIGURED = True

# =========================
# File handler (root only)
# =========================
def _attach_file_handler(prefix: str, ts: Optional[str] = None, log_dir: Path = LOG_DIR) -> Path:
    """Attach a rotating file handler to ROOT. Removes previous file handlers first."""
    global _LOG_FILE_PATH

    _LOG_FILE_PATH = _make_session_logfile(prefix, ts, log_dir)

    # Remove existing file handlers before adding new one
    root = logging.getLogger()
    existing_handlers = root.handlers.copy()
    for handler in existing_handlers:
        if isinstance(handler, (RotatingFileHandler, logging.FileHandler)):
            root.removeHandler(handler)
            handler.close()  # Ensure file is properly closed

    fh = RotatingFileHandler(
        _LOG_FILE_PATH,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,  # create on first write
    )
    fh.setFormatter(logging.Formatter(LOG_FORMAT))

    root.addHandler(fh)
    root.log_file = _LOG_FILE_PATH  # convenience

    return _LOG_FILE_PATH

# =========================
# Public: get_logger
# =========================
def get_logger(
    name: Optional[str] = None,
    *,
    file_handler: bool = False,
    timestamp: Optional[str] = None,
    log_dir: Path = LOG_DIR,
    root_logger: bool = False,
) -> logging.Logger:
    """Get a logger (auto-configures on first call)."""

    if root_logger:
        import warnings
        warnings.warn("root_logger=True is deprecated. Use just get_logger() without any arguments.")

    if not _LOGGING_CONFIGURED:
        _configure_base_logging()

    # Default to caller module
    if name is None:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            mod = inspect.getmodule(frame.f_back)
            name = mod.__name__ if (mod and mod.__name__) else _detect_package_root()
        else:
            name = _detect_package_root()

    # Attach ROOT file handler (once)
    if file_handler:
        prefix = name.split(".")[0] if "." in name else name
        _attach_file_handler(prefix, timestamp, log_dir)

    logger = logging.getLogger(name)

    # Levels: ensure DEBUG for app/external verbose families; others inherit
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.DEBUG)

    # Console: ensure one stream handler (no duplicates)
    _ensure_console(logger)

    # Propagate to ROOT so file writes happen via root handler
    logger.propagate = True

    # Convenience attribute for callers
    if _FILE_HANDLER_ATTACHED and _LOG_FILE_PATH is not None:
        try:
            logger.log_file = _LOG_FILE_PATH
        except Exception:
            pass

    return logger


# =========================
# Decorators & utilities
# =========================
def log_execution_time(func):
    """Decorator: INFO log the call + execution time using 'timeit' logger."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(name="timeit")
        logger.info(f"Function {func.__module__}.{func.__name__} called")
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            logger.info(f"Function {func.__module__}.{func.__name__} executed in {time.time() - start:.3f}s")
    return wrapper

# =========================
# Parser (format-driven)
# =========================
_FIELD_RE = re.compile(r"%\((?P<field>\w+)\)s")
_TOKEN_PATTERNS: Dict[str, str] = {
    "asctime":   r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}",
    "levelname": r"[A-Z]+",
    "message":   r".*",   # greedy to end of line
    # Other fields default to non-greedy .+?
}

def _regex_from_format(fmt: str, token_patterns: Optional[Dict[str, str]] = None) -> Pattern[str]:
    token_patterns = token_patterns or _TOKEN_PATTERNS
    parts, last = [], 0
    for m in _FIELD_RE.finditer(fmt):
        parts.append(re.escape(fmt[last:m.start()]))
        field = m.group("field")
        pat = token_patterns.get(field, r".+?")
        if field == "message":
            pat = r".*"
        parts.append(f"(?P<{field}>{pat})")
        last = m.end()
    parts.append(re.escape(fmt[last:]))
    return re.compile("^" + "".join(parts) + "$")

def _detect_log_format_from_handlers() -> Optional[str]:
    # Prefer root handlers (our RotatingFileHandler lives here)
    for h in logging.getLogger().handlers:
        fmt = getattr(getattr(h, "formatter", None), "_fmt", None)
        if fmt:
            return fmt
    return None

def _parse_one_file(dir_path: Path, fname: str, line_re: Pattern[str]) -> pd.DataFrame:
    rows = []
    with open(dir_path / fname, "r", encoding="utf-8", errors="replace") as f:
        entry = None
        for raw in f:
            line = raw.rstrip("\n")
            m = line_re.match(line)
            if m:
                if entry:
                    rows.append(entry)
                entry = m.groupdict()
                entry["filename"] = fname
            else:
                if entry:
                    entry["message"] = entry.get("message", "") + "\n" + line
        if entry:
            rows.append(entry)
    df = pd.DataFrame(rows)
    if not df.empty and "asctime" in df.columns:
        dt = pd.to_datetime(df["asctime"], format="%Y-%m-%d %H:%M:%S,%f", errors="coerce")
        if dt.isna().any():
            dt = pd.to_datetime(df["asctime"], errors="coerce")
        df["datetime"] = dt
    return df

def parse_log_files(
    log_dir_path: str,
    *,
    log_format: Optional[str] = None,
    log_file_name: Optional[str] = None,
) -> pd.DataFrame:
    """Parse .log files in a directory (optionally a single file) using the provided or detected format."""
    fmt = log_format or _detect_log_format_from_handlers() or LOG_FORMAT
    line_re = _regex_from_format(fmt)

    rows: List[pd.DataFrame] = []
    for file_name in os.listdir(log_dir_path):
        if not file_name.endswith(".log"):
            continue
        if log_file_name and file_name != log_file_name:
            continue
        rows.append(_parse_one_file(Path(log_dir_path), file_name, line_re))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def parse_session_logs(
    log_file_path = None,
    include_backups: bool = True,
    log_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Parse the current session (active file + rotated siblings) into one DataFrame,
    oldest → newest. Auto-detects format if not provided.
    """
    if log_file_path is None:
        log_file_path = _LOG_FILE_PATH
    base_path = Path(log_file_path)

    fmt = log_format or _detect_log_format_from_handlers() or LOG_FORMAT
    line_re = _regex_from_format(fmt)

    dir_path = base_path.parent
    base_name = base_path.name  # e.g. myapp_YYYYmmdd_HHMMSS.log

    # Collect base + .N siblings
    family = []
    suf_re = re.compile(rf"^{re.escape(base_name)}(?:\.(\d+))?$")
    for p in dir_path.iterdir():
        m = suf_re.match(p.name)
        if not m:
            continue
        if p.name == base_name:
            family.append((-1, p.name))  # base (newest)
        elif include_backups:
            family.append((int(m.group(1)), p.name))

    # Oldest first (highest suffix) -> newest last (base)
    family.sort(reverse=True)

    frames = [_parse_one_file(dir_path, fname, line_re) for _, fname in family]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()