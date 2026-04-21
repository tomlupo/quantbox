"""Tests for ``quantbox.bootstrap.configure_logging``."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from quantbox.bootstrap import configure_logging


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Clear handlers after each test so cross-test pollution doesn't
    mask issues."""
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        h.close()
        root.removeHandler(h)


def test_configure_logging_no_file(tmp_path: Path) -> None:
    logger = configure_logging(tmp_path, level="INFO")
    assert logger is logging.getLogger()
    # Only a console handler (no file handler, no per_run_file).
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    stream_handlers = [
        h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    assert file_handlers == []
    assert len(stream_handlers) == 1
    assert stream_handlers[0].level == logging.INFO


def test_configure_logging_per_run_file(tmp_path: Path) -> None:
    configure_logging(tmp_path, per_run_file="robo", level="WARNING")
    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    fh = file_handlers[0]
    assert fh.level == logging.DEBUG  # file_level default
    log_path = Path(fh.baseFilename)
    assert log_path.parent == tmp_path
    assert log_path.name.startswith("robo_")
    assert log_path.suffix == ".log"


def test_configure_logging_idempotent(tmp_path: Path) -> None:
    configure_logging(tmp_path, per_run_file="a")
    configure_logging(tmp_path, per_run_file="b")
    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    # Second call replaced the first handler set entirely.
    assert len(file_handlers) == 1
    assert Path(file_handlers[0].baseFilename).name.startswith("b_")


def test_configure_logging_custom_format(tmp_path: Path) -> None:
    fmt = "CUSTOM-%(levelname)s-%(message)s"
    configure_logging(tmp_path, log_format=fmt)
    root = logging.getLogger()
    for h in root.handlers:
        assert h.formatter._fmt == fmt


def test_configure_logging_writes_to_file(tmp_path: Path) -> None:
    configure_logging(tmp_path, per_run_file="probe", console=False)
    root = logging.getLogger()
    root.info("hello-quantbox")
    # Flush handler buffers
    for h in root.handlers:
        h.flush()
    log_files = list(tmp_path.glob("probe_*.log"))
    assert len(log_files) == 1
    content = log_files[0].read_text(encoding="utf-8")
    assert "hello-quantbox" in content
