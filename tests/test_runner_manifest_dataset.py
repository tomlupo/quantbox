"""Verify the manifest dict assembled in runner.py routes through the new
dataset block helpers and no longer carries data.source_identity."""

import inspect

from quantbox import runner


def test_runner_manifest_uses_dataset_block_helpers():
    src = inspect.getsource(runner)
    assert "_dataset_block(data)" in src
    assert "_run_capability_checks(data" in src


def test_runner_manifest_no_longer_emits_source_identity():
    src = inspect.getsource(runner)
    assert "source_identity" not in src
    assert "_dataset_evidence(data)" not in src


def test_runner_strict_mode_logic_is_present():
    src = inspect.getsource(runner)
    assert "strict mode rejects Tier-0" in src or "strict mode capability failures" in src
