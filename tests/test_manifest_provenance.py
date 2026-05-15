"""Tests for installed-package provenance capture in run_manifest.json."""

from __future__ import annotations

import json

import pytest

from quantbox.runner import _installed_packages


def test_quantbox_self_captured() -> None:
    """quantbox itself must always be captured since we're running its tests."""
    info = _installed_packages(("quantbox",))
    assert "quantbox" in info
    assert "version" in info["quantbox"]


def test_unknown_package_silently_skipped() -> None:
    """A name that isn't installed must not raise — manifest must still build."""
    info = _installed_packages(("definitely-not-a-real-package-xyz123",))
    assert info == {}


def test_pep610_vcs_fields_when_present() -> None:
    """For a package installed from a git URL, commit_id + url should be captured."""
    import importlib.metadata as md

    # Look for any installed package that has direct_url.json with vcs_info.
    found_vcs = False
    for dist in md.distributions():
        try:
            durl_str = dist.read_text("direct_url.json")
        except FileNotFoundError:
            continue
        if not durl_str:
            continue
        try:
            durl = json.loads(durl_str)
        except ValueError:
            continue
        if (durl.get("vcs_info") or {}).get("commit_id"):
            pkg_name = dist.metadata["Name"]
            info = _installed_packages((pkg_name,))
            assert pkg_name in info
            assert "commit_id" in info[pkg_name]
            assert "url" in info[pkg_name]
            found_vcs = True
            break

    if not found_vcs:
        pytest.skip("No git-installed packages found in this environment to exercise the VCS path")


def test_editable_flag_when_present() -> None:
    """If quantbox itself is installed editable, editable=True should be captured."""
    import importlib.metadata as md

    try:
        dist = md.distribution("quantbox")
    except md.PackageNotFoundError:
        pytest.skip("quantbox not installed via distribution metadata")

    try:
        durl_str = dist.read_text("direct_url.json")
    except FileNotFoundError:
        pytest.skip("quantbox not installed via PEP 610 (no direct_url.json)")

    if not durl_str:
        pytest.skip("direct_url.json empty")

    durl = json.loads(durl_str)
    if not durl.get("dir_info", {}).get("editable"):
        pytest.skip("quantbox not installed editable")

    info = _installed_packages(("quantbox",))
    assert info["quantbox"].get("editable") is True
