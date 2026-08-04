"""Cache-key behaviour for StrategyCache.config_hash.

The cache is incremental — already-computed dates are skipped and never
recomputed — so the key is the only thing standing between a change in
the input data and indefinitely-served stale weights. These tests pin
that contract, including the backward-compatibility guarantee that
omitting ``data_vintage`` reproduces the previous key.
"""

from __future__ import annotations

import pytest

from quantbox.cache.strategy_cache import StrategyCache

STRATEGY = {"function": "classical_asset_allocation", "parameters": {"target": 0.1}}
INDICATORS = {"covariance": {"method": "ewma_lw", "half_life_obs": [63, 52]}}


@pytest.fixture()
def cache(tmp_path):
    return StrategyCache(tmp_path / "strategies")


def test_omitting_vintage_is_backward_compatible(cache):
    """A caller that never passes data_vintage keeps its existing keys.

    Existing cache directories on disk are addressed by these hashes, so a
    change here silently orphans every cached result.
    """
    assert cache.config_hash(STRATEGY) == cache.config_hash(STRATEGY, data_vintage=None)
    assert cache.config_hash(STRATEGY, INDICATORS) == cache.config_hash(STRATEGY, INDICATORS, None)


def test_vintage_changes_the_key(cache):
    """The whole point: different data must not reuse cached weights."""
    before = cache.config_hash(STRATEGY, INDICATORS, {"securities": "aaaa"})
    after = cache.config_hash(STRATEGY, INDICATORS, {"securities": "bbbb"})
    assert before != after


def test_same_vintage_reuses_the_key(cache):
    """Unchanged data must still hit the cache, or it is worthless."""
    vintage = {"securities": "aaaa", "baskets": "bbbb"}
    assert cache.config_hash(STRATEGY, INDICATORS, vintage) == cache.config_hash(STRATEGY, INDICATORS, dict(vintage))


def test_vintage_key_order_does_not_matter(cache):
    """Dict ordering is not semantic — the hash must not depend on it."""
    a = cache.config_hash(STRATEGY, None, {"securities": "x", "baskets": "y"})
    b = cache.config_hash(STRATEGY, None, {"baskets": "y", "securities": "x"})
    assert a == b


def test_vintage_accepts_a_bare_string(cache):
    """A caller with a single stamp should not have to invent a dict."""
    assert cache.config_hash(STRATEGY, None, "vintage-2026-08-04") != cache.config_hash(
        STRATEGY, None, "vintage-2026-08-05"
    )


def test_vintage_is_independent_of_config(cache):
    """Config and vintage must both move the key, and not cancel out."""
    other_strategy = {**STRATEGY, "parameters": {"target": 0.2}}
    vintage = {"securities": "aaaa"}
    assert cache.config_hash(STRATEGY, None, vintage) != cache.config_hash(other_strategy, None, vintage)
    assert cache.config_hash(STRATEGY, None, vintage) != cache.config_hash(STRATEGY, None, {"securities": "zzzz"})
