"""Tests for the Hyperliquid data plugin's _post 429 retry/backoff."""

import pytest

from quantbox.plugins.datasources import hyperliquid_data_plugin as hl


class _FakeResp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_post_retries_on_429_then_succeeds(monkeypatch):
    """A 429 should back off and retry, not abort the run."""
    calls = []
    responses = [_FakeResp(429), _FakeResp(429), _FakeResp(200, {"ok": True})]

    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        return responses[len(calls) - 1]

    sleeps = []
    monkeypatch.setattr(hl.requests, "post", fake_post)
    monkeypatch.setattr(hl.time, "sleep", lambda s: sleeps.append(s))

    out = hl._post({"type": "meta"})
    assert out == {"ok": True}
    assert len(calls) == 3
    # exponential backoff: 1.0 then 2.0
    assert sleeps == [1.0, 2.0]


def test_post_raises_after_exhausting_retries(monkeypatch):
    """Persistent 429 raises rather than returning a poisoned/empty result."""
    monkeypatch.setattr(hl.requests, "post", lambda *a, **k: _FakeResp(429))
    monkeypatch.setattr(hl.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError):
        hl._post({"type": "meta"}, max_retries=3)


def test_post_backoff_is_capped(monkeypatch):
    """Backoff delay is capped at 30s."""
    monkeypatch.setattr(hl.requests, "post", lambda *a, **k: _FakeResp(429))
    sleeps = []
    monkeypatch.setattr(hl.time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError):
        hl._post({"type": "meta"}, max_retries=8)

    assert max(sleeps) <= 30.0
    assert sleeps[-1] == 30.0
