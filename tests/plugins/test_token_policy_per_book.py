"""Per-book seen-token state namespacing (quantbox#86).

seen_tokens state must be namespaced per book so a token new to one book's
universe is not silently suppressed by another book that already saw it.
"""

import json
from pathlib import Path

import pytest
import yaml

from quantbox.plugins.trading.token_policy import TokenPolicy, _book_key_from_config


def _write_config(tmp_path: Path, book: str | None) -> Path:
    # from_config resolves data/ as config_path.parent.parent.parent/data
    cfg_dir = tmp_path / "repo" / "configs" / "live"
    cfg_dir.mkdir(parents=True)
    cfg: dict = {"token_policy": {"mode": "allowlist", "allowed": ["BTC"]}}
    if book is not None:
        cfg["notify"] = {"books": {book: "live"}}
    path = cfg_dir / "book.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_book_key_derivation():
    assert _book_key_from_config({"notify": {"books": {"carver-HL": "live"}}}) == "carver-HL"
    assert _book_key_from_config({"notify": {"books": {"a/b c": "live"}}}) == "a_b_c"
    assert _book_key_from_config({}) is None
    assert _book_key_from_config({"notify": {"books": {}}}) is None


def test_state_file_namespaced_per_book(tmp_path):
    cfg_a = _write_config(tmp_path / "a", "carver-HL")
    cfg_b = _write_config(tmp_path / "b", "crypto-trend-kraken")
    pa = TokenPolicy.from_config(str(cfg_a))
    pb = TokenPolicy.from_config(str(cfg_b))
    assert pa.state_file.name == "carver-HL.json"
    assert pb.state_file.name == "crypto-trend-kraken.json"
    assert pa.state_file != pb.state_file


def test_no_book_falls_back_to_shared_path(tmp_path):
    cfg = _write_config(tmp_path, None)
    p = TokenPolicy.from_config(str(cfg))
    assert p.state_file.name == "seen_tokens.json"
    assert p.legacy_state_file is None


def test_two_books_do_not_share_seen_state(tmp_path):
    cfg_a = _write_config(tmp_path / "a", "carver-HL")
    cfg_b = _write_config(tmp_path / "b", "crypto-trend-kraken")
    pa = TokenPolicy.from_config(str(cfg_a))
    pa._seen_tokens = {"SOL"}
    pa._save_seen_tokens()
    # A token seen by book A must remain unseen by book B.
    pb = TokenPolicy.from_config(str(cfg_b))
    assert "SOL" not in pb._seen_tokens


def test_migration_seeds_from_legacy_shared_file(tmp_path):
    cfg = _write_config(tmp_path, "carver-HL")
    data_dir = cfg.parent.parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "seen_tokens.json").write_text(json.dumps({"seen_tokens": ["BTC", "ETH"]}))
    p = TokenPolicy.from_config(str(cfg))
    # First per-book run seeds from the legacy union → no whole-universe re-alert.
    assert p._seen_tokens == {"BTC", "ETH"}


# --- from_dict path (the inline pipeline path; issue #86 "worse" entry point) ---


def test_from_dict_namespaced_per_book(tmp_path):
    cfg = {"token_policy": {"mode": "allowlist", "allowed": ["BTC"]}}
    pa = TokenPolicy.from_dict(cfg, book_key="carver-HL", data_dir=tmp_path)
    pb = TokenPolicy.from_dict(cfg, book_key="crypto-trend-kraken", data_dir=tmp_path)
    assert pa.state_file != pb.state_file
    assert pa.state_file.name == "carver-HL.json"
    assert pb.state_file.name == "crypto-trend-kraken.json"


def test_from_dict_two_books_do_not_share_seen_state(tmp_path):
    cfg = {"token_policy": {"mode": "allowlist", "allowed": ["BTC"]}}
    pa = TokenPolicy.from_dict(cfg, book_key="carver-HL", data_dir=tmp_path)
    pa._seen_tokens = {"SOL"}
    pa._save_seen_tokens()
    # A token seen by book A must NOT suppress book B's alert for it.
    pb = TokenPolicy.from_dict(cfg, book_key="crypto-trend-kraken", data_dir=tmp_path)
    assert "SOL" not in pb._seen_tokens


def test_from_dict_migration_seeds_from_legacy(tmp_path):
    (tmp_path / "seen_tokens.json").write_text(json.dumps({"seen_tokens": ["BTC", "ETH"]}))
    cfg = {"token_policy": {"mode": "allowlist", "allowed": ["BTC"]}}
    p = TokenPolicy.from_dict(cfg, book_key="carver-HL", data_dir=tmp_path)
    assert p._seen_tokens == {"BTC", "ETH"}


@pytest.mark.parametrize("bad_key", ["../x", "a/b", "..", "a\\b"])
def test_from_dict_rejects_unsafe_book_key(tmp_path, bad_key):
    cfg = {"token_policy": {"mode": "allowlist", "allowed": ["BTC"]}}
    with pytest.raises(ValueError):
        TokenPolicy.from_dict(cfg, book_key=bad_key, data_dir=tmp_path)


def test_from_config_rejects_unsafe_explicit_book_key(tmp_path):
    cfg = _write_config(tmp_path, None)
    with pytest.raises(ValueError):
        TokenPolicy.from_config(str(cfg), book_key="../escape")
