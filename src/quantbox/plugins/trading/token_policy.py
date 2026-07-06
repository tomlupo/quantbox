"""Token Policy Manager — allowlist/denylist filter for tradeable tokens.

Controls which tokens the trading pipeline is allowed to trade. Supports
explicit allowlist mode (safer) and denylist mode (quantlab-style opt-out).
Also detects new tokens entering the top-N universe and fires Telegram alerts.

Usage:
    policy = TokenPolicy.from_config(config_path)
    allowed_symbols = policy.filter_allowed(rankings_df['symbol'].tolist())
    new_tokens = policy.detect_new_tokens(rankings_df, top_n=100)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _book_key_from_config(config: dict[str, Any]) -> str | None:
    """Derive the per-book key from a live config's ``notify.books`` section.

    Returns the sanitized first book key, or ``None`` when no book is declared
    (backtest/research configs) — callers then fall back to the shared path.
    Mirrors quantbox-live's ``book_key_from_config`` (per-book reports/snapshots,
    quantbox-live#32/#38) so seen-token state is namespaced the same way.
    """
    books = ((config.get("notify") or {}).get("books")) or {}
    first = next(iter(books), None)
    if not first:
        return None
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(first))


class TokenPolicy:
    """Allowlist/denylist filter for tradeable tokens.

    Supports two modes:
    - ``"allowlist"``: only trade tokens explicitly in allowed list (default, safer)
    - ``"denylist"``: trade everything except denied tokens
    """

    def __init__(
        self,
        mode: str = "allowlist",
        allowed: list[str] | None = None,
        denied: list[dict[str, Any]] | None = None,
        alert_on_new: bool = True,
        top_n_monitor: int = 100,
        state_file: Path | None = None,
        legacy_state_file: Path | None = None,
    ):
        self.mode = mode
        self.allowed: set[str] = set(allowed or [])
        self.denied: dict[str, str] = {d["symbol"]: d.get("reason", "No reason provided") for d in (denied or [])}
        self.alert_on_new = alert_on_new
        self.top_n_monitor = top_n_monitor
        self.state_file = state_file or Path("data/seen_tokens.json")
        # One-time migration seed: when a per-book state file does not exist yet
        # but the legacy shared file does, seed from it so a book does NOT
        # re-alert its entire universe on the first per-book run. The shared file
        # is the cross-book union, so seeding is non-regressive (it preserves
        # today's over-suppression); books diverge correctly from the next save.
        self.legacy_state_file = legacy_state_file
        self._seen_tokens = self._load_seen_tokens()

    @classmethod
    def from_config(cls, config_path: str, book_key: str | None = None) -> TokenPolicy:
        """Load token policy from YAML config file.

        ``book_key`` (threaded from the pipeline's run params) is authoritative
        when supplied; otherwise it is derived from the config's ``notify.books``
        (mirrors quantbox-live#32/#38). A standalone token-policy file usually has
        no ``notify.books``, so the explicit arg is what namespaces that path.
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)
        policy_config = config.get("token_policy", {})
        data_dir = Path(config_path).parent.parent.parent / "data"
        legacy_state_file = data_dir / "seen_tokens.json"
        # Namespace seen-token state per book so a new listing in one book's
        # universe is not silently suppressed by another book that already saw
        # it (issue quantbox#86). Falls back to the shared path for configs with
        # no book declared and no explicit key (backtest/research).
        resolved_key = book_key if book_key and book_key != "default" else _book_key_from_config(config)
        if resolved_key:
            from quantbox.reconciliation import safe_book_key

            seen_dir = data_dir / "seen_tokens"
            safe = safe_book_key(resolved_key, seen_dir)
            state_file = seen_dir / f"{safe}.json"
        else:
            state_file = legacy_state_file
        return cls(
            mode=policy_config.get("mode", "allowlist"),
            allowed=policy_config.get("allowed", []),
            denied=policy_config.get("denied", []),
            alert_on_new=policy_config.get("alert_on_new", True),
            top_n_monitor=policy_config.get("top_n_monitor", 100),
            state_file=state_file,
            legacy_state_file=legacy_state_file if state_file != legacy_state_file else None,
        )

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
        book_key: str | None = None,
        data_dir: Path | str | None = None,
    ) -> TokenPolicy:
        """Load token policy from config dictionary.

        The inline-config path (used by the trading pipeline) has no YAML file to
        anchor its seen-token store, so without ``book_key`` it previously fell
        through to a single cwd-relative ``data/seen_tokens.json`` shared across
        every book (issue #86, the worse of the two entry points). Passing
        ``book_key`` namespaces the store per book: ``<data_dir>/seen_tokens/
        <book_key>.json``. ``book_key`` is validated as a single safe path segment
        (reusing the reconciliation ledger's ``safe_book_key`` — reject, not
        silently sanitize) because it is config-controlled and names a path.
        """
        policy_config = config.get("token_policy", {})
        base_dir = Path(data_dir) if data_dir is not None else Path("data")
        legacy_state_file = base_dir / "seen_tokens.json"
        state_file: Path | None = None
        legacy_for_migration: Path | None = None
        if book_key:
            from quantbox.reconciliation import safe_book_key

            seen_dir = base_dir / "seen_tokens"
            safe = safe_book_key(book_key, seen_dir)
            state_file = seen_dir / f"{safe}.json"
            legacy_for_migration = legacy_state_file if state_file != legacy_state_file else None
        return cls(
            mode=policy_config.get("mode", "allowlist"),
            allowed=policy_config.get("allowed", []),
            denied=policy_config.get("denied", []),
            alert_on_new=policy_config.get("alert_on_new", True),
            top_n_monitor=policy_config.get("top_n_monitor", 100),
            state_file=state_file,
            legacy_state_file=legacy_for_migration,
        )

    def _load_seen_tokens(self) -> set[str]:
        try:
            if self.state_file.exists():
                with open(self.state_file) as f:
                    return set(json.load(f).get("seen_tokens", []))
            # Per-book file not yet written: seed once from the legacy shared file.
            if self.legacy_state_file and self.legacy_state_file.exists():
                with open(self.legacy_state_file) as f:
                    seeded = set(json.load(f).get("seen_tokens", []))
                logger.info(
                    "Seeded %d seen tokens for %s from legacy %s",
                    len(seeded),
                    self.state_file,
                    self.legacy_state_file,
                )
                return seeded
        except Exception as exc:
            logger.warning("Could not load seen tokens: %s", exc)
        return set()

    def _save_seen_tokens(self) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(
                    {"seen_tokens": sorted(self._seen_tokens), "last_updated": datetime.now().isoformat()},
                    f,
                    indent=2,
                )
        except Exception as exc:
            logger.warning("Could not save seen tokens: %s", exc)

    def is_allowed(self, symbol: str) -> bool:
        symbol = symbol.upper()
        if symbol in self.denied:
            return False
        if self.mode == "allowlist":
            return symbol in self.allowed
        return True

    def get_denial_reason(self, symbol: str) -> str | None:
        return self.denied.get(symbol.upper())

    def filter_allowed(self, symbols: list[str]) -> list[str]:
        return [s for s in symbols if self.is_allowed(s)]

    def filter_denied(self, symbols: list[str]) -> list[tuple[str, str]]:
        return [(s, self.get_denial_reason(s) or "Not in allowlist") for s in symbols if not self.is_allowed(s)]

    def detect_new_tokens(self, rankings_df: pd.DataFrame, top_n: int | None = None) -> list[dict[str, Any]]:
        """Detect new tokens that appeared in top-N rankings since last run."""
        if not self.alert_on_new:
            return []
        top_n = top_n or self.top_n_monitor
        current_top = set(rankings_df.head(top_n)["symbol"].str.upper().tolist())
        new_symbols = current_top - self._seen_tokens
        if not new_symbols:
            return []
        details = []
        for symbol in new_symbols:
            row = rankings_df[rankings_df["symbol"].str.upper() == symbol]
            if not row.empty:
                details.append(
                    {
                        "symbol": symbol,
                        "rank": int(row.iloc[0].get("rank", 0)),
                        "market_cap": float(row.iloc[0].get("market_cap", 0)),
                        "is_allowed": self.is_allowed(symbol),
                        "denial_reason": self.get_denial_reason(symbol),
                    }
                )
        self._seen_tokens.update(current_top)
        self._save_seen_tokens()
        details.sort(key=lambda x: x["rank"])
        return details

    def format_new_token_alert(self, new_tokens: list[dict[str, Any]]) -> str:
        if not new_tokens:
            return ""
        lines = [f"\U0001f195 **New tokens in top {self.top_n_monitor}:**"]
        for token in new_tokens:
            mcap_b = token["market_cap"] / 1e9
            status = "✅ allowed" if token["is_allowed"] else "❌ not allowed"
            reason = f" ({token['denial_reason']})" if token["denial_reason"] else ""
            lines.append(f"  • **{token['symbol']}** (rank #{token['rank']}, ${mcap_b:.1f}B) - {status}{reason}")
        lines.append("\nAdd to allowed? `/allow TOKEN` or `/deny TOKEN reason`")
        return "\n".join(lines)

    def add_allowed(self, symbol: str) -> None:
        self.allowed.add(symbol.upper())
        self.denied.pop(symbol.upper(), None)

    def add_denied(self, symbol: str, reason: str = "Manually denied") -> None:
        self.denied[symbol.upper()] = reason
        self.allowed.discard(symbol.upper())

    def get_stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "allowed_count": len(self.allowed),
            "denied_count": len(self.denied),
            "seen_tokens_count": len(self._seen_tokens),
            "alert_on_new": self.alert_on_new,
            "top_n_monitor": self.top_n_monitor,
        }

    def __repr__(self) -> str:
        return f"TokenPolicy(mode={self.mode}, allowed={len(self.allowed)}, denied={len(self.denied)})"
