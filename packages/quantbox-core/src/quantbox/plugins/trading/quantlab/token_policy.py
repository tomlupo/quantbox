"""
Token Policy Manager for Quantbox

Manages allowlist/denylist of tradable tokens with:
- Explicit allowlist mode (safer than quantlab's opt-out)
- New token detection and alerting
- Documented denial reasons

Usage:
    policy = TokenPolicy.from_config(config_path)
    allowed_symbols = policy.filter_allowed(rankings_df['symbol'].tolist())
    new_tokens = policy.detect_new_tokens(rankings_df, top_n=100)
"""

import yaml
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class TokenPolicy:
    """
    Token policy manager for controlling which tokens can be traded.
    
    Supports two modes:
    - "allowlist": Only trade tokens explicitly in allowed list (default, safer)
    - "denylist": Trade everything except denied tokens (like quantlab)
    """
    
    def __init__(
        self,
        mode: str = "allowlist",
        allowed: List[str] = None,
        denied: List[Dict] = None,
        alert_on_new: bool = True,
        top_n_monitor: int = 100,
        state_file: Path = None
    ):
        self.mode = mode
        self.allowed = set(allowed or [])
        self.denied = {d['symbol']: d.get('reason', 'No reason provided') for d in (denied or [])}
        self.alert_on_new = alert_on_new
        self.top_n_monitor = top_n_monitor
        self.state_file = state_file or Path("data/seen_tokens.json")
        self._seen_tokens = self._load_seen_tokens()
        
    @classmethod
    def from_config(cls, config_path: str) -> 'TokenPolicy':
        """Load token policy from YAML config file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        policy_config = config.get('token_policy', {})
        
        return cls(
            mode=policy_config.get('mode', 'allowlist'),
            allowed=policy_config.get('allowed', []),
            denied=policy_config.get('denied', []),
            alert_on_new=policy_config.get('alert_on_new', True),
            top_n_monitor=policy_config.get('top_n_monitor', 100),
            state_file=Path(config_path).parent.parent.parent / 'data' / 'seen_tokens.json'
        )
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'TokenPolicy':
        """Load token policy from config dictionary."""
        policy_config = config.get('token_policy', {})
        
        return cls(
            mode=policy_config.get('mode', 'allowlist'),
            allowed=policy_config.get('allowed', []),
            denied=policy_config.get('denied', []),
            alert_on_new=policy_config.get('alert_on_new', True),
            top_n_monitor=policy_config.get('top_n_monitor', 100),
        )
    
    def _load_seen_tokens(self) -> Set[str]:
        """Load previously seen tokens from state file."""
        try:
            if self.state_file.exists():
                with open(self.state_file) as f:
                    data = json.load(f)
                    return set(data.get('seen_tokens', []))
        except Exception as e:
            logger.warning(f"Could not load seen tokens: {e}")
        return set()
    
    def _save_seen_tokens(self):
        """Save seen tokens to state file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump({
                    'seen_tokens': sorted(self._seen_tokens),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save seen tokens: {e}")
    
    def is_allowed(self, symbol: str) -> bool:
        """Check if a symbol is allowed for trading."""
        symbol = symbol.upper()
        
        # Always deny if in denied list
        if symbol in self.denied:
            return False
        
        if self.mode == "allowlist":
            # Only allow if explicitly in allowed list
            return symbol in self.allowed
        else:
            # Denylist mode: allow if not in denied
            return True
    
    def get_denial_reason(self, symbol: str) -> Optional[str]:
        """Get the reason why a symbol is denied."""
        return self.denied.get(symbol.upper())
    
    def filter_allowed(self, symbols: List[str]) -> List[str]:
        """Filter a list of symbols to only allowed ones."""
        return [s for s in symbols if self.is_allowed(s)]
    
    def filter_denied(self, symbols: List[str]) -> List[Tuple[str, str]]:
        """Return list of denied symbols with their reasons."""
        return [(s, self.get_denial_reason(s) or "Not in allowlist") 
                for s in symbols if not self.is_allowed(s)]
    
    def detect_new_tokens(
        self, 
        rankings_df: pd.DataFrame, 
        top_n: int = None
    ) -> List[Dict]:
        """
        Detect new tokens that appeared in top N rankings.
        
        Returns list of dicts with token info:
            [{'symbol': 'NEW', 'rank': 42, 'market_cap': 5e9, 'is_allowed': False}]
        """
        if not self.alert_on_new:
            return []
        
        top_n = top_n or self.top_n_monitor
        
        # Get top N symbols from rankings
        current_top = set(rankings_df.head(top_n)['symbol'].str.upper().tolist())
        
        # Find new tokens
        new_tokens = current_top - self._seen_tokens
        
        if not new_tokens:
            return []
        
        # Get details for new tokens
        new_token_details = []
        for symbol in new_tokens:
            row = rankings_df[rankings_df['symbol'].str.upper() == symbol]
            if not row.empty:
                new_token_details.append({
                    'symbol': symbol,
                    'rank': int(row.iloc[0].get('rank', 0)),
                    'market_cap': float(row.iloc[0].get('market_cap', 0)),
                    'is_allowed': self.is_allowed(symbol),
                    'denial_reason': self.get_denial_reason(symbol)
                })
        
        # Update seen tokens
        self._seen_tokens.update(current_top)
        self._save_seen_tokens()
        
        # Sort by rank
        new_token_details.sort(key=lambda x: x['rank'])
        
        return new_token_details
    
    def format_new_token_alert(self, new_tokens: List[Dict]) -> str:
        """Format new token alert message."""
        if not new_tokens:
            return ""
        
        lines = ["🆕 **Nowe tokeny w top {0}:**".format(self.top_n_monitor)]
        
        for token in new_tokens:
            mcap_b = token['market_cap'] / 1e9
            status = "✅ allowed" if token['is_allowed'] else "❌ not allowed"
            reason = f" ({token['denial_reason']})" if token['denial_reason'] else ""
            lines.append(f"  • **{token['symbol']}** (rank #{token['rank']}, ${mcap_b:.1f}B) - {status}{reason}")
        
        lines.append("\nDodać do allowed? Odpowiedz: `/allow TOKEN` lub `/deny TOKEN reason`")
        
        return "\n".join(lines)
    
    def add_allowed(self, symbol: str):
        """Add a symbol to allowed list."""
        self.allowed.add(symbol.upper())
        # Remove from denied if present
        self.denied.pop(symbol.upper(), None)
    
    def add_denied(self, symbol: str, reason: str = "Manually denied"):
        """Add a symbol to denied list."""
        self.denied[symbol.upper()] = reason
        # Remove from allowed if present
        self.allowed.discard(symbol.upper())
    
    def get_stats(self) -> Dict:
        """Get policy statistics."""
        return {
            'mode': self.mode,
            'allowed_count': len(self.allowed),
            'denied_count': len(self.denied),
            'seen_tokens_count': len(self._seen_tokens),
            'alert_on_new': self.alert_on_new,
            'top_n_monitor': self.top_n_monitor
        }
    
    def __repr__(self):
        return f"TokenPolicy(mode={self.mode}, allowed={len(self.allowed)}, denied={len(self.denied)})"
