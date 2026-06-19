"""Regression tests for FuturesPaperBroker state-load hardening."""

from __future__ import annotations

import json
import math

from quantbox.plugins.broker.futures_paper import FuturesPaperBroker


def test_load_state_rejects_nonfinite_margin(tmp_path):
    """A persisted NaN/inf margin_balance (or fees) must NOT clobber the seed.

    Mirror of the spot-broker regression: a non-finite persisted account value
    loaded over the healthy config seed made portfolio value NaN, tripping the
    rebalancer's "Portfolio value is zero or negative" guard. The read side now
    rejects non-finite numeric fields and falls back to the dataclass default.
    """
    state_file = tmp_path / "state.json"
    # json.dumps emits bare NaN/Infinity tokens, which json.loads parses back
    # to float('nan')/float('inf') (Python extension to strict JSON).
    state_file.write_text(
        json.dumps(
            {
                "margin_balance": float("nan"),
                "positions": {"BTC": -1.5},
                "entry_prices": {"BTC": 60_000.0},
                "cumulative_fees": float("inf"),
                "fill_log": [],
            }
        )
    )

    broker = FuturesPaperBroker(margin_balance=25_000.0, state_file=str(state_file))

    assert math.isfinite(broker.margin_balance)
    assert broker.margin_balance == 25_000.0  # fell back to the configured seed
    assert math.isfinite(broker._cumulative_fees)
    assert broker._cumulative_fees == 0.0
    # Finite fields still load normally.
    assert broker.positions == {"BTC": -1.5}
    assert broker.entry_prices == {"BTC": 60_000.0}
