"""Build the deterministic synthetic-panel fixture used by canonical reproductions.

The fixture is committed to the repo so canonical tests don't depend on
network. Re-run this script when the fixture spec genuinely needs to
change — and bump the goldens via ``regen_goldens.py`` immediately after.

Spec:
- 5 symbols (BTC + 4 fictional alts), 252 daily bars (one trading year)
- Seeded geometric Brownian motion + per-symbol drift + per-symbol vol
- One injected regime flip (~day 120) to exercise trend + carry logic
- Volume panel scales with realised volatility (gives volume-rank
  trend-catcher something to filter on)

Usage:
    uv run python cookbook/canonical/build_fixture.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "fixture.parquet"
VOL_OUT = Path(__file__).resolve().parent / "fixture_volume.parquet"

SEED = 42
N_DAYS = 252
SYMBOLS = ["BTC", "ALT1", "ALT2", "ALT3", "ALT4"]
DRIFTS = [0.0006, 0.0008, 0.0004, -0.0002, 0.0010]
VOLS = [0.030, 0.045, 0.040, 0.050, 0.055]


def main() -> None:
    rng = np.random.default_rng(SEED)
    idx = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")

    prices = pd.DataFrame(index=idx, columns=SYMBOLS, dtype=float)
    volume = pd.DataFrame(index=idx, columns=SYMBOLS, dtype=float)

    for sym, mu, sigma in zip(SYMBOLS, DRIFTS, VOLS, strict=True):
        # Geometric Brownian motion log-returns
        log_rets = rng.normal(loc=mu, scale=sigma, size=N_DAYS)
        # Inject a regime flip ~day 120 — flip drift sign for a 30-day stretch
        # so trend strategies have something to bite into.
        log_rets[120:150] *= -1.0
        price_path = 100.0 * np.exp(np.cumsum(log_rets))
        prices[sym] = price_path
        # Volume scales with realised rolling vol (richer when market's moving)
        rolling_vol = pd.Series(log_rets).rolling(20, min_periods=1).std().fillna(sigma)
        volume[sym] = 1e6 * (1 + 5 * rolling_vol).values

    prices.index.name = "date"
    volume.index.name = "date"
    prices.to_parquet(OUT)
    volume.to_parquet(VOL_OUT)
    print(f"Wrote {OUT}: {prices.shape}, mean = {prices.values.mean():.2f}")
    print(f"Wrote {VOL_OUT}: {volume.shape}")


if __name__ == "__main__":
    main()
