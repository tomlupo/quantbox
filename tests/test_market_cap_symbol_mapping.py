"""One symbol mapping for live and backtest market cap (TOM-1419, replay of TOM-1411).

The incident: on Hyperliquid, ``kPEPE`` quotes 1000 PEPE. The backtest path
(``load_pit_market_cap``) mapped ``kPEPE`` to PEPE's cap; the live provider
(``MarketCapProvider.estimate_market_cap``) matched by ``ticker.upper()`` only, so
``kPEPE`` got NaN, filled to 0 by the screen, and fell out of ``top_by_mcap``.
The same config picked a different universe live than in its backtest.
Both paths now use :func:`quantbox.market_cap.map_symbol`.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from quantbox.market_cap import HL_K_PREFIX_UNITS, MarketCapProvider, load_pit_market_cap, map_symbol
from quantbox.universe import select_universe

_IDX = pd.date_range("2026-01-01", periods=4, freq="D")


def _provider(rows: list[dict]) -> MarketCapProvider:
    prov = MarketCapProvider(cache_dir=None)
    rankings = pd.DataFrame(rows).assign(rank=range(1, len(rows) + 1), fetch_timestamp=pd.Timestamp.now())
    prov.fetch_rankings = lambda: rankings  # type: ignore[method-assign]
    return prov


def test_kpepe_gets_pepes_market_cap_in_the_live_provider():
    prov = _provider(
        [
            {"symbol": "BTC", "market_cap": 1.2e12, "total_volume": 4e10, "circulating_supply": 2e7},
            {"symbol": "PEPE", "market_cap": 4.0e9, "total_volume": 1e9, "circulating_supply": 4.2e14},
        ]
    )
    # kPEPE's price is 1000x PEPE's; the reported cap anchors at the latest bar.
    prices = pd.DataFrame({"BTC": 60000.0, "kPEPE": [0.009, 0.010, 0.011, 0.012]}, index=_IDX)

    mc = prov.estimate_market_cap(prices, pd.DataFrame())

    assert (mc["kPEPE"] > 0).all()
    assert mc["kPEPE"].iloc[-1] == pytest.approx(4.0e9)
    assert prov.estimate_aggregate_volume(prices)["kPEPE"].iloc[-1] == pytest.approx(1e9)


def test_a_k_ticker_on_the_supply_layers_is_priced_per_base_coin():
    # No reported cap: the L2 (supply) and L3 (hardcoded) layers multiply price by
    # a supply counted in BASE coins, so kSHIB's 1000x price is divided back.
    prov = _provider([{"symbol": "PEPE", "market_cap": 0.0, "total_volume": 0.0, "circulating_supply": 4.2e14}])
    prices = pd.DataFrame({"kPEPE": 0.012, "kSHIB": 0.02, "SHIB": 0.00002}, index=_IDX)

    mc = prov.estimate_market_cap(prices, pd.DataFrame())

    assert mc["kPEPE"].iloc[-1] == pytest.approx(0.012 / 1000 * 4.2e14)
    assert mc["kSHIB"].iloc[-1] == pytest.approx(mc["SHIB"].iloc[-1])


def test_the_live_screen_keeps_kpepe_in_top_by_mcap():
    prov = _provider(
        [
            {"symbol": "BTC", "market_cap": 1.2e12, "total_volume": 4e10, "circulating_supply": 2e7},
            {"symbol": "PEPE", "market_cap": 4.0e9, "total_volume": 1e9, "circulating_supply": 4.2e14},
            {"symbol": "SMOL", "market_cap": 1.0e8, "total_volume": 5e8, "circulating_supply": 1e9},
        ]
    )
    prices = pd.DataFrame({"BTC": 60000.0, "kPEPE": 0.012, "SMOL": 0.1}, index=_IDX)
    volume = pd.DataFrame(1.0, index=_IDX, columns=prices.columns)

    mask = select_universe(
        prices,
        volume,
        market_cap=prov.estimate_market_cap(prices, volume),
        top_by_mcap=2,
        top_by_volume=2,
        exclude_tickers=[],
        screen_volume=prov.estimate_aggregate_volume(prices),
    )

    assert mask.iloc[-1].to_dict() == {"BTC": 1.0, "kPEPE": 1.0, "SMOL": 0.0}


def test_live_and_backtest_map_the_same_tickers(monkeypatch, tmp_path):
    curated = pd.DataFrame({"BTC": [1.2e12], "PEPE": [4.0e9], "SHIB": [1.0e10]}, index=pd.DatetimeIndex(["2025-12-31"]))
    curated.to_parquet(tmp_path / "market_cap.parquet")
    builtins = types.ModuleType("quantbox_datasets.builtins")
    builtins.crypto_spot_daily = lambda: types.SimpleNamespace(_ds_path=lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, "quantbox_datasets", types.ModuleType("quantbox_datasets"))
    monkeypatch.setitem(sys.modules, "quantbox_datasets.builtins", builtins)
    prov = _provider(
        [
            {"symbol": s, "market_cap": float(v), "total_volume": 1.0, "circulating_supply": 1.0}
            for s, v in curated.iloc[0].items()
        ]
    )
    prices = pd.DataFrame({"BTC": 60000.0, "kPEPE": 0.012, "kSHIB": 0.02, "kNOPE": 1.0}, index=_IDX)

    backtest = load_pit_market_cap(prices)
    live = prov.estimate_market_cap(prices, pd.DataFrame())

    assert sorted(backtest.columns) == ["BTC", "kPEPE", "kSHIB"]
    assert sorted(live.columns[live.notna().all()]) == ["BTC", "kPEPE", "kSHIB"]
    np.testing.assert_allclose(live[backtest.columns].iloc[-1], backtest.iloc[-1])


@pytest.mark.parametrize(
    ("ticker", "known", "expected"),
    [
        ("BTC", {"BTC"}, ("BTC", 1.0)),
        ("btc", {"BTC"}, ("BTC", 1.0)),
        ("kPEPE", {"PEPE"}, ("PEPE", HL_K_PREFIX_UNITS)),
        ("kPEPE", {"KPEPE", "PEPE"}, ("KPEPE", 1.0)),  # an exact listing wins over the k notation
        ("KAVA", {"AVA"}, None),  # only a lower-case k is the 1000x notation
        ("k", {""}, None),
        ("ZZZ", {"BTC"}, None),
    ],
)
def test_map_symbol(ticker, known, expected):
    assert map_symbol(ticker, known) == expected
