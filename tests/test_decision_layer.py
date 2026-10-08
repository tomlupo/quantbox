"""Normalisation is part of the decision: target weights are final before any rebalancing policy reads them (TOM-1520).

Tom, 2026-10-05: "Normalizacja jest częścią strategii, bo pokazuje, co byśmy
robili w realnym świecie." signals -> weights -> the decision
(:mod:`quantbox.decision`: short clip, gross cap, group limits,
``venue.leverage`` normalisation) -> TARGET weights -> rebalancing policy ->
execution. "Konto bez lewara nie otrzymuje wag, gdzie wymagany jest lewar."

The acceptance criteria of TOM-1520, each as a test:

- the same config gives IDENTICAL target weights in the backtest and in
  trading, and their net is <= 1 under normalize;
- without borrow, the held net never goes above 1 after a rebalance —
  deferral included (the one case where a final target can still need cash
  the account does not have). It fails when the cash cap is disabled.

TOM-1525 adds one gross-cap default (``risk.max_leverage: 1``) read in one
place (:func:`quantbox.decision.gross_cap`): the same config WITHOUT the key
gives the same targets in the backtest and in trading.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import quantbox
from quantbox.decision import (
    DEFAULT_MAX_LEVERAGE,
    DecisionRules,
    final_book,
    final_targets,
    gross_cap,
    risk_caps_row,
)
from quantbox.engine.groups import resolve_group_limits
from quantbox.engine.schedule import schedule_book
from quantbox.instrument_calendar import execution_bars, instrument_calendar
from quantbox.plugins.pipeline.backtest_pipeline import BacktestPipeline
from quantbox.plugins.pipeline.trading_pipeline import TradingPipeline
from quantbox.plugins.rebalancing.futures_rebalancer import FuturesRebalancer
from quantbox.plugins.rebalancing.standard_rebalancer import StandardRebalancer
from quantbox.store import FileArtifactStore

TICKERS = ["A", "B", "C", "D"]

# ----------------------------------------------------------------------
# The decision itself: one ordered transform
# ----------------------------------------------------------------------


def _row(**w: float) -> pd.DataFrame:
    return pd.DataFrame([w], index=pd.DatetimeIndex(["2024-01-01"]))


def test_the_short_clip_runs_before_the_gross_cap():
    """Long 1.5 / short 0.5 under a gross cap of 1, no shorts: clip first (long 1.5), then cap (long 1.0).
    The other order (the live rebalancers' before TOM-1520) left long 0.75: a cap spent on a clipped short."""
    out, report = final_targets(_row(A=1.5, B=-0.5), DecisionRules(allow_short=False, max_leverage=1.0))
    assert out.iloc[0].to_dict() == pytest.approx({"A": 1.0, "B": 0.0})
    assert report["rows_short_clipped"] == 1 and report["rows_gross_capped"] == 1
    assert risk_caps_row({"A": 1.5, "B": -0.5}, allow_short=False, max_leverage=1.0) == pytest.approx(
        {"A": 1.0, "B": 0.0}
    )


@pytest.mark.parametrize(
    ("leverage", "expected", "normalised"),
    [("normalize", {"A": 0.75, "B": 0.25}, 1), ("borrow", {"A": 1.2, "B": 0.4}, 0), ("none", {"A": 1.2, "B": 0.4}, 0)],
)
def test_normalize_scales_a_row_above_net_one_and_borrow_keeps_it(leverage, expected, normalised):
    out, report = final_targets(_row(A=1.2, B=0.4), DecisionRules(max_leverage=None, leverage=leverage))
    assert out.iloc[0].to_dict() == pytest.approx(expected)
    assert report["rows_normalised"] == normalised
    assert report["rows_above_net_1"] == 1  # measured in every mode
    assert report["max_net_exposure_decided"] == pytest.approx(1.6)


def test_a_long_short_row_is_normalised_on_its_net_not_its_gross():
    """Gross 2.0, net 0.6: nothing to normalise (net is what needs cash); the gross cap is a separate rule."""
    out, report = final_targets(_row(A=1.3, B=-0.7), DecisionRules(max_leverage=None))
    assert out.iloc[0].to_dict() == pytest.approx({"A": 1.3, "B": -0.7})
    assert report["rows_normalised"] == 0


def test_group_limits_run_before_the_normalisation():
    """A group max holds after normalisation (it only scales down), and the row ends at net 1."""
    universe = pd.DataFrame({"symbol": ["A", "B", "C"], "asset_class": ["equity", "equity", "bond"]})
    groups = resolve_group_limits({"by": "asset_class", "limits": {"equity": {"max": 0.6}}}).bind(universe)
    out, report = final_targets(_row(A=0.8, B=0.4, C=0.4), DecisionRules(max_leverage=None, groups=groups))
    row = out.iloc[0]
    assert row["A"] + row["B"] <= 0.6 + 1e-12
    assert row.sum() == pytest.approx(1.0)
    assert report["groups"]["rows_adjusted"] == 1 and report["rows_normalised"] == 1


def test_a_nan_cell_no_step_changed_stays_nan():
    """The seam fills a NaN on the price bars (HOLD); the decision must not fill it on the strategy's rows first."""
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    raw = pd.DataFrame({"A": [0.5, np.nan, 0.6], "B": [0.3, 0.3, np.nan]}, index=idx)
    out, _ = final_targets(raw, DecisionRules(max_leverage=None))
    assert np.isnan(out.loc[idx[1], "A"]) and np.isnan(out.loc[idx[2], "B"])
    # ...but a held cell that makes its row need normalising is written explicitly (the row's net counts it).
    raw2 = pd.DataFrame({"A": [0.5, np.nan], "B": [0.3, 0.9]}, index=idx[:2])
    out2, _ = final_targets(raw2, DecisionRules(max_leverage=None))
    assert out2.loc[idx[1]].sum() == pytest.approx(1.0) and out2.loc[idx[1], "A"] == pytest.approx(0.5 / 1.4)


def test_final_book_decides_every_strategy_slice_on_its_own():
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    cols = pd.MultiIndex.from_product([["s1", "s2"], ["A", "B"]], names=["slice", "ticker"])
    book = pd.DataFrame([[1.0, 1.0, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]], index=idx, columns=cols)
    out, reports = final_book(book, DecisionRules(max_leverage=None))
    assert out["s1"].sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert out["s2"].sum(axis=1).tolist() == pytest.approx([0.4, 1.0])
    assert [r["rows_normalised"] for r in reports] == [1, 1]


def test_the_rebalancers_apply_the_decision_order():
    """StandardRebalancer clipped AFTER its cap before TOM-1520; it now runs the decision's own steps 1-2."""
    std = StandardRebalancer()._apply_risk_transforms({"A": 1.5, "B": -0.5}, {"max_leverage": 1.0})
    assert std == pytest.approx({"A": 1.0})
    fut = FuturesRebalancer()._apply_risk_transforms({"A": 1.5, "B": -0.5}, {"max_leverage": 1.0})
    assert fut == pytest.approx({"A": 0.75, "B": -0.25})  # futures: no clip, the gross cap only


# ----------------------------------------------------------------------
# AC 1: the same config, identical target weights in backtest and trading
# ----------------------------------------------------------------------

N_BARS = 45


def _prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="D")
    rng = np.random.default_rng(3)
    steps = rng.normal(0.0, 0.02, size=(N_BARS, len(TICKERS)))
    return pd.DataFrame(100.0 * np.cumprod(1.0 + steps, axis=0), index=idx, columns=TICKERS)


def _decided(prices: pd.DataFrame) -> pd.DataFrame:
    """Rows above net 1, above a gross cap of 1.4, and with shorts the config clips."""
    rng = np.random.default_rng(9)
    return pd.DataFrame(rng.normal(0.3, 0.35, size=prices.shape), index=prices.index, columns=TICKERS)


class _Data:
    def __init__(self, prices: pd.DataFrame) -> None:
        self.prices = prices

    def load_universe(self, params: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame({"symbol": list(self.prices.columns)})

    def load_market_data(self, universe: Any, asof: str, params: dict[str, Any]) -> dict[str, pd.DataFrame]:
        return {"prices": self.prices}


class _Replay:
    meta = type("M", (), {"name": "strategy.replay.v1"})()

    def __init__(self, decided: pd.DataFrame) -> None:
        self.decided = decided

    def run(self, data: Any, params: Any = None, context: Any = None) -> dict[str, Any]:
        return {"weights": self.decided.loc[: data["prices"].index[-1]]}


#: ONE config block both pipelines read.
CONFIGS = {
    "periodic_daily": {"risk": {"max_leverage": 1.4, "allow_short": False}},
    "tranche3_daily": {
        "risk": {"max_leverage": 1.4, "allow_short": False},
        "rebalancing_policy": {"cadence": "tranche", "tranches": 3, "frequency": "daily"},
    },
    "long_short_periodic": {"risk": {"max_leverage": 1.4, "allow_short": True}},
}


def _backtest_targets(tmp_path, cfg: dict[str, Any], prices: pd.DataFrame, decided: pd.DataFrame) -> pd.DataFrame:
    store = FileArtifactStore(str(tmp_path / "bt"), "run")
    params = {"fees": 0.0, "strategies": [{"name": "strategy.replay.v1", "weight": 1.0}], **cfg}
    BacktestPipeline().run(
        mode="backtest",
        asof=str(prices.index[-1].date()),
        params=params,
        data=_Data(prices),
        store=store,
        broker=None,
        risk=[],
        strategies=[_Replay(decided)],
    )
    # Every instrument prints on every bar, the policy trades every bar, lag 1: the book held on bar
    # d + 1 IS the target decided on bar d.
    return store.read_parquet("traded_weights").set_index("date").reindex(columns=TICKERS)


def _trading_targets(tmp_path, cfg: dict[str, Any], prices: pd.DataFrame, decided: pd.DataFrame, d: int) -> pd.Series:
    store = FileArtifactStore(str(tmp_path / f"live{d}"), "run")
    params = {"strategies": [{"name": "strategy.replay.v1", "weight": 1.0, "params": {}}], **cfg}
    TradingPipeline().run(
        mode="backtest",  # no broker: the targets are the run's output, nothing is sized or sent
        asof=str(prices.index[d].date()),
        params=params,
        data=_Data(prices.iloc[: d + 1]),
        store=store,
        broker=None,
        risk=[],
        strategies=[_Replay(decided)],
    )
    targets = store.read_parquet("targets")
    return pd.Series(targets["weight"].to_numpy(dtype=float), index=targets["symbol"]).reindex(TICKERS).fillna(0.0)


@pytest.mark.parametrize("name", CONFIGS)
def test_the_same_config_gives_identical_target_weights_in_backtest_and_trading(tmp_path, name):
    cfg = CONFIGS[name]
    prices = _prices()
    decided = _decided(prices)
    # The config really exercises every step: shorts to clip, gross above the cap, net above 1.
    assert (decided < 0).any().any()
    assert (decided.clip(lower=0).sum(axis=1) > 1.4).any() and (decided.sum(axis=1) > 1.0).any()
    held = _backtest_targets(tmp_path, cfg, prices, decided)
    for d in (5, 17, 30, N_BARS - 2):
        live = _trading_targets(tmp_path, cfg, prices, decided, d)
        backtest = held.loc[prices.index[d + 1]]
        np.testing.assert_allclose(live.to_numpy(), backtest.to_numpy(), rtol=0, atol=1e-12, err_msg=f"bar {d}")
        assert live.sum() <= 1.0 + 1e-9, f"bar {d}: live net {live.sum()}"
        assert backtest.sum() <= 1.0 + 1e-9, f"bar {d}: backtest net {backtest.sum()}"


def test_a_levered_perps_book_is_normalised_live_unless_it_declares_borrow(tmp_path):
    """carver_hyperliquid's shape (futures rebalancer, max_leverage 2): a net-1.5 row reaches the order
    generator at net 1 by default (TOM-1520 gap (a): live never normalised); borrow keeps 1.5."""
    from quantbox.plugins.broker.sim import SimPaperBroker

    prices = _prices()
    decided = pd.DataFrame({"A": 1.0, "B": 0.5, "C": 0.0, "D": 0.0}, index=prices.index)

    def run(venue: dict[str, Any] | None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "strategies": [{"name": "strategy.replay.v1", "weight": 1.0, "params": {}}],
            "stable_coin_symbol": "USD",
            "trading_enabled": False,
            "_rebalancer_cfg": {"params": {"max_leverage": 2}},
        }
        if venue is not None:
            params["venue"] = venue
        result = TradingPipeline().run(
            mode="paper",
            asof=str(prices.index[-1].date()),
            params=params,
            data=_Data(prices),
            store=FileArtifactStore(str(tmp_path / f"perps-{venue}"), "run"),
            broker=SimPaperBroker(cash=100_000.0, quote_currency="USD"),
            risk=[],
            strategies=[_Replay(decided)],
            rebalancer=FuturesRebalancer(),
        )
        decision = result.notes["decision"]
        assert decision["rules"]["max_leverage"] == 2.0 and decision["rules"]["allow_short"] is True
        return decision

    normalised = run(None)
    assert normalised["rows_normalised"] == 1
    assert normalised["max_net_exposure_final"] == pytest.approx(1.0)
    kept = run({"leverage": "borrow"})
    assert kept["rows_normalised"] == 0
    assert kept["max_net_exposure_final"] == pytest.approx(1.5)


# ----------------------------------------------------------------------
# TOM-1525: one gross-cap default (1), read in one place, in every door
# ----------------------------------------------------------------------


def test_the_gross_cap_default_is_one_and_one_reader_returns_it():
    """Tom, 2026-10-06: "Jedno: 1 wszędzie". Before: 99 in the backtest, 1 in trading, none in the L1 doors."""
    assert DEFAULT_MAX_LEVERAGE == 1.0
    assert gross_cap(None) == 1.0 and gross_cap({}) == 1.0 and gross_cap({"allow_short": True}) == 1.0
    assert gross_cap({"max_leverage": 2}) == 2.0
    assert DecisionRules().max_leverage == 1.0


def test_every_reader_of_max_leverage_gets_the_same_default():
    """The backtest plan (what ``config explain`` prints as ``venue.max_leverage``), trading, both rebalancers."""
    assert BacktestPipeline().plan({})["venue"]["max_leverage"] == 1.0
    assert TradingPipeline._risk_rules({})["max_leverage"] == 1.0
    assert StandardRebalancer().risk_rules({})["max_leverage"] == 1.0
    assert FuturesRebalancer().risk_rules({})["max_leverage"] == 1.0


def test_max_leverage_is_read_in_one_place():
    """No module but ``quantbox.decision`` reads ``max_leverage`` out of a config with its own default:
    the 99 lived in ``backtest_pipeline._max_leverage``, and a second literal default is how the paths split."""
    src = Path(quantbox.__file__).parent
    pattern = re.compile(r"""\.get\(\s*["']max_leverage["']\s*,""")  # a .get with a default of its own
    readers = sorted(
        str(p.relative_to(src)) for p in src.rglob("*.py") if p.name != "decision.py" and pattern.search(p.read_text())
    )
    assert readers == [], f"max_leverage read with a local default in {readers}; call quantbox.decision.gross_cap"


@pytest.mark.parametrize("allow_short", [True, False])
def test_without_max_leverage_backtest_and_trading_give_the_same_targets_gross_at_most_one(tmp_path, allow_short):
    """The acceptance test of TOM-1525: the config does not name ``max_leverage``."""
    cfg = {"risk": {"allow_short": allow_short}}
    prices = _prices()
    decided = _decided(prices)
    book = decided if allow_short else decided.clip(lower=0)
    assert (book.abs().sum(axis=1) > 1.0).mean() > 0.5  # the decided book really needs the cap
    held = _backtest_targets(tmp_path, cfg, prices, decided)
    for d in (5, 17, 30, N_BARS - 2):
        live = _trading_targets(tmp_path, cfg, prices, decided, d)
        backtest = held.loc[prices.index[d + 1]]
        np.testing.assert_allclose(live.to_numpy(), backtest.to_numpy(), rtol=0, atol=1e-12, err_msg=f"bar {d}")
        assert live.abs().sum() <= 1.0 + 1e-9, f"bar {d}: live gross {live.abs().sum()}"
        assert backtest.abs().sum() <= 1.0 + 1e-9, f"bar {d}: backtest gross {backtest.abs().sum()}"


def _levered() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long 1.5 / short 0.5 on every bar: gross 2, net 1 — normalize leaves it, only the gross cap moves it."""
    prices = _prices()
    return prices, pd.DataFrame({"A": 1.5, "B": -0.5, "C": 0.0, "D": 0.0}, index=prices.index)


def test_backtest_caps_gross_at_one_unless_the_call_declares_its_leverage():
    from quantbox.plugins.backtesting import backtest

    prices, weights = _levered()
    default = backtest(prices, weights, engine="rsims", fees=0.0)
    decision = default["book"].data_validation["decision"]
    assert decision["rules"]["max_leverage"] == 1.0
    assert decision["rows_gross_capped"] == len(prices)
    levered = backtest(prices, weights, engine="rsims", fees=0.0, max_leverage=2)
    decision = levered["book"].data_validation["decision"]
    assert decision["rules"]["max_leverage"] == 2.0 and decision["rows_gross_capped"] == 0
    assert levered["metrics"]["total_return"] != pytest.approx(default["metrics"]["total_return"])


def test_optimize_caps_gross_at_one_unless_the_call_declares_its_leverage():
    from quantbox.plugins.backtesting import backtest, optimize

    prices, weights = _levered()

    def weights_fn(p: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        return weights.loc[p.index] * params["k"]

    kw: dict[str, Any] = {"engine": "rsims", "fees": 0.0}
    default = optimize(prices, weights_fn, {"k": [1.0]}, **kw)["all_results"].loc[0, "total_return"]
    levered = optimize(prices, weights_fn, {"k": [1.0]}, **kw, max_leverage=2)["all_results"].loc[0, "total_return"]
    assert default == pytest.approx(backtest(prices, weights, **kw)["metrics"]["total_return"])
    assert levered == pytest.approx(backtest(prices, weights, **kw, max_leverage=2)["metrics"]["total_return"])
    assert levered != pytest.approx(default)


class _Levered:
    """A sweep strategy: the levered book, whatever its parameter."""

    def __init__(self, k: float = 1.0) -> None:
        self.k = k

    def run(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        return {"weights": _levered()[1] * self.k}


def test_the_sweep_caps_gross_at_one_unless_backtest_kwargs_declare_leverage():
    from quantbox.sweep import sweep

    prices, _ = _levered()

    def total_return(**extra: Any) -> float:
        kw = {"engine": "rsims", "fees": 0.0, **extra}
        out = sweep(_Levered, {}, {"k": [1.0]}, {"prices": prices}, backtest_kwargs=kw, metrics=("total_return",))
        return float(out["total_return"].iloc[0])

    assert total_return() == pytest.approx(total_return(max_leverage=1))
    assert total_return(max_leverage=2) != pytest.approx(total_return())


class _NoRulesRebalancer:
    """A third-party rebalancer: ``generate_orders`` only, no ``risk_rules``. It records what it is asked to size."""

    meta = type("M", (), {"name": "rebalancing.third_party.v1"})()

    def __init__(self) -> None:
        self.sized: list[dict[str, float]] = []

    def generate_orders(self, weights: dict[str, float], broker: Any, params: dict[str, Any]) -> dict[str, Any]:
        self.sized.append({str(k): float(v) for k, v in weights.items()})
        return {"rebalancing": pd.DataFrame(), "orders": pd.DataFrame(), "total_value": 0.0, "weights": dict(weights)}


@pytest.mark.parametrize("rebal_params", [{}, {"max_leverage": 2}], ids=["default", "declared_2"])
def test_a_rebalancer_without_risk_rules_gets_the_default_gross_cap_in_trading(tmp_path, rebal_params):
    """TOM-1526: before, a rebalancer with no ``risk_rules`` got NO gross cap in trading (gross 2 sized as 2),
    while every other door capped at :data:`DEFAULT_MAX_LEVERAGE`. Tom, 2026-10-06: "Dostać domyślne 1"."""
    from quantbox.plugins.broker.sim import SimPaperBroker

    prices, levered = _levered()  # gross 2 on every row, shorts included
    rebalancer = _NoRulesRebalancer()
    result = TradingPipeline().run(
        mode="paper",
        asof=str(prices.index[-1].date()),
        params={
            "strategies": [{"name": "strategy.replay.v1", "weight": 1.0, "params": {}}],
            "stable_coin_symbol": "USD",
            "trading_enabled": False,
            "_rebalancer_cfg": {"params": rebal_params},
        },
        data=_Data(prices),
        store=FileArtifactStore(str(tmp_path / "third-party"), "run"),
        broker=SimPaperBroker(cash=100_000.0, quote_currency="USD"),
        risk=[],
        strategies=[_Replay(levered)],
        rebalancer=rebalancer,
    )
    cap = gross_cap(rebal_params)
    decision = result.notes["decision"]
    assert decision["rules"]["max_leverage"] == cap
    assert decision["rules"]["allow_short"] is True  # the rebalancer still owns the short side
    assert rebalancer.sized, "the rebalancer was never asked to size the targets"
    sized = rebalancer.sized[-1]
    assert sum(abs(w) for w in sized.values()) == pytest.approx(cap)
    assert sized["B"] < 0


# ----------------------------------------------------------------------
# AC 2: without borrow the held net never goes above 1 after a rebalance, deferral included
# ----------------------------------------------------------------------


def _deferral_case(seed: int, n: int = 160, k: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rotating long/short books, every row's net at most 1 AFTER the decision, instruments closed on random
    bars (orders deferred to their next print while the deferred cell keeps its old, drifting weight)."""
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    cols = [f"Y{i}" for i in range(k)]
    prices = pd.DataFrame(100.0 * np.cumprod(1 + rng.normal(0.0, 0.03, (n, k)), axis=0), index=idx, columns=cols)
    closed = rng.random((n, k)) < 0.2
    closed[0] = closed[-1] = False
    prices = prices.mask(closed)
    raw = rng.normal(0.35, 0.5, (n, k)) * (rng.random((n, k)) < 0.7)
    return prices, pd.DataFrame(raw, index=idx, columns=cols)


def _drifted_held_net(prices: pd.DataFrame, cal_prices: pd.DataFrame, book) -> np.ndarray:
    """The cost-free held book (ADR-0008) replayed from the schedule's orders and targets: after each placed
    rebalance the ordered cells hold their targets, every other cell its DRIFTED weight."""
    p = cal_prices.to_numpy(dtype=float)
    orders = book.orders.to_numpy()
    targets = book.weights.to_numpy(dtype=float)
    held = np.zeros(orders.shape[1])
    last = None
    nets = []
    for r in np.flatnonzero(orders.any(axis=1)):
        if last is not None:
            growth = p[r] / p[last]
            value = held * np.where(np.isfinite(growth), growth, 1.0)
            held = value / ((1.0 - held.sum()) + value.sum())
        o = orders[r]
        held[o] = targets[r][o]
        nets.append(held.sum())
        last = r
    return np.asarray(nets)


@pytest.mark.parametrize("seed", range(12))
def test_without_borrow_the_held_net_never_goes_above_one_after_a_rebalance(seed):
    prices, raw = _deferral_case(seed)
    final, _ = final_targets(raw, DecisionRules(max_leverage=None))
    assert (final.sum(axis=1) <= 1.0 + 1e-9).all()  # the decision's half: every target is final
    cal = instrument_calendar(prices)
    bars = execution_bars(cal, "majority")
    book = schedule_book(final, cal, bars, 1, 1, leverage="normalize")
    assert book.report["timing"]["deferred_trades"] > 0  # the case the cash cap exists for
    nets = _drifted_held_net(prices, cal.prices, book)
    assert nets.max() <= 1.0 + 1e-9, f"held net {nets.max():.6f} after a rebalance"


def test_borrow_lifts_the_cash_cap():
    """The same deferral book under borrow holds above net 1 somewhere: the cap is normalize's, not a clamp."""
    worst = 0.0
    for seed in range(12):
        prices, raw = _deferral_case(seed)
        final, _ = final_targets(raw, DecisionRules(max_leverage=None))
        cal = instrument_calendar(prices)
        book = schedule_book(final, cal, execution_bars(cal, "majority"), 1, 1, leverage="borrow")
        worst = max(worst, _drifted_held_net(prices, cal.prices, book).max())
    assert worst > 1.0 + 1e-6


def test_vectorbt_holds_exactly_the_book_after_a_deferral(tmp_path):
    """vectorbt cannot borrow: before TOM-1520 a deferred cell could leave it short of cash, and it cut buys on
    its own (engine_underfilled_rebalances > 0). With the cash cap the seam never asks for that cash."""
    prices, raw = _deferral_case(4, n=200)
    store = FileArtifactStore(str(tmp_path), "run")
    result = BacktestPipeline().run(
        mode="backtest",
        asof=str(prices.index[-1].date()),
        params={
            "fees": 0.0,
            "strategies": [{"name": "strategy.replay.v1", "weight": 1.0}],
            "venue": {"allow_shorts": True},
        },
        data=_Data(prices),
        store=store,
        broker=None,
        risk=[],
        strategies=[_Replay(raw)],
    )
    assert result.metrics["calendar_deferred_trades"] > 0
    assert result.metrics["leverage_cash_capped_rebalances"] > 0
    assert result.metrics["engine_underfilled_rebalances"] == 0.0
    assert result.metrics["engine_max_fill_gap"] < 1e-9
