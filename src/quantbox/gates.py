"""Acceptance gates on a backtest's return series: thresholds and verdicts only (TOM-1351, TOM-1618).

Five deterministic gates, five DIFFERENT questions, each a function returning a
JSON-ready dict with a ``gate_pass`` verdict, and each reachable as
``quantbox gates <name> --json`` (see :mod:`quantbox.gates_cli`):

  dsr        Is the Sharpe real once deflated for skew, kurtosis AND the number
             of trials tried?
  nw         Is the mean return significant once autocorrelation is corrected
             for, over a long enough out-of-sample window?
  factor     Is the return novel, or paid-for factor exposure?  Jensen's alpha
             with a HAC standard error, one-sided.
  bootstrap  Does a CLAIM about a metric (Sharpe, mean, max drawdown) versus a
             baseline survive a paired stationary block bootstrap?
  episode    Does the claim still hold with the largest drawdown episode removed?

Metrics DESCRIBE (:mod:`quantbox.metrics`), inference TESTS
(:mod:`quantbox.inference`), gates DECIDE. Every statistic here comes from
those two modules; this module adds only the POLICY: trial range, thresholds,
minimum windows, risk-free handling, the claim's leg. Conventions are
qute-research's ``acceptance_gates.py`` (whose pinned values
``tests/test_gates.py`` reproduces):

- Kurtosis is PEARSON (3.0 = normal), never excess.
- All Sharpe arithmetic is PER PERIOD; ``periods`` only annualises the output,
  except on the summary path (an annualised Sharpe in), where it moves the verdict.
- The trial count is a RANGE and the DSR verdict is taken at its MAX.
- ``factor`` regresses EXCESS returns: pass ``rf`` unless the series already is.
- Comparisons in ``bootstrap``/``episode`` are STRICT (``above`` means ``>``), so
  a candidate identical to its baseline never "beats" it.
- The gates' JSON reports a drawdown as a POSITIVE depth (the leg value of
  ``metric="max_drawdown"`` and ``episode.depth``): that is the qute-research
  contract the thresholds are written against. It is the one place the
  package's negative drawdown is turned into a magnitude (:func:`_depth`).

Every input that cannot produce a statistic raises :class:`GateInputError` — the
CLI maps it to exit 2, which must never be read as a pass or as a fail.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ._numerics import flat
from .inference import (
    InferenceInputError,
    bootstrap,
    deflated_sharpe_ratio,
    factor_regression,
    largest_drawdown_episode,
    newey_west_tstat,
    require_finite,
    return_moments,
)
from .metrics import max_drawdown as _max_drawdown
from .metrics import sharpe_ratio

DEFAULT_N_TRIALS: tuple[int, ...] = (1, 5, 10, 20, 50, 100)
METRICS = ("sharpe", "mean", "max_drawdown")
COMPARES = ("diff", "ratio")
PASS_IF = ("above", "below")

#: The gate cannot be evaluated on this input — exit 2, never a verdict. The same
#: class as :class:`quantbox.inference.InferenceInputError` (a ``ValueError``).
GateInputError = InferenceInputError


def _depth(drawdown: float) -> float:
    """A drawdown (negative fraction, the package convention) as the gates' positive depth."""
    return -drawdown


# ── DSR ────────────────────────────────────────────────────────────────────


def parse_n_trials(raw: str | Sequence[int]) -> list[int]:
    """``"1,10,48"`` (or a sequence) → sorted unique positive ints; anything else refused."""
    parts = [p.strip() for p in raw.split(",") if p.strip()] if isinstance(raw, str) else list(raw)
    if not parts:
        raise GateInputError(f"n_trials must not be empty, got {raw!r}")
    out = []
    for part in parts:
        try:
            n = int(part)
        except (TypeError, ValueError) as exc:
            raise GateInputError(f"n_trials entries must be integers, got {part!r}") from exc
        if n <= 0:
            raise GateInputError(f"n_trials entries must be positive, got {n}")
        out.append(n)
    return sorted(set(out))


def dsr_gate(
    *,
    sr: float,
    T: int,
    skew: float,
    kurtosis: float,
    n_trials: str | Sequence[int] = DEFAULT_N_TRIALS,
    periods: int | None = None,
    threshold: float = 0.95,
    trials_sr_std: float | None = None,
) -> dict:
    """DSR across the trial range from PER-PERIOD moments; verdict at the MAX trial count."""
    trials = parse_n_trials(n_trials)
    if periods is not None and periods <= 0:
        raise GateInputError(f"periods must be positive, got {periods!r}")
    if not (math.isfinite(threshold) and 0 < threshold < 1):
        raise GateInputError(f"threshold must be finite and strictly between 0 and 1, got {threshold!r}")
    ann = math.sqrt(periods) if periods is not None else None
    rows = {}
    for n in trials:
        res = deflated_sharpe_ratio(sr, T, skew, kurtosis, n, trials_sr_std=trials_sr_std)
        rows[n] = {
            "n_trials": n,
            "sr_std": res.sr_std,
            "sr0": res.sr0_period,
            "z": res.z,
            "psr_vs_zero": res.psr_vs_zero,
            "dsr": res.dsr,
            "sr0_annualized": res.sr0_period * ann if ann else None,
        }
    conservative = max(rows)
    dsr_c = rows[conservative]["dsr"]
    return {
        "gate": "dsr",
        "sr_period": sr,
        "periods": periods,
        "sr_annualized": sr * ann if ann else None,
        "T": T,
        "n_years": T / periods if periods else None,
        "skew": skew,
        "kurtosis": kurtosis,
        "trials_sr_std": trials_sr_std,
        "by_n_trials": {str(n): rows[n] for n in sorted(rows)},
        "n_trials_conservative": conservative,
        "dsr_conservative": dsr_c,
        "threshold": threshold,
        "gate_pass": bool(dsr_c >= threshold),
    }


def dsr_gate_from_returns(
    returns,
    *,
    n_trials: str | Sequence[int] = DEFAULT_N_TRIALS,
    periods: int | None = None,
    threshold: float = 0.95,
    trials_sr_std: float | None = None,
    allow_nonfinite_drop: bool = False,
) -> dict:
    """:func:`dsr_gate` on a per-period return series (Sharpe with ddof=1, Pearson kurtosis)."""
    m = return_moments(returns, allow_nonfinite_drop=allow_nonfinite_drop)
    out = dsr_gate(
        sr=m.sr_period,
        T=m.T,
        skew=m.skew,
        kurtosis=m.kurtosis,
        n_trials=n_trials,
        periods=periods,
        threshold=threshold,
        trials_sr_std=trials_sr_std,
    )
    out["n_nonfinite_dropped"] = m.n_nonfinite_dropped
    return out


# ── Newey-West ─────────────────────────────────────────────────────────────


def nw_gate(
    returns,
    *,
    lags: int | None = None,
    t_threshold: float = 2.0,
    min_oos_periods: int = 252,
    oos_periods: int | None = None,
    allow_nonfinite_drop: bool = False,
) -> dict:
    """HAC t on the mean AND a minimum OOS window; both must clear.

    ``oos_periods`` restricts BOTH to the last N finite observations — the rows
    before them are in-sample and must not lend the t-stat their significance.
    """
    raw = np.asarray(returns, dtype=float).ravel()
    r, dropped = require_finite(raw, allow_nonfinite_drop=allow_nonfinite_drop)
    if min_oos_periods <= 0:
        raise GateInputError(f"min_oos_periods must be positive, got {min_oos_periods}")
    if lags is not None and lags < 0:
        raise GateInputError(f"lags must be >= 0, got {lags}")
    n_file = int(r.size)
    if oos_periods is not None:
        if oos_periods <= 0:
            raise GateInputError(f"oos_periods must be positive, got {oos_periods}")
        if oos_periods > n_file:
            raise GateInputError(
                f"oos_periods={oos_periods} exceeds the {n_file} finite observations — a window "
                "longer than the data cannot be claimed"
            )
        r = r[-oos_periods:]
    if r.size < 2:
        raise GateInputError(f"need at least 2 finite return observations, got {r.size}")
    nw = newey_west_tstat(r, lags=lags)
    if nw["nw_tstat"] is None:
        raise GateInputError("zero-variance returns or degenerate HAC variance — no t-statistic exists")
    t, mean = nw["nw_tstat"], nw["mean_return"]
    nw_pass = bool(mean > 0 and t >= t_threshold)
    window_pass = bool(r.size >= min_oos_periods)
    return {
        "gate": "nw",
        "n_obs": int(r.size),
        "n_obs_file": n_file,
        "n_obs_raw": int(raw.size),
        "n_nonfinite_dropped": dropped,
        "nw_lags": nw["nw_lags"],
        "mean_return": mean,
        "nw_se": nw["nw_se"],
        "nw_tstat": t,
        "nw_pvalue_twosided": nw["nw_pvalue"],
        "t_threshold": t_threshold,
        "nw_pass": nw_pass,
        "oos_window_periods": int(r.size),
        "min_oos_periods": min_oos_periods,
        "oos_window_pass": window_pass,
        "gate_pass": bool(nw_pass and window_pass),
    }


# ── factor decomposition ───────────────────────────────────────────────────


def factor_gate(
    y,
    factors,
    names: list[str],
    *,
    rf=None,
    lags: int | None = None,
    t_threshold: float = 2.0,
    min_obs: int = 60,
    n_dropped: int = 0,
) -> dict:
    """Jensen's alpha after factor controls, HAC-robust, one-sided (H1: alpha > 0).

    ``rf`` is subtracted from ``y`` first: a per-period constant or an aligned
    array. ``None`` asserts ``y`` is already excess — it is recorded, never implied.
    """
    if not names:
        raise GateInputError(
            "no factor columns — an intercept-only fit is not a decomposition and would pass on "
            "the raw mean return alone"
        )
    if lags is not None and lags < 0:
        raise GateInputError(f"lags must be >= 0, got {lags}")
    y = np.asarray(y, dtype=float).ravel()
    F = np.asarray(factors, dtype=float)
    if rf is not None:
        y = y - np.asarray(rf, dtype=float)
    fit = factor_regression(y, F, list(names), lags=lags)
    if fit["alpha_tstat"] is None:
        raise GateInputError(
            f"no alpha t-statistic: {fit['n_obs']} observations for {len(names) + 1} parameters, "
            "collinear factors, or a degenerate HAC variance"
        )
    alpha, t, n = fit["alpha"], fit["alpha_tstat"], fit["n_obs"]
    alpha_pass = bool(alpha > 0 and t >= t_threshold)
    obs_pass = bool(n >= min_obs)
    if not obs_pass:
        note = f"too few overlapping observations ({n} < {min_obs})"
    elif alpha_pass:
        note = "residual alpha significant after factor controls — plausibly a novel edge"
    else:
        note = "no significant residual alpha — the return is explained by factor exposure"
    rf_out = None if rf is None else (float(rf) if np.ndim(rf) == 0 else "series")
    return {
        "gate": "factor",
        "n_obs": n,
        "n_nonfinite_dropped": n_dropped,
        "factors": list(names),
        "betas": fit["betas"],
        "alpha": alpha,
        "alpha_se": fit["alpha_se"],
        "alpha_tstat": t,
        "alpha_pvalue_onesided": fit["alpha_pvalue_onesided"],
        "r_squared": fit["r_squared"],
        "hac_lags": fit["hac_lags"],
        "rf": rf_out,
        "t_threshold": t_threshold,
        "alpha_pass": alpha_pass,
        "min_obs": min_obs,
        "obs_pass": obs_pass,
        "gate_pass": bool(alpha_pass and obs_pass),
        "note": note,
    }


# ── the claim leg: one metric, optionally versus a baseline ────────────────


def _metric(r: np.ndarray, metric: str) -> float:
    if metric == "mean":
        return float(r.mean())
    if metric == "max_drawdown":
        # The starting equity 1.0 counts as a peak, so a loss on the first row is a drawdown.
        return _depth(_max_drawdown(r, start_is_peak=True))
    if r.size < 2:
        return float("nan")
    # Relative, not `> 0` (see DEGENERATE_RTOL): a constant series carries std ~1e-19
    # of rounding noise, which would otherwise be a Sharpe of ~1e15 — a sure PASS.
    if flat(r, ddof=1):
        return float("nan")
    return sharpe_ratio(r, 1)


def _leg_value(cand: np.ndarray, base: np.ndarray | None, metric: str, compare: str) -> float:
    m = _metric(cand, metric)
    if base is None:
        return m
    b = _metric(base, metric)
    if compare == "diff":
        return m - b
    return m / b if b != 0 else float("nan")


def _leg_passes(value: float, pass_if: str, threshold: float) -> bool:
    if not math.isfinite(value):
        return False
    return value > threshold if pass_if == "above" else value < threshold


def _check_leg(metric: str, compare: str, pass_if: str, threshold: float) -> None:
    if metric not in METRICS:
        raise GateInputError(f"metric must be one of {METRICS}, got {metric!r}")
    if compare not in COMPARES:
        raise GateInputError(f"compare must be one of {COMPARES}, got {compare!r}")
    if pass_if not in PASS_IF:
        raise GateInputError(f"pass_if must be one of {PASS_IF}, got {pass_if!r}")
    if not math.isfinite(threshold):
        raise GateInputError(f"threshold must be finite, got {threshold!r}")


def _paired(candidate, baseline, *, allow_drop: bool) -> tuple[np.ndarray, np.ndarray | None, int]:
    """The two series as aligned finite arrays; a row is dropped from BOTH or neither."""
    c = np.asarray(candidate, dtype=float).ravel()
    if baseline is None:
        r, dropped = require_finite(c, allow_nonfinite_drop=allow_drop)
        return r, None, dropped
    b = np.asarray(baseline, dtype=float).ravel()
    if b.size != c.size:
        raise GateInputError(
            f"candidate has {c.size} observations and baseline {b.size} — a paired test needs the "
            "same rows in both (align them on dates first)"
        )
    both, dropped = require_finite(np.column_stack([c, b]), allow_nonfinite_drop=allow_drop, what="paired")
    return np.ascontiguousarray(both[:, 0]), np.ascontiguousarray(both[:, 1]), dropped


# ── paired stationary block bootstrap ─────────────────────────────────────


def paired_block_bootstrap(
    candidate,
    baseline,
    *,
    metric: str = "sharpe",
    compare: str = "diff",
    pass_if: str = "above",
    threshold: float = 0.0,
    draws: int = 2000,
    mean_block: float = 20,
    seed: int = 0,
    min_probability: float = 0.95,
    allow_nonfinite_drop: bool = False,
) -> dict:
    """P(the claim's leg holds) under a paired stationary block bootstrap.

    ``candidate`` and ``baseline`` are aligned per-period return series. Every draw
    resamples the SAME rows from both (that is what "paired" means: the common
    market shocks stay matched; :func:`quantbox.inference.bootstrap`), computes the
    leg — ``metric(candidate)`` versus ``metric(baseline)`` by ``compare``
    (``diff`` = c - b, ``ratio`` = c / b) — and asks whether it lies strictly
    ``above``/``below`` ``threshold``. ``probability`` is the share of draws where
    it does; the gate passes when ``probability >= min_probability``. A draw whose
    leg is undefined (a zero standard deviation, a zero baseline drawdown under
    ``ratio``) counts as NOT passing and is reported in ``n_undefined_draws``.

    Sharpe is per period (ddof=1); ``max_drawdown`` is a positive depth of the
    compounded equity, so "candidate MDD at most 70% of baseline's" is
    ``metric="max_drawdown", compare="ratio", pass_if="below", threshold=0.7``.
    """
    _check_leg(metric, compare, pass_if, threshold)
    if not (isinstance(draws, (int, np.integer)) and draws >= 1):
        raise GateInputError(f"draws must be a positive integer, got {draws!r}")
    if not (0 < min_probability <= 1):
        raise GateInputError(f"min_probability must be in (0, 1], got {min_probability!r}")
    c, b, dropped = _paired(candidate, baseline, allow_drop=allow_nonfinite_drop)
    if b is None:
        raise GateInputError("a paired bootstrap needs a baseline series")
    n = int(c.size)
    if n < 2:
        raise GateInputError(f"need at least 2 paired observations, got {n}")
    if not (math.isfinite(mean_block) and 1 <= mean_block <= n):
        raise GateInputError(f"mean_block must be in [1, {n}] (the sample length), got {mean_block!r}")
    point = _leg_value(c, b, metric, compare)
    if not math.isfinite(point):
        raise GateInputError(f"the {metric} leg is undefined on the full sample (zero variance or a zero baseline)")

    values = bootstrap(
        lambda cs, bs: _leg_value(cs, bs, metric, compare),
        c,
        b,
        draws=draws,
        rng=np.random.default_rng(seed),
        mean_block=mean_block,
    )
    defined = values[np.isfinite(values)]
    hits = sum(_leg_passes(v, pass_if, threshold) for v in defined)
    probability = hits / draws
    q = np.quantile(defined, [0.025, 0.5, 0.975]) if defined.size else [None] * 3
    return {
        "gate": "bootstrap",
        "method": "paired-stationary-block-bootstrap",
        "metric": metric,
        "compare": compare,
        "pass_if": pass_if,
        "threshold": threshold,
        "n_obs": n,
        "n_nonfinite_dropped": dropped,
        "draws": draws,
        "mean_block": mean_block,
        "seed": seed,
        "point_estimate": point,
        "bootstrap_mean": float(defined.mean()) if defined.size else None,
        "bootstrap_std": float(defined.std(ddof=1)) if defined.size > 1 else None,
        "quantiles": {
            k: (float(v) if v is not None else None) for k, v in zip(("0.025", "0.5", "0.975"), q, strict=True)
        },
        "n_undefined_draws": int(draws - defined.size),
        "probability": probability,
        "min_probability": min_probability,
        "gate_pass": bool(probability >= min_probability),
    }


# ── largest drawdown episode ──────────────────────────────────────────────


def episode_gate(
    candidate,
    baseline=None,
    *,
    metric: str = "sharpe",
    compare: str = "diff",
    pass_if: str = "above",
    threshold: float = 0.0,
    allow_nonfinite_drop: bool = False,
) -> dict:
    """The claim's leg re-evaluated with the single largest drawdown episode removed.

    The episode (:func:`quantbox.inference.largest_drawdown_episode`) is found on
    the BASELINE when one is given (the prereg's "largest baseline drawdown
    episode"), else on the candidate, and its rows are removed from both series.
    ``gate_pass`` is the verdict WITHOUT the episode; ``passes_only_with_episode``
    flags the case a prereg relabels. What that relabel is (e.g. "inconclusive")
    is the caller's policy, not this function's. A series with no drawdown has
    nothing to remove: the result equals the full one.
    """
    _check_leg(metric, compare, pass_if, threshold)
    c, b, dropped = _paired(candidate, baseline, allow_drop=allow_nonfinite_drop)
    if c.size < 2:
        raise GateInputError(f"need at least 2 observations, got {c.size}")
    source = "baseline" if b is not None else "returns"
    episode = largest_drawdown_episode(b if b is not None else c)
    keep = np.ones(c.size, dtype=bool)
    if episode is not None:
        keep[episode["start"] : episode["end"] + 1] = False
        episode = {**episode, "depth": _depth(episode["depth"])}
    n_left = int(keep.sum())
    if n_left < 2:
        raise GateInputError(
            f"removing the largest drawdown episode leaves {n_left} observation(s) — the episode IS "
            "the sample, so there is no result without it"
        )
    full = _leg_value(c, b, metric, compare)
    ex = _leg_value(c[keep], b[keep] if b is not None else None, metric, compare)
    if not (math.isfinite(full) and math.isfinite(ex)):
        raise GateInputError(f"the {metric} leg is undefined (zero variance or a zero baseline)")
    full_pass = _leg_passes(full, pass_if, threshold)
    ex_pass = _leg_passes(ex, pass_if, threshold)
    return {
        "gate": "episode",
        "metric": metric,
        "compare": compare if b is not None else None,
        "pass_if": pass_if,
        "threshold": threshold,
        "n_obs": int(c.size),
        "n_nonfinite_dropped": dropped,
        "episode_source": source,
        "episode": episode,
        "full": {"value": full, "pass": full_pass, "n_obs": int(c.size)},
        "ex_episode": {"value": ex, "pass": ex_pass, "n_obs": n_left},
        "passes_only_with_episode": bool(full_pass and not ex_pass),
        "gate_pass": bool(ex_pass),
    }


__all__ = [
    "COMPARES",
    "DEFAULT_N_TRIALS",
    "METRICS",
    "PASS_IF",
    "GateInputError",
    "dsr_gate",
    "dsr_gate_from_returns",
    "episode_gate",
    "factor_gate",
    "nw_gate",
    "paired_block_bootstrap",
    "parse_n_trials",
]
