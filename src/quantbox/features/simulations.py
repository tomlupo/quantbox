"""Parametric Monte Carlo simulations and summary statistics.

These are the two building blocks historically vendored into host projects
(see e.g. robo's ``src/market/simulations.py``). They are pure-numpy/pandas
+ ``scipy.stats`` for the distribution shocks.

``parametric_mc`` generates a GBM-style correlated simulation of asset
returns; ``simulations_stats`` produces standard summary quantiles
(end value, CAGR, vol, Sharpe, max drawdown) from the output panel.

Callers are expected to pass a seeded ``numpy.random.Generator`` as the
``seed`` argument for reproducibility — a bare integer seed is accepted
for backwards compatibility but deterministic use requires a Generator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _draw_uncorrelated(size, distribution, df, dtype, seed, target_bytes=128 * 1024**2):
    """Fill an ``(n_assets, n_cols)`` panel of iid shocks, natively at *dtype*.

    ``scipy.stats`` has no dtype argument, so ``norm.rvs`` / ``t.rvs``
    always materialise float64 and a float32 caller then pays for a second
    full-size array to downcast into. On a long horizon that is the single
    largest allocation in ``parametric_mc``: a (4, 100_800_000) panel is
    3.0 GiB in float64 to produce a 1.5 GiB float32 result, and it is what
    a production batch died on (2026-09-09, MemoryError with ~10 GiB free).

    ``numpy``'s Generator does take a dtype — for ``standard_normal`` and
    ``standard_gamma``, though not for ``standard_t`` or ``chisquare`` — so
    Student-t is assembled from those two as ``Z / sqrt(X / df)`` with
    ``X ~ chi2(df) = 2 * Gamma(df/2)``. Drawing in column blocks into one
    preallocated panel keeps the transient cost to a block rather than a
    second copy of the whole thing.

    This changes the random stream: output is drawn from the same
    distributions with the same parameters, but the numbers differ, exactly
    as reseeding would. It is not a drop-in for a byte-identical rerun.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    n_assets, n_cols = size
    out = np.empty(size, dtype=dtype)

    # One block ~= `target_bytes` of float64 working space, so the
    # transient cost is bounded instead of scaling with the horizon the
    # way a second full panel would.
    block = max(1, int(target_bytes // max(1, n_assets * 8)))

    for start in range(0, n_cols, block):
        stop = min(start + block, n_cols)
        shape = (n_assets, stop - start)

        if distribution == "normal":
            # No division, so nothing here can amplify a small value:
            # the draw is safe directly at the panel's own dtype.
            out[:, start:stop] = rng.standard_normal(shape, dtype=dtype)
            continue

        # Student-t as Z / sqrt(X / df) with X ~ chi2(df) = 2 * Gamma(df/2).
        #
        # This block is assembled in float64 even when the panel is
        # float32, and that is not incidental. `standard_gamma` returns
        # values arbitrarily close to zero; in float32 the sqrt of one of
        # those loses enough precision that the division turns a
        # legitimate tail draw into a spurious ~1e15 outlier. Measured:
        # drawing this construction natively in float32 left the body of
        # the distribution (1st-99th percentile) correct while the sample
        # mean and standard deviation diverged by fifteen orders of
        # magnitude. The block is bounded, so float64 here costs a
        # temporary rather than a second panel.
        z = rng.standard_normal(shape)
        g = rng.standard_gamma(df / 2.0, shape)
        g *= 2.0 / df
        np.sqrt(g, out=g)
        z /= g
        out[:, start:stop] = z

    return out


def parametric_mc(
    prices=None,
    returns_data=False,
    frequency=252,
    step_frequency=None,
    iterations=1000,
    steps=252,
    correlated=True,
    mu=None,
    var=None,
    cov=None,
    distribution="normal",
    df=3,
    seed=None,
    precision="float64",
    *args,
    **kwargs,
):
    """Run a parametric (GBM) Monte Carlo simulation.

    Returns a MultiIndex DataFrame of simulated returns with levels
    ``(ticker, sim_no)`` as columns and ``step`` as the index.
    """
    # log returns
    if prices is not None:
        if returns_data:
            log_returns = np.log(1 + prices)
        else:
            log_returns = np.log(1 + prices.pct_change())

    # parameters (annualized)
    if mu is None:
        mu = log_returns.mean() * frequency
    if correlated:
        if cov is None:
            cov = log_returns.cov() * frequency
        var = pd.Series(np.diag(cov), index=cov.index)
    else:
        if var is None:
            if cov is None:
                var = log_returns.var() * frequency
            else:
                var = pd.Series(np.diag(cov), index=cov.index)

    # align parameters
    index = mu.index
    var = var.loc[index]
    cov = cov.loc[index, index]

    # convert to output frequency
    #
    # Rebind, never in place. `mu` is the caller's own Series — nothing
    # above reassigns it — so `mu /= step_frequency` divided the caller's
    # data by `step_frequency` and left it that way. Calling this function
    # twice with the same parameters silently ran the second call on
    # values already divided once.
    #
    # `cov` escaped that only because the `.loc` alignment above happens
    # to return a copy: luck, not intent. Drop or change that line and the
    # mutation starts escaping too, so it is written the safe way here as
    # well. `var` was already correct, which is what makes the other two
    # an inconsistency rather than a design.
    #
    # Found 2026-09-09, while verifying an unrelated change: passing the
    # same `mu` to two implementations made their outputs differ by ~3e-4,
    # identically at float32 and float64 — too large, and too
    # precision-independent, to be rounding.
    if step_frequency is None:
        step_frequency = frequency
    mu = mu / step_frequency
    cov = cov / step_frequency
    var = var / step_frequency

    # brownian motion - drift
    drift = mu - 0.5 * var

    # Convert precision string to numpy dtype
    if precision not in ["float32", "float64"]:
        raise ValueError(f"precision must be 'float32' or 'float64', got '{precision}'")
    dtype = getattr(np, precision)

    # Convert inputs to specified dtype for memory efficiency
    drift = drift.astype(dtype)
    if correlated:
        cov = cov.astype(dtype)
        chol = np.linalg.cholesky(cov)
    else:
        var = var.astype(dtype)

    # shock — generate uncorrelated random variables, natively at `dtype`
    # and in column blocks. See `_draw_uncorrelated`.
    size = (len(mu), iterations * steps)
    uncorr_x = _draw_uncorrelated(size, distribution, df, dtype, seed)

    if correlated:
        # `out=` writes the product straight into its destination rather
        # than allocating the result and copying it again in `.astype`.
        shock = np.empty(size, dtype=dtype)
        np.dot(chol, uncorr_x, out=shock)
    else:
        # Broadcasting the column instead of `np.tile`-ing it to the full
        # panel width: same elementwise products, one fewer panel-sized
        # array.
        shock = uncorr_x * np.atleast_2d(np.sqrt(var)).T
    del uncorr_x

    # simulate returns — in place on `shock`.
    #
    # As four chained expressions this allocated a fresh panel for each of
    # the add, the exp, the astype and the subtract, all while `shock` was
    # still alive. On a long horizon that is several GB of temporaries to
    # produce one array of exactly the same size. The operations, their
    # order and their dtype are unchanged, so the result is identical.
    shock += np.atleast_2d(drift).T
    np.exp(shock, out=shock)
    shock -= 1
    returns_sim = shock

    # reshape
    sim = []
    for i in range(len(mu.index)):
        rets_df = pd.DataFrame(returns_sim[i].reshape(steps, iterations))
        rets_df.columns.names = ["sim_no"]
        sim.append(rets_df)

    # concat
    sim = pd.concat(sim, axis=1, names=["ticker"], keys=mu.index)
    sim.index.names = ["step"]
    sim.index += 1

    return sim


def simulations_stats(
    df,
    percentiles=None,
    steps=None,
):
    """Compute summary statistics over simulation paths.

    Returns a DataFrame with stats (End Value, CAGR, Volatility, Sharpe,
    Max Drawdown) at the requested percentiles and steps.
    """
    if percentiles is None:
        percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    # get output
    sim_df = df.unstack(["date", "ticker"])

    # settings
    if steps is None:
        steps = [sim_df.index[-1]]
    group = sim_df.columns.names[1:]

    # stats
    end_value = sim_df.loc[steps].unstack("step").groupby(group + ["step"]).quantile(percentiles)
    cagr = end_value ** (252 / end_value.index.get_level_values("step")) - 1

    dd = (sim_df.cummin().loc[steps].unstack("step").groupby(group + ["step"]).quantile(percentiles)) - 1
    vol = (
        sim_df.pct_change()
        .expanding()
        .std()
        .loc[steps]
        .unstack("step")
        .groupby(group + ["step"])
        .quantile(percentiles)
        .mul(np.sqrt(252))
    )
    sharpe = cagr / vol

    dfs = [end_value, cagr, vol, sharpe, dd]
    result = pd.concat(
        dfs,
        keys=["End Value", "CAGR(%)", "Volatility", "Sharpe", "Max Drawdown"],
        axis=1,
        names=["stat"],
    )

    result.index.names = ["date", "ticker", "step", "percentile"]

    return result
