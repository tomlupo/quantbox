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
import scipy.stats as stats


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
    if step_frequency is None:
        step_frequency = frequency
    mu /= step_frequency
    cov /= step_frequency
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

    # shock — generate uncorrelated random variables
    if distribution == "normal":
        uncorr_x = stats.norm.rvs(size=(len(mu), iterations * steps), random_state=seed).astype(dtype)
    else:
        uncorr_x = stats.t.rvs(df, size=(len(mu), iterations * steps), random_state=seed).astype(dtype)

    if correlated:
        shock = np.dot(chol, uncorr_x).astype(dtype)
    else:
        shock = (uncorr_x * np.tile(np.atleast_2d(np.sqrt(var)).T, uncorr_x.shape[1])).astype(dtype)
    del uncorr_x

    # simulate returns
    returns_sim = np.exp(np.atleast_2d(drift).T + shock).astype(dtype) - 1

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
