"""Parametric Monte Carlo simulations and summary statistics.

These are the two building blocks historically vendored into host projects
(see e.g. robo's ``src/market/simulations.py``). They are pure numpy and
pandas — the shocks are drawn from ``numpy.random.Generator`` directly, so
they are produced at the caller's precision with any float64 intermediate
bounded to one block rather than the whole panel.

``parametric_mc`` generates a GBM-style correlated simulation of asset
returns; ``simulations_stats`` produces standard summary quantiles
(end value, CAGR, vol, Sharpe, max drawdown) from the output panel.

Callers are expected to pass a seeded ``numpy.random.Generator`` as the
``seed`` argument for reproducibility — a bare integer seed is accepted
for backwards compatibility but deterministic use requires a Generator.

``numpy.random.RandomState`` is NOT accepted. It was, implicitly, while
the shocks came from ``scipy.stats``'s ``random_state``; drawing them from
a Generator narrowed that, and a ``RandomState`` now raises from
``default_rng``. No known caller passes one, but a vendored copy might.

``seed=None`` also changed, and silently. Under ``scipy.stats`` it fell
through to numpy's global singleton, so a host doing ``np.random.seed(42)``
before the call got a reproducible panel. ``default_rng(None)`` seeds from
OS entropy and ignores the global state, so that host now gets a different
panel every run with no error. Pass a Generator.
"""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd


def _draw_uncorrelated(size, distribution, df, dtype, seed, target_bytes=128 * 1024**2):
    """Fill an ``(n_assets, n_cols)`` panel of iid shocks at *dtype*.

    Any float64 intermediate is bounded to ONE BLOCK rather than the whole
    panel. That is the achievement, and it is deliberately weaker than "no
    float64 stage": the Student-t path — robo's only production
    configuration, at float32 — assembles each block in float64 and casts
    on assignment.

    ``scipy.stats`` has no dtype argument, so ``norm.rvs`` / ``t.rvs``
    always materialise float64 across the FULL panel, and a float32 caller
    then pays for a second full-size array to downcast into. On a long
    horizon that is the single largest allocation in ``parametric_mc``: a
    (4, 100_800_000) panel is 3.0 GiB in float64 to produce a 1.5 GiB
    float32 result, and it is what a production batch died on (2026-09-09,
    MemoryError with ~10 GiB free).

    ``numpy``'s Generator does take a dtype — for ``standard_normal`` and
    ``standard_gamma``, though not for ``standard_t`` or ``chisquare`` — so
    Student-t is assembled from those two as ``Z / sqrt(X / df)`` with
    ``X ~ chi2(df) = 2 * Gamma(df/2)``.

    This changes the random stream: output is drawn from the same
    distributions with the same parameters, but the numbers differ, exactly
    as reseeding would. It is not a drop-in for a byte-identical rerun.
    """
    # Validated here rather than in the caller: this is where the arguments
    # are used, and the test suite — and any vendored copy — reaches this
    # helper directly.
    if distribution not in ("normal", "student-t"):
        raise ValueError(f"distribution must be 'normal' or 'student-t', got {distribution!r}")
    if distribution != "normal":
        # `bool` is excluded deliberately: it is an int subclass, so df=True
        # would otherwise pass and silently draw t(1) — Cauchy, the widest
        # possible tail — from a plainly wrong argument. numpy scalars must
        # pass, which rules out a bare isinstance(df, (int, float)): np.int64
        # is neither. And `inf` must NOT pass: standard_gamma(inf) returns
        # inf, `2.0/inf` is 0.0, and inf*0.0 is nan — so the panel comes
        # back silently all-NaN. (A too-SMALL df is silent garbage too, but
        # there is no clean floor to pick, so that is caught after the draw
        # instead. The block check would now catch inf as well, making this
        # branch belt-and-braces rather than the only guard.)
        #
        # This is not "restoring what scipy did", which an earlier version
        # of this comment claimed. scipy raised its domain error only for
        # 0, -3 and nan. Measured against the seven cases tested: two are
        # faults scipy let through entirely (`inf` returned NaN silently,
        # True quietly drew Cauchy) and two are better messages for things
        # it already rejected as TypeErrors ("3", None).
        # `numbers.Real` rather than `float(df)`: coercing would accept the
        # STRING "3", which passes a numeric check and then fails four lines
        # later on `df / 2.0`. numpy registers its scalar types with the
        # numbers ABCs, so np.int64 and np.float32 pass here.
        if isinstance(df, bool) or not isinstance(df, numbers.Real) or not np.isfinite(float(df)) or float(df) <= 0:
            raise ValueError(f"df must be a finite positive number, got {df!r}")

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    n_assets, n_cols = size
    out = np.empty(size, dtype=dtype)

    # Block sized so the transient cost is bounded instead of scaling with
    # the horizon the way a second full panel would.
    #
    # `target_bytes` is nominal in BOTH directions, so do not read the name
    # literally: the divisor is hard-coded at float64's 8 bytes, so the
    # Student-t path holds `z` and `g` at once and lives at 2x, while the
    # normal float32 path lives at 0.5x, because the only block-sized
    # allocation on it is the draw itself: `standard_normal(dtype=dtype)`
    # produces the block natively and assigning it into the strided column
    # slice is a direct copy, with no second full-block temporary.
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
        # Assembled in float64 even when the panel is float32. This is a
        # precaution, not a fix for an observed fault, and it is worth
        # being accurate about which: drawing it natively in float32 was
        # measured at (4, 15_000_000), df=3, and produced no zeros, no
        # infinities and a healthy sample (max |t| ~2.7e3, std 1.77). An
        # earlier note here claimed a fifteen-order-of-magnitude blow-up
        # in that configuration. That was a misreading — the statistic in
        # question was the sample mean of a t(3) panel, whose seed-to-seed
        # spread covers many orders of magnitude on its own, compared
        # across a single seed each way.
        #
        # Overflow is not the reason either: a float32 quotient would need
        # g < ~1e-77 * z^2, which is p ~ 1e-38 at df=1 and ~1e-115 at
        # df=3. It does not happen, and an earlier draft of this comment
        # that appealed to "headroom before the result is representable"
        # was confusing significand digits with exponent range.
        #
        # Nor is it the division's own accuracy: IEEE division is correctly
        # rounded to within half an ulp whatever the divisor's magnitude, so
        # a small `g` does not degrade `z / g`.
        #
        # What float64 actually buys is UPSTREAM of the division —
        # resolution of the gamma draw itself near zero. A float32
        # `standard_gamma` at shape df/2 quantises in a region where float64
        # still resolves the value, and can land on exactly 0.0, which is
        # what would turn a finite t into `inf`. The block is bounded, so
        # this costs a temporary rather than a second panel: a cheap margin
        # on the one input that can be arbitrarily small, not a fix for
        # anything observed.
        z = rng.standard_normal(shape)
        g = rng.standard_gamma(df / 2.0, shape)
        g *= 2.0 / df
        np.sqrt(g, out=g)
        # Suppressed so the check below is the SINGLE failure path. Without
        # this, a host running under `np.seterr(all="raise")` gets a bare
        # FloatingPointError and never reaches the message naming df and
        # dtype. Both operations are covered deliberately: a tiny df trips
        # divide/invalid in the quotient, and a merely small one trips
        # overflow in the float32 cast on assignment.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            z /= g
            out[:, start:stop] = z

        # Check the property, rather than enumerating the arguments that
        # violate it.
        #
        # A previous version rejected `df=inf` because it yields a silently
        # all-NaN panel, and then accepted `df=0.1`, which yields 1105
        # non-finite entries in 10M, and `df=1e-8`, which yields 99.99%
        # of them — the same silent garbage, waved through. As df falls,
        # `standard_gamma` concentrates near zero, and the quotient either
        # overflows the panel's dtype on the cast or divides by an exact
        # zero. There is no clean floor to pick: it depends on df, on
        # dtype, and on how many values are drawn.
        #
        # So check the output instead. Be precise about what that is worth:
        # this is a SAMPLING test, not validation. Whether a bad df is
        # caught depends on how many values are drawn — at df=0.1 roughly
        # 1 in 10_000 draws is non-finite, so a production horizon raises
        # reliably while a 2_000-value toy call passes about four times in
        # five. It is a high-probability backstop at production sizes, not
        # a domain proof, and it does not make the RESULT safe end to end:
        # a value that survives the cast as finite-but-huge still overflows
        # in `parametric_mc`'s `np.exp` further down with no error here.
        if not np.isfinite(out[:, start:stop]).all():
            msg = (
                f"Student-t draw produced non-finite values at df={df!r}, "
                f"dtype={np.dtype(dtype).name}. The chi-square denominator "
                f"reached zero or the quotient overflowed — df is too small "
                f"to sample at this precision."
            )
            # Drop the panel BEFORE raising. `out` is 1.5 GiB at the size
            # this function exists to make affordable, and Python keeps the
            # raising frame alive on the exception's __traceback__ — so a
            # host that catches and logs the stack (Prefect does) would pin
            # exactly the allocation this branch was written to avoid.
            del out, z, g
            raise ValueError(msg)

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
    # `distribution` and `df` are validated in `_draw_uncorrelated`, where
    # they are used — scipy used to reject a bad df with "Domain error in
    # arguments" and drawing the t ourselves lost that. Putting the check
    # there rather than here means the private helper is guarded too, which
    # matters because the test suite and any vendored copy call it directly.
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
