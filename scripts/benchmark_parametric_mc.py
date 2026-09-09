#!/usr/bin/env python
"""Evidence for the `parametric_mc` native-dtype draw, re-runnable from the repo.

The two claims that branch made were produced by scratch scripts living
outside the tree, which a reviewer cannot re-run and a future reader cannot
check. They live here instead.

    python scripts/benchmark_parametric_mc.py memory
    python scripts/benchmark_parametric_mc.py equivalence

`memory` measures peak RSS. It runs ONE variant per process — `ru_maxrss` is
a high-water mark for the whole process, so two variants in one process would
report the larger of them twice — and re-executes itself to do that. The
`--variant` flag is how the child says which one it is; you do not normally
pass it.

`baseline` is the pre-2026-09 draw: `scipy.stats` at float64 across the full
panel, downcast afterwards. It is transcribed here rather than imported from
git history so the comparison stays runnable once this is merged.

`equivalence` is the argument that the new draw is the same SIMULATOR, which
is what matters given that it deliberately does not reproduce the old stream.
Sample mean and standard deviation are deliberately not compared: at df=3
their seed-to-seed spread is enormous and they discriminate nothing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats

from quantbox.features.simulations import _draw_uncorrelated

# The size that killed a production batch on 2026-09-09: 4 assets, a horizon
# of 100_800_000 columns, float32, Student-t with df=3 (robo's configuration).
PROD_ASSETS = 4
PROD_ITERATIONS = 672_000
PROD_STEPS = 150


def _baseline_draw(size, distribution, df, dtype, seed):
    """The draw as it was before this change: float64 panel, then downcast."""
    if distribution == "normal":
        out = stats.norm.rvs(size=size, random_state=seed)
    else:
        out = stats.t.rvs(df, size=size, random_state=seed)
    return out.astype(dtype, copy=False)


def _panel(mu, cov, draw, iterations, steps, dtype, distribution, df, seed):
    """`parametric_mc`'s body around whichever draw it is handed.

    Transcribed from `quantbox.features.simulations.parametric_mc`, correlated
    path only, as of the commit that introduced this script. It is a COPY, and
    nothing here notices if that function changes — if the memory numbers ever
    stop making sense, diff this against the original first. Deliberately not
    pinned to a branch SHA: this branch squash-merges, so any SHA named here
    would be unresolvable in `dev` the moment it landed.
    """
    var = pd.Series(np.diag(cov), index=cov.index).loc[mu.index] / 252
    mu = mu / 252
    cov = cov.loc[mu.index, mu.index] / 252
    drift = (mu - 0.5 * var).astype(dtype)
    chol = np.linalg.cholesky(cov.astype(dtype))

    size = (len(mu), iterations * steps)
    uncorr_x = draw(size, distribution, df, dtype, seed)
    shock = np.empty(size, dtype=dtype)
    np.dot(chol, uncorr_x, out=shock)
    del uncorr_x
    shock += np.atleast_2d(drift).T
    np.exp(shock, out=shock)
    shock -= 1
    return shock


def _toy_parameters(n_assets, seed=0):
    rng = np.random.default_rng(seed)
    tickers = [f"A{i}" for i in range(n_assets)]
    a = rng.normal(size=(n_assets, n_assets))
    cov = pd.DataFrame(a @ a.T / 100, index=tickers, columns=tickers)
    mu = pd.Series(rng.uniform(0.02, 0.09, n_assets), index=tickers)
    return mu, cov


def _run_memory_child(args):
    import resource

    draw = _baseline_draw if args.variant == "baseline" else _draw_uncorrelated
    mu, cov = _toy_parameters(args.assets)
    dtype = getattr(np, args.precision)

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    panel = _panel(
        mu,
        cov,
        draw,
        args.iterations,
        args.steps,
        dtype,
        args.distribution,
        args.df,
        np.random.default_rng(2026),
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert panel.dtype == dtype
    # ru_maxrss is KiB on Linux, bytes on macOS. Only Linux is claimed here.
    print(f"{args.variant}\t{before / 1024**2:.3f}\t{peak / 1024**2:.3f}")


def cmd_memory(args):
    if args.variant:
        return _run_memory_child(args)

    base = [sys.executable, __file__, "memory"]
    for flag in ("assets", "iterations", "steps", "precision", "distribution", "df"):
        base += [f"--{flag}", str(getattr(args, flag))]

    results = {}
    for variant in ("baseline", "native"):
        out = subprocess.run([*base, "--variant", variant], capture_output=True, text=True)
        if out.returncode != 0:
            # The baseline child deliberately allocates the float64 panel this
            # branch exists to remove — ~6 GiB at the default size — so being
            # OOM-killed is a likely outcome on a small box and must not
            # surface as a bare CalledProcessError with the reason captured
            # and thrown away.
            sys.stderr.write(out.stderr)
            raise SystemExit(
                f"variant {variant!r} exited {out.returncode}. At the default size the baseline "
                f"child needs ~6.2 GiB and the native one ~3.2 GiB; try a smaller --iterations."
            )
        name, before, peak = out.stdout.strip().split("\t")
        results[name] = (float(before), float(peak))

    cols = args.assets, args.iterations * args.steps
    print(f"peak RSS, one fresh process per variant, ({cols[0]}, {cols[1]}) {args.precision} {args.distribution}")
    for name, (before, peak) in results.items():
        print(f"  {name:9s} baseline={before:6.3f} GiB  peak={peak:6.3f} GiB  allocated={peak - before:6.3f} GiB")

    saved = results["baseline"][1] - results["native"][1]
    ratio = results["baseline"][1] / results["native"][1]
    float64_panel = cols[0] * cols[1] * 8 / 1024**3
    print(f"  saved={saved:.3f} GiB  ratio={ratio:.2f}x")
    print(f"  for reference, the float64 panel that no longer exists is {float64_panel:.3f} GiB")


def _bootstrap_quantile_se(sample, qs, replicates, rng):
    """Standard error of each quantile of `sample`, at `sample`'s OWN size.

    Resampling at a smaller size would estimate the standard error of a
    smaller estimator and understate how demanding the comparison is.
    """
    draws = np.empty((replicates, len(qs)))
    for i in range(replicates):
        draws[i] = np.quantile(rng.choice(sample, len(sample)), qs)
    return draws.std(axis=0)


def cmd_equivalence(args):
    """Same distributions, different stream — checked where it can be checked.

    Per seed and per quantile, |baseline - native| against 3x the pooled
    standard error of the two quantile estimates. A quantile's standard error
    is estimated by bootstrap rather than assumed normal, because at df=3 the
    tail quantiles are not close to normal.

    The bootstrap resamples at the FULL sample size, which is the whole point
    and was got wrong once: resampling a 200_000-element subsample to estimate
    the standard error of a quantile of 2_000_000 values overstates that error
    by sqrt(10), because a quantile's standard error scales as 1/sqrt(n). The
    threshold then reads as "3 standard errors" while actually being ~9.5, so
    the check silently could not fail.

    Replicates are looped rather than vectorised into one (bootstrap, n) array,
    which at the defaults would be ~640 MiB. Measured with `tracemalloc`, the
    loop peaks at THREE times the sample, not one: `Generator.choice` with
    replacement materialises an int64 index array — twice a float32 sample's
    bytes on its own — plus the taken array, and `np.quantile` then copies
    again to partition. (7.63 MiB sample -> 22.89 MiB peak above baseline,
    3.00x.) Still the right trade at `bootstrap` replicates, but the number is
    3x, and this file has now had two comments that were wrong because nobody
    measured them.

    `--bootstrap` defaults to 40, which gives the SE estimate ~11% relative
    error of its own (1/sqrt(2(B-1))), so the "3 SE" threshold wobbles by
    roughly +-0.3 SE between runs. That is far too coarse to matter against the
    separation actually observed — null worst-z ~2.1, a t(5)-vs-t(3) positive
    control ~40-100 — but do not read the threshold as sharper than it is.
    """
    qs = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    dtype = getattr(np, args.precision)
    size = (args.assets, args.columns)
    rng_boot = np.random.default_rng(12345)

    worst = 0.0
    failures = 0
    for seed in range(args.seeds):
        a = _baseline_draw(size, args.distribution, args.df, dtype, np.random.default_rng(1000 + seed)).ravel()
        b = _draw_uncorrelated(size, args.distribution, args.df, dtype, np.random.default_rng(2000 + seed)).ravel()

        qa = np.quantile(a, qs)
        qb = np.quantile(b, qs)

        boot_a = _bootstrap_quantile_se(a, qs, args.bootstrap, rng_boot)
        boot_b = _bootstrap_quantile_se(b, qs, args.bootstrap, rng_boot)
        pooled = np.sqrt(boot_a**2 + boot_b**2)

        z = np.abs(qa - qb) / np.where(pooled == 0, np.inf, pooled)
        worst = max(worst, float(z.max()))
        if z.max() > 3:
            failures += 1
            bad = int(np.argmax(z))
            print(f"  seed {seed}: q={qs[bad]} baseline={qa[bad]:.6g} native={qb[bad]:.6g} z={z.max():.2f}")

    print(
        f"{args.seeds} seeds x {len(qs)} quantiles, {args.distribution} df={args.df} "
        f"{args.precision}, {size[0]}x{size[1]} draws each"
    )
    print(f"  worst |difference| in pooled standard errors: {worst:.2f}")
    print(f"  seeds exceeding 3 SE: {failures}")
    return 1 if failures else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("memory", help="peak RSS, baseline draw vs native draw")
    m.add_argument("--assets", type=int, default=PROD_ASSETS)
    m.add_argument("--iterations", type=int, default=PROD_ITERATIONS)
    m.add_argument("--steps", type=int, default=PROD_STEPS)
    m.add_argument("--precision", default="float32", choices=["float32", "float64"])
    m.add_argument("--distribution", default="student-t", choices=["normal", "student-t"])
    m.add_argument("--df", type=float, default=3)
    m.add_argument("--variant", choices=["baseline", "native"], help=argparse.SUPPRESS)
    m.set_defaults(func=cmd_memory)

    e = sub.add_parser("equivalence", help="same simulator, different stream")
    e.add_argument("--assets", type=int, default=2)
    e.add_argument("--columns", type=int, default=1_000_000)
    e.add_argument("--seeds", type=int, default=8)
    e.add_argument("--bootstrap", type=int, default=40)
    e.add_argument("--precision", default="float32", choices=["float32", "float64"])
    e.add_argument("--distribution", default="student-t", choices=["normal", "student-t"])
    e.add_argument("--df", type=float, default=3)
    e.set_defaults(func=cmd_equivalence)

    args = p.parse_args()
    raise SystemExit(args.func(args) or 0)


if __name__ == "__main__":
    main()
