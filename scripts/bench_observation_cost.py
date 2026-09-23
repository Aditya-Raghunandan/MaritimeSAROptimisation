"""What one reinforcement-learning step costs, per observation choice (D023).

D023 asks whether the agent has to look at the probability map at all, given that the
reward is computed on particle weights and never on a grid. Part of that answer is cost,
and cost is something to measure rather than assume.

    python scripts/bench_observation_cost.py
    python scripts/bench_observation_cost.py --reps 50

WHAT IT MEASURES. Three things an environment step might do, timed separately:

    sweep     cut the weight of every particle within W/2 of the track segment just
              flown. EVERY arm pays this -- it is the reward, and it is what makes
              coverage drift with the water (ADR002 section 4)
    bin       build the raster observation, one bincount pass
    summary   build the low-dimensional alternative: remaining mass, centroid,
              covariance. Six reductions over the same array

WHAT IT FOUND, and it is the opposite of what was expected. At 1e6 particles the
summary vector is MORE expensive than the raster (33.4 ms against 19.0 ms), because a
mean and a covariance are six passes while binning is one. The observation is not the
cost of an RL step; the sweep is, and every arm pays it. The cost driver is N, not the
choice of observation -- which is why training subsamples to 1e5 and evaluation does not.

Measured 2026-09-23, single core: at 1e5, sweep 2.15 ms / bin 1.74 ms / summary 1.85 ms;
at 1e6, 35.2 / 19.0 / 33.4 ms. Numbers quoted in `decisions/D023` in the vault.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

SWEEP_HALF_WIDTH = 0.002       # fraction of the box; stands in for W/2
EPISODE_STEPS = 45             # a 45-minute window at 60 s (D007)


def sweep(x, y, w, segment, half_width):
    """Zero the weight of particles within `half_width` of a track segment.

    Point-to-segment distance, clamped at both ends so the turn at a waypoint does not
    sweep a disc. This is the reward: the weight removed is the newly swept mass.
    """
    (x0, y0), (x1, y1) = segment
    dx, dy = x1 - x0, y1 - y0
    t = np.clip(((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
    d2 = (x - (x0 + t * dx)) ** 2 + (y - (y0 + t * dy)) ** 2
    w[d2 < half_width * half_width] = 0.0
    return w


def bin_map(x, y, w, n):
    """The raster observation: weights summed into an n x n grid, one pass."""
    i = np.clip((x * n).astype(np.int64), 0, n - 1)
    j = np.clip((y * n).astype(np.int64), 0, n - 1)
    return np.bincount(i * n + j, weights=w, minlength=n * n).reshape(n, n)

def summary(x, y, w):
    """The low-dimensional alternative: mass, centroid and covariance. Six reductions."""
    m = w.sum()
    mx, my = (w * x).sum() / m, (w * y).sum() / m
    vx, vy = (w * (x - mx) ** 2).sum() / m, (w * (y - my) ** 2).sum() / m
    vxy = (w * (x - mx) * (y - my)).sum() / m
    return np.array([m, mx, my, vx, vy, vxy])


def bench(fn, *args, reps: int) -> float:
    """Milliseconds per call, after one warm-up so allocation is not counted."""
    fn(*args)
    start = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    return (time.perf_counter() - start) / reps * 1000.0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=20, help="timed calls per measurement")
    ap.add_argument("--counts", type=int, nargs="+", default=[100_000, 1_000_000])
    args = ap.parse_args(argv)

    rng = np.random.default_rng(0)
    segment = ((0.4, 0.4), (0.6, 0.45))

    print(f"{'N':>10} {'sweep':>9} {'bin 400^2':>11} {'bin 200^2':>11} {'summary':>9}"
          "   ms per environment step")
    rows = []
    for n in args.counts:
        x, y = rng.random((2, n))
        w = np.ones(n)
        sw = bench(sweep, x, y, w.copy(), segment, SWEEP_HALF_WIDTH, reps=args.reps)
        b4 = bench(bin_map, x, y, w, 400, reps=args.reps)
        b2 = bench(bin_map, x, y, w, 200, reps=args.reps)
        su = bench(summary, x, y, w, reps=args.reps)
        rows.append((n, sw, b4, su))
        print(f"{n:10,} {sw:8.2f}ms {b4:10.2f}ms {b2:10.2f}ms {su:8.2f}ms")

    print("\nTraining one arm, one seed, environment time only, single core:")
    for budget in (1e6, 1e7):
        episodes = budget / EPISODE_STEPS
        print(f"  {budget:.0e} env steps = {episodes:,.0f} episodes")
        for n, sw, b4, _ in rows:
            hours = (sw + b4) * EPISODE_STEPS / 1000 * episodes / 3600
            print(f"      N = {n:>9,}  ->  {hours:8.1f} h")

    print("\nReward noise from subsampling, by the mass one step removes:")
    for p in (0.001, 0.005, 0.02):
        line = "  ".join(f"N={n:>9,}: {100 / (p * n) ** 0.5:5.1f} %" for n, *_ in rows)
        print(f"  {p:6.3f} of the mass ->  {line}")


if __name__ == "__main__":
    main()
