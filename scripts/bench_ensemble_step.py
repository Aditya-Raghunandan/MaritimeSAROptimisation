"""What one ensemble step costs: the drift pipeline plus gridded forcing sampling.

ADR002 section 5 costed a 48 h scenario at 1e6 particles at 504 s from a stand-in step.
This times the real `DriftPipeline.advance` (ConstantForcing, sigma 0.1) plus a vectorised
bilinear-in-space, linear-in-time sample of u and v on a HYCOM-sized (238 x 476) and an
ERA5-sized (77 x 77) grid -- the work the gridded forcing backend, not built yet, will add
to every step.

    python scripts/bench_ensemble_step.py
    python scripts/bench_ensemble_step.py --particles 1000000
    sbatch scripts/bench_ensemble_step.sbatch                 # one task on a compute node
    BENCH_TASKS=4 sbatch --ntasks=4 scripts/bench_ensemble_step.sbatch   # four on one node

WHAT IT FOUND, 2026-09-24 (jobs 58728 and 58729, jaguar11, i7-3770, NumPy 2.5.3):
    one task alone, 1e6 particles:   99 ms pipeline + 364 ms sampling = 463 ms per step,
                                     1,334 s per 48 h run (laptop: 449 ms)
    four tasks on one node, 1e6:     1,245-1,261 ms per step each -- 2.7x slower
The step is memory-bound: a node delivers 1.48x one core, not 4x, and ADR002's 504 s is
2.6x too low alone and 7x with a full node. Sampling is 85 % of the step under load.
Vault: notebook/aditya/2026-09-24 How many particles, and what the cluster can afford.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time

import numpy as np

from sar.pipeline.forcing import ConstantForcing
from sar.pipeline.track import DriftPipeline

GRIDS = {"current": (238, 476), "wind": (77, 77)}   # (lat, lon) cells over the box
STEPS_48H, STEPS_24H = 2880, 1440                  # at the 60 s step (D009)


def median_ms(fn, reps: int) -> float:
    fn()                                           # warm-up
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return 1000.0 * float(np.median(times))


def make_sampler(rng):
    """Bilinear in space, linear in time, u and v, for both products."""
    fields = {k: rng.standard_normal((2, 2, *s)).astype(np.float32) for k, s in GRIDS.items()}

    def sample(lat, lon, frac=0.37):
        out = []
        for k, (ny, nx) in GRIDS.items():
            f = fields[k]
            y = (lat - 17.0) / 19.0 * (ny - 1)
            x = (lon - 278.0) / 19.0 * (nx - 1)
            i = np.clip(y.astype(np.int64), 0, ny - 2)
            j = np.clip(x.astype(np.int64), 0, nx - 2)
            fy, fx = y - i, x - j
            w00, w01 = (1 - fy) * (1 - fx), (1 - fy) * fx
            w10, w11 = fy * (1 - fx), fy * fx
            for c in (0, 1):
                a, b = f[0, c], f[1, c]
                va = w00 * a[i, j] + w01 * a[i, j + 1] + w10 * a[i + 1, j] + w11 * a[i + 1, j + 1]
                vb = w00 * b[i, j] + w01 * b[i, j + 1] + w10 * b[i + 1, j] + w11 * b[i + 1, j + 1]
                out.append((1 - frac) * va + frac * vb)
        return out

    return sample


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--particles", type=int, nargs="+",
                    default=[10**4, 10**5, 10**6, 2 * 10**6])
    args = ap.parse_args(argv)
    rng = np.random.default_rng(1)
    sample = make_sampler(rng)
    t0 = np.datetime64("2019-06-01T06:00")

    print(f"host {platform.node()} | {platform.processor() or platform.machine()} | "
          f"numpy {np.__version__} | py {sys.version.split()[0]}")
    print(f"{'N':>10} {'pipeline ms':>12} {'grid-sample ms':>15} {'total ms/step':>14} "
          f"{'48 h (s)':>9} {'24 h (s)':>9}")
    for n in args.particles:
        pipe = DriftPipeline(ConstantForcing(current=(1.8, 0.0), wind=(5.0, 0.0)),
                             timestep=60.0, sigma=0.1, seed=7)
        pos = pipe.start_positions(np.full(n, 26.5), np.full(n, 281.0))
        pos = pos + rng.normal(0, 0.05, (n, 2))
        reps = 20 if n <= 10**5 else 6
        p_ms = median_ms(lambda: pipe.advance(pos, t0), reps)
        g_ms = median_ms(lambda: sample(pos[:, 0], pos[:, 1]), reps)
        total = p_ms + g_ms
        print(f"{n:>10,} {p_ms:>12.1f} {g_ms:>15.1f} {total:>14.1f} "
              f"{total * STEPS_48H / 1000:>9.0f} {total * STEPS_24H / 1000:>9.0f}")


if __name__ == "__main__":
    main()
