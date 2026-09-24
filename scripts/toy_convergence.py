"""How fast the centroid, POS and 90 % area of a particle cloud converge with N.

#58 needs a convergence tolerance, and the first proposal was "N is enough when the
centroid moves less than ~1 km". This measures, on a cloud whose true answers are known
exactly, what each statistic the paper reports does as N grows.

    python scripts/toy_convergence.py
    python scripts/toy_convergence.py --spreads 2 10 --seed 20260924

WHAT IT MEASURES. A round Gaussian cloud of spread s km per axis, at N = 1e2 ... 1e6,
repeated over many seeds:
    centroid   rms distance of the sample centroid from the true centre
    POS        a fixed parallel-track search, 185 m swaths every 1 km over the whole
               cloud, so the true POS is 18.5 %; its sd across seeds
    90 % area  fewest 250 m cells holding 90 % of the particles, against the exact
               pi s^2 2 ln 10; its mean and sd across seeds

WHAT IT FOUND, 2026-09-24:
    The centroid settles first: a 1 km rule passes at N ~ 100-200, where POS is still
    +/- 4 points out and the 90 % area is under 10 % of its true size.
    POS noise is binomial and does not depend on cloud size: 4.1 / 1.3 / 0.4 / 0.1 /
    0.04 points at 1e2 ... 1e6.
    The 90 % area is BIASED low, not noisy: 30 % of true at 1e4 for a 10 km cloud, with
    under 1.2 % scatter, so repeating seeds cannot see it. The N it needs scales with
    cloud area / cell area. An independent check reproduced 30 / 82 / 98 %.
Vault: notebook/aditya/2026-09-24 How many particles, and what the cluster can afford.
"""

from __future__ import annotations

import argparse

import numpy as np

SWATH_KM, SPACING_KM, CELL_KM = 0.185, 1.0, 0.25
TRUE_POS = SWATH_KM / SPACING_KM
LADDER = (10**2, 10**3, 10**4, 10**5, 10**6)


def statistics(x: np.ndarray, y: np.ndarray, s: float) -> tuple[float, float, float]:
    """Centroid error (km), POS (fraction) and 90 % area (km^2) of one cloud."""
    centroid_err = float(np.hypot(x.mean(), y.mean()))
    pos = float(np.mean(np.mod(x, SPACING_KM) < SWATH_KM))
    edges = np.arange(-6 * s, 6 * s + CELL_KM, CELL_KM)
    counts, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    ranked = np.sort(counts.ravel())[::-1]
    cells = np.searchsorted(np.cumsum(ranked), 0.9 * len(x)) + 1
    return centroid_err, pos, cells * CELL_KM**2


def repeats(n: int) -> int:
    return 200 if n <= 10**4 else (60 if n == 10**5 else 20)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spreads", type=float, nargs="+", default=[2.0, 10.0],
                    help="cloud spread per axis in km (default 2 and 10)")
    ap.add_argument("--seed", type=int, default=20260924)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    for s in args.spreads:
        true_area = np.pi * s**2 * 2 * np.log(10)
        print(f"\ncloud spread s = {s:g} km per axis | true POS {TRUE_POS:.1%} | "
              f"true 90 % area {true_area:.0f} km2")
        print(f"{'N':>9} {'centroid rms err':>17} {'POS sd (pp)':>12} "
              f"{'area mean/true':>15} {'area sd/true':>13}")
        for n in LADDER:
            r = np.array([statistics(*rng.normal(0, s, (2, n)), s) for _ in range(repeats(n))])
            rms = np.sqrt(np.mean(r[:, 0] ** 2))
            print(f"{n:>9,} {rms:>14.3f} km {100 * r[:, 1].std():>11.2f} "
                  f"{r[:, 2].mean() / true_area:>15.3f} {r[:, 2].std() / true_area:>13.3f}")


if __name__ == "__main__":
    main()
