"""fit_random_velocity.py: is the drift error a random walk, or a velocity error with memory?

The engine's random term is a random walk: a fresh push of sigma sqrt(dt) every step, so
the spread grows as sqrt(t). The dev drifters say otherwise: their gap to the model grows as
t^0.87-0.95 (beta = 1.74-1.91 on the squared gap, docs/sigma-calibration.md). That is the
signature of a velocity error that persists.

Taylor (1921, "Diffusion by continuous movements") gives the spread of a particle whose
velocity error is random with standard deviation sigma_u per axis and loses its memory
exponentially over T_L (a first-order Markov or "random flight" model, Griffa 1996):

    s(t)^2 = 2 sigma_u^2 T_L [t - T_L (1 - exp(-t / T_L))]

which is sigma_u t while t << T_L (straight-line motion) and sqrt(2 sigma_u^2 T_L t) once
t >> T_L (a random walk). This fits that curve, and the random walk's s = sigma sqrt(t), to
the drifters' measured spread at every horizon in the calibration's fit.json files. The
spread is the quantile sigma times sqrt(t): the radius a round Gaussian needs to hold 90 %
of the buoys at that horizon.

Two checks that do not use the fit:
  * sigma_u against HYCOM's own velocity error, measured separately (lagcheck.json:
    HYCOM's current at the start against the buoy's velocity, RMS of the difference);
  * with --search, what each model does to a search: the Expanding Square and the Sector
    Search over clouds grown from one point by each model (D026), at a few arrival times.

    python scripts/fit_random_velocity.py --fit derived/sigma/h4/fit.json \\
        derived/sigma/fit.json --lagcheck derived/sigma/lagcheck.json [--search]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import least_squares

from sar.model.position import CALIBRATED_SIGMA
from sar.pipeline.ensemble import Ensemble
from sar.search.episode import STEPS, SearchEpisode, pattern_policy, run
from sar.search.patterns import MarkerTrack, expanding_square, sector_search
from sar.search.platform import STEP_S, SWEEP_WIDTH_M
from sar.utils.geo import offset_position

TIER = "undrogued, alpha=0.02"


def measured_spread(paths) -> dict:
    """Horizon (h) -> 90 % spread per axis (m), from every fit.json given; first one wins."""
    out = {}
    for path in paths:
        table = json.load(open(path))["table_a"][TIER]
        for hour, row in table.items():
            out.setdefault(float(hour), row["sigma_quantile"]["value"] * np.sqrt(float(hour) * 3600))
    return dict(sorted(out.items()))


def random_velocity_spread(t_s, sigma_u, t_l_s):
    t_s = np.asarray(t_s, dtype=float)
    return np.sqrt(2 * sigma_u ** 2 * t_l_s * (t_s - t_l_s * (1 - np.exp(-t_s / t_l_s))))


def fit(spread: dict) -> dict:
    hours = np.array(list(spread))
    t, s = hours * 3600, np.array(list(spread.values()))
    rv = least_squares(lambda p: np.log(random_velocity_spread(t, p[0], p[1] * 3600)) - np.log(s),
                       x0=[0.2, 10.0], bounds=([1e-3, 0.1], [5.0, 1e4]))
    rw = least_squares(lambda p: np.log(p[0] * np.sqrt(t)) - np.log(s), x0=[30.0])
    sigma_u, t_l_h = rv.x
    return {
        "hours": hours.tolist(),
        "measured_spread_m": s.tolist(),
        "random_velocity": {
            "sigma_u_ms": float(sigma_u), "t_l_h": float(t_l_h),
            "spread_m": random_velocity_spread(t, sigma_u, t_l_h * 3600).tolist(),
            "rms_log_error": float(np.sqrt(np.mean(rv.fun ** 2)))},
        "random_walk_best_single_sigma": {
            "sigma": float(rw.x[0]), "spread_m": (rw.x[0] * np.sqrt(t)).tolist(),
            "rms_log_error": float(np.sqrt(np.mean(rw.fun ** 2)))},
        "random_walk_engine": {
            "sigma": CALIBRATED_SIGMA, "spread_m": (CALIBRATED_SIGMA * np.sqrt(t)).tolist()},
    }


def velocity_check(path) -> dict:
    """HYCOM's velocity error at the start, from the lag check, per axis."""
    row = json.load(open(path))["by_tier"]["undrogued"]["0"]
    return {"rms_vector_ms": row["rms_diff_ms"], "rms_per_axis_ms": row["rms_diff_ms"] / np.sqrt(2),
            "windows": row["n"]}


def cloud(model: str, arrival_steps: int, n: int, rng, sigma_u: float, t_l_s: float,
          sigma: float = CALIBRATED_SIGMA) -> Ensemble:
    """A cloud grown from one point by `model`, kept for the search window from arrival."""
    x = np.zeros((n, 2))
    u = rng.standard_normal((n, 2)) * sigma_u
    keep = np.exp(-STEP_S / t_l_s)
    frames = []
    for k in range(arrival_steps + STEPS + 1):
        if k >= arrival_steps:
            frames.append(x.copy())
        if model == "random velocity":
            x = x + u * STEP_S
            u = keep * u + np.sqrt(1 - keep ** 2) * sigma_u * rng.standard_normal((n, 2))
        else:
            x = x + sigma * np.sqrt(STEP_S) * rng.standard_normal((n, 2))
    p = np.array(frames)
    lat, lon = offset_position(26.5, 281.0, p[..., 0], p[..., 1])
    t = np.arange(STEPS + 1) * STEP_S
    times = np.datetime64("2019-06-01T00:00", "us") + (t * 1e6).astype("timedelta64[us]")
    return Ensemble(times, lat, lon, np.full(n, 1.0 / n), np.zeros(lat.shape, dtype=bool))


def search(sigma_u: float, t_l_h: float, n: int, seeds: int, arrivals_h) -> list:
    marker = MarkerTrack.fixed(26.5, 281.0, STEPS * STEP_S)
    patterns = {"expanding_square": expanding_square(SWEEP_WIDTH_M, 0.0),
                "sector_search": sector_search(first_bearing_deg=0.0)}
    rows = []
    for hours in arrivals_h:
        steps = int(round(hours * 3600 / STEP_S))
        for model in ("random walk", "random velocity"):
            pos = {name: [] for name in patterns}
            spread = []
            for seed in range(1, seeds + 1):
                window = cloud(model, steps, n, np.random.default_rng(seed), sigma_u, t_l_h * 3600)
                east = (window.lon[0] - 281.0) * 111_195 * np.cos(np.radians(26.5))
                spread.append(float(np.std(east)))
                for name, pattern in patterns.items():
                    # Turning at once, as when D030's numbers were recorded (6 Oct, before D032).
                    pos[name].append(run(pattern_policy(pattern),
                                         SearchEpisode(window, marker,
                                                       turn_rate_deg_s=float("inf")))["pos"])
            rows.append({"arrival_h": hours, "model": model,
                         "spread_at_arrival_m": float(np.mean(spread)),
                         **{f"{name}_pos": float(np.mean(v)) for name, v in pos.items()}})
    return rows


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fit", nargs="+", required=True, help="calibration fit.json files")
    parser.add_argument("--lagcheck", help="lagcheck.json, for the velocity check")
    parser.add_argument("--search", action="store_true", help="also fly both patterns")
    parser.add_argument("--particles", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--arrival-h", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    args = parser.parse_args(argv)

    result = fit(measured_spread(args.fit))
    if args.lagcheck:
        result["velocity_check"] = velocity_check(args.lagcheck)
    if args.search:
        rv = result["random_velocity"]
        result["search"] = search(rv["sigma_u_ms"], rv["t_l_h"], args.particles, args.seeds,
                                  args.arrival_h)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
