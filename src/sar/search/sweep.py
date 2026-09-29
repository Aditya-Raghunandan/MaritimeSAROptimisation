"""sweep.py: how much of the probability one leg of flight finds (issue #45).

This one function is the reward, the POD metric and the scorer for every strategy.
Expanding Square, greedy and PPO are all scored by this code path, which is what keeps the
comparison controlled (D006, D023). Coverage is recorded on the particles, not on a grid
mask, because the water moves 108 m a minute and a mask pinned to the map is stale at once
(ADR002 section 4).

THE RULE. Definite range (D027): anything within half the sweep width, W/2 = 92.6 m, of the
helicopter is seen, and nothing further away is. A seen particle keeps (1 - pod) of its
weight; D027's pod is 1. Weights are never renormalised here, because the reward is the raw
mass removed (ADR002, normalisation row).

DISTANCE TO THE LEG, NOT TO ITS LINE. Measuring to the infinite line through a leg credits
water beyond both ends of it. The swept region is a capsule: a strip W wide along the leg,
closed by a half-disc of radius W/2 at each end.

PARTICLES THAT MOVE DURING THE LEG. In the Gulf Stream a particle moves 108 m in a 60 s
step, more than W/2, so sweeping against where it was at the start of the step can miss a
target the helicopter passed right over, or count one it never reached (ADR003 section 3).
Given the particles' positions at the end of the leg too, the test is the closest point of
approach. The helicopter's position minus the particle's moves in a straight line from r0
to r1 over the leg, so the closest the two come is the distance from the origin to the
segment r0 -> r1. For a particle that does not move this is exactly its distance to the
leg, so one formula covers both cases. The site does the same for its one buoy
(`closestApproach` in frontend/src/searchRun.js). Both motions must be straight over the
leg: a 60 s drift step is, and a caller sweeping a shorter sub-leg (#47, ADR003 section 2)
passes the particles' positions interpolated to the sub-leg's ends.

FLAT EARTH, cos(lat) PER PARTICLE. Distances are metres on a local flat earth centred on
each particle, with cos(lat) at that particle's latitude (ADR002 section 7). Across one leg
of a few kilometres that is right to a few centimetres, against a 92.6 m half-width.

CLI. Sweeps one leg through the centre of a synthetic Gaussian cloud and prints what it
found, as JSON:

    python -m sar.search.sweep --lat 26.5 --lon -79 --spread-km 2 --particles 10000 \\
        --seed 1 --bearing 90 [--length-m 2778] [--sweep-width-m 185.2] [--pod 1.0]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from sar.pipeline.ensemble import synthetic_cloud
from sar.search.platform import SEARCH_SPEED_MS, STEP_S, SWEEP_WIDTH_M
from sar.utils.geo import M_PER_DEG_LAT, east_north, metres_per_degree_lon, offset_position


def relative_m(lat, lon, ref_lat, ref_lon):
    """(east, north) metres from (ref_lat, ref_lon) to (lat, lon), cos(lat) at the reference.

    The inverse of `sar.utils.geo.offset_position`. Longitudes may be in either convention:
    the difference is wrapped to -180..180, so 359.9 and 0.1 are 0.2 degrees apart. Scalar
    or array.
    """
    ref_lat = np.asarray(ref_lat, dtype=float)
    dlat = np.asarray(lat, dtype=float) - ref_lat
    dlon = (np.asarray(lon, dtype=float) - np.asarray(ref_lon, dtype=float) + 180.0) % 360.0
    return (dlon - 180.0) * metres_per_degree_lon(ref_lat), dlat * M_PER_DEG_LAT


def closest_approach_m(r0_east, r0_north, r1_east, r1_north):
    """The least distance from the origin to the segment r0 -> r1, in metres. Vectorised.

    r0 and r1 are the helicopter's position relative to a particle at the start and at the
    end of a leg. The separation moves in a straight line between them, so this is the
    closest the two come. The moment it happens, as a fraction of the leg, is clipped to the
    leg; a separation that does not change is taken at the start.
    """
    r0e, r0n, r1e, r1n = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (r0_east, r0_north, r1_east, r1_north)))
    de, dn = r1e - r0e, r1n - r0n
    dd = de * de + dn * dn
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.where(dd > 0.0, -(r0e * de + r0n * dn) / dd, 0.0)
    tau = np.clip(tau, 0.0, 1.0)
    return np.hypot(r0e + tau * de, r0n + tau * dn)


def _positive(name: str, value) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive number, got {value}")
    return value


def _fix(name: str, value) -> tuple[float, float]:
    """One helicopter position, a finite (lat, lon) pair."""
    fix = np.asarray(value, dtype=float)
    if fix.shape != (2,) or not np.all(np.isfinite(fix)):
        raise ValueError(f"{name} must be one finite (lat, lon) pair, got {value!r}")
    return float(fix[0]), float(fix[1])


def seen(lat, lon, start, end, sweep_width_m: float = SWEEP_WIDTH_M, lat_end=None,
         lon_end=None) -> np.ndarray:
    """Which particles the leg start -> end comes within W/2 of, as a boolean array.

    lat and lon are the particles at the start of the leg. lat_end and lon_end, if given,
    are the same particles at its end, and the test becomes the closest point of approach;
    without them the particles are held still. A distance of exactly W/2 counts as seen, as
    it does on the site. A particle with a NaN position is never seen.
    """
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    if lat.shape != lon.shape:
        raise ValueError(f"lat {lat.shape} and lon {lon.shape} must have the same shape")
    if (lat_end is None) != (lon_end is None):
        raise ValueError("lat_end and lon_end go together: give both or neither")
    if lat_end is None:
        lat_end, lon_end = lat, lon
    else:
        lat_end, lon_end = np.asarray(lat_end, dtype=float), np.asarray(lon_end, dtype=float)
        if lat_end.shape != lat.shape or lon_end.shape != lat.shape:
            raise ValueError(f"end positions {lat_end.shape} and {lon_end.shape} must match "
                             f"the start positions {lat.shape}")
    half_width = _positive("sweep_width_m", sweep_width_m) / 2.0
    start_lat, start_lon = _fix("start", start)
    end_lat, end_lon = _fix("end", end)

    r0 = relative_m(start_lat, start_lon, lat, lon)
    r1 = relative_m(end_lat, end_lon, lat_end, lon_end)
    return closest_approach_m(*r0, *r1) <= half_width


def sweep(lat, lon, weight, start, end, sweep_width_m: float = SWEEP_WIDTH_M,
          pod: float = 1.0, lat_end=None, lon_end=None) -> tuple[np.ndarray, float]:
    """Fly one leg, start -> end, over the particles: returns (weight_after, mass_removed).

    Every particle `seen` has its weight multiplied by 1 - pod. The input array is left as
    it was, and nothing is renormalised, so mass_removed is the probability this leg found.

    #47 flies a step's sub-legs one call each (ADR003 section 2). With pod < 1 a particle
    near a turn is seen by both sub-legs meeting there, in the same pass; at D027's pod = 1
    the second look finds nothing, so it does not matter there.
    """
    weight = np.asarray(weight, dtype=float)
    if weight.shape != np.shape(lat):
        raise ValueError(f"weight {weight.shape} must match the particles {np.shape(lat)}")
    pod = float(pod)
    if not 0.0 <= pod <= 1.0:
        raise ValueError(f"pod must lie within 0 to 1, got {pod}")

    hit = seen(lat, lon, start, end, sweep_width_m, lat_end, lon_end)
    weight_after = np.where(hit, weight * (1.0 - pod), weight)
    return weight_after, float(np.sum(weight - weight_after))


def _cli(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lat", type=float, required=True, help="cloud centre, degrees north")
    parser.add_argument("--lon", type=float, required=True, help="cloud centre, degrees east")
    parser.add_argument("--spread-km", type=float, default=2.0,
                        help="the cloud's standard deviation per axis, default 2 km")
    parser.add_argument("--particles", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bearing", type=float, default=0.0,
                        help="the leg's heading, degrees true; it is centred on the cloud")
    parser.add_argument("--length-m", type=float, default=SEARCH_SPEED_MS * STEP_S,
                        help=f"default {SEARCH_SPEED_MS * STEP_S:.0f} m, one 60 s step at 90 kt")
    parser.add_argument("--sweep-width-m", type=float, default=SWEEP_WIDTH_M,
                        help=f"default {SWEEP_WIDTH_M:.1f} m, 0.1 NM (D027)")
    parser.add_argument("--pod", type=float, default=1.0, help="default 1, definite range (D027)")
    args = parser.parse_args(argv)

    cloud = synthetic_cloud((args.lat, args.lon), args.spread_km, args.particles,
                            np.random.default_rng(args.seed))
    lat, lon = cloud.lat[0], cloud.lon[0]
    half_east, half_north = east_north(args.bearing, _positive("length_m", args.length_m) / 2)
    start = offset_position(args.lat, args.lon, -half_east, -half_north)
    end = offset_position(args.lat, args.lon, half_east, half_north)

    hit = seen(lat, lon, start, end, args.sweep_width_m)
    weight_after, removed = sweep(lat, lon, cloud.weight, start, end, args.sweep_width_m,
                                  args.pod)
    return {"particles": int(lat.size),
            "particles_seen": int(np.count_nonzero(hit)),
            "mass_removed": removed,
            "mass_left": float(np.sum(weight_after)),
            "sweep_width_m": args.sweep_width_m,
            "pod": args.pod,
            "leg": {"start": [float(v) for v in start], "end": [float(v) for v in end],
                    "length_m": args.length_m, "bearing_deg": args.bearing}}


if __name__ == "__main__":
    print(json.dumps(_cli(), indent=2))
