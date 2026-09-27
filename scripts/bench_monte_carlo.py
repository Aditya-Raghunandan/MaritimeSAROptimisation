"""Time full Monte Carlo runs at several N, forced by the HYCOM current and ERA5 wind fetched at the datum."""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

from sar.fetch.current import fetch_current
from sar.fetch.wind import fetch_wind
from sar.model.position import DEFAULT_SIGMA
from sar.pipeline.ensemble import (
    check_cloud,
    describe_run,
    estimate_rows,
    parse_span,
    run_ensemble,
    run_name,
    write_csv,
)
from sar.pipeline.forcing import ConstantForcing

# D024's ladder, one run per rung.
LADDER = (10**2, 10**3, 10**4, 10**5, 10**6)


def fetch_forcing(lat, lon, start) -> tuple[ConstantForcing, dict]:
    """The nearest HYCOM surface current and ERA5 10 m wind to the datum at the start, held steady."""
    when = str(np.datetime64(start, "m"))
    date, hhmm = when[:10], when[11:16]
    t = time.perf_counter()
    current = fetch_current(date, hhmm, lat, lon)
    current_s = time.perf_counter() - t
    t = time.perf_counter()
    wind = fetch_wind(date, hhmm, lat, lon)
    wind_s = time.perf_counter() - t
    uv = (current["water_u"], current["water_v"])
    if not np.isfinite(uv).all():
        raise ValueError(f"HYCOM has no current at {lat}, {lon} on {when}: the datum is on land")
    fetched = {"current_u": uv[0], "current_v": uv[1], "current_time": str(current["time"]),
               "current_source": "HYCOM GLBy0.08/expt_93.0, nearest cell, depth 0 m",
               "wind_u": wind["u10"], "wind_v": wind["v10"], "wind_time": str(wind["time"]),
               "wind_source": "ARCO-ERA5, nearest cell, 10 m",
               "current_fetch_s": current_s, "wind_fetch_s": wind_s}
    return ConstantForcing(uv, (wind["u10"], wind["v10"])), fetched


def bench_name(params) -> str:
    """The run name without its N, since one benchmark holds every N."""
    return re.sub(r"_N\d+", "", run_name({**params, "particles": 1})).replace("ensemble_", "bench_", 1)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lat", type=float, required=True, help="datum latitude, degrees north")
    parser.add_argument("--lon", type=float, required=True, help="datum longitude, degrees east")
    parser.add_argument("--datum-sigma-km", type=float, required=True,
                        help="R1e's datum spread per axis in km, required")
    parser.add_argument("--start", required=True, help="start time, ISO 8601; also when the forcing is read")
    parser.add_argument("--timestep", type=float, required=True, help="step in seconds; D009 fixes 60")
    parser.add_argument("--duration", type=parse_span, required=True, help="how long to track, such as 4h")
    parser.add_argument("--save-every", type=parse_span, required=True,
                        help="keep a state this often, such as 30m; the visualiser's times must be multiples")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help=f"D009's sigma in m/s^0.5, default {DEFAULT_SIGMA:g} (unmeasured)")
    parser.add_argument("--seed", type=int, help="base seed, the same for every N; recorded either way")
    parser.add_argument("--particles", type=int, nargs="+", default=list(LADDER),
                        help="the N to run, default 1e2 to 1e6")
    parser.add_argument("--force", action="store_true",
                        help="write over the size threshold, or over existing CSVs")
    parser.add_argument("--out", required=True,
                        help="data root; the CSVs and the timings JSON go in <out>/derived")
    args = parser.parse_args(argv)

    params = {"lat": args.lat, "lon": args.lon, "start": args.start, "timestep": args.timestep,
              "duration": args.duration, "seed": args.seed}
    derived = Path(args.out) / "derived"
    try:
        for n in args.particles:
            check_cloud(n, args.datum_sigma_km)
            estimate_rows(n, args.duration, args.timestep, args.save_every, force=args.force)
            if (derived / f"{run_name({**params, 'particles': n})}.csv").exists() and not args.force:
                raise ValueError(f"the N = {n} CSV already exists in {derived}; pass --force to replace it")
    except ValueError as error:
        parser.error(str(error))

    forcing, fetched = fetch_forcing(args.lat, args.lon, args.start)
    print(f"current ({fetched['current_u']:.3f}, {fetched['current_v']:.3f}) m/s at "
          f"{fetched['current_time']}, wind ({fetched['wind_u']:.3f}, {fetched['wind_v']:.3f}) m/s "
          f"at {fetched['wind_time']}")

    timings = derived / f"{bench_name(params)}.json"
    record = {"host": platform.node(), "cpu": platform.processor() or platform.machine(),
              "numpy": np.__version__, "python": sys.version.split()[0],
              "arguments": vars(args), "forcing": fetched, "runs": []}
    print(f"{'N':>10} {'run (s)':>9} {'write (s)':>10} {'rows':>12}")
    for n in args.particles:
        spec = {"forcing": forcing, "particles": n, "start": args.start, "lat": args.lat,
                "lon": args.lon, "duration": args.duration, "timestep": args.timestep,
                "datum_sigma_km": args.datum_sigma_km, "sigma": args.sigma,
                "save_every": args.save_every}
        seq = np.random.SeedSequence(args.seed)
        run = {**describe_run(**spec, seed=args.seed, entropy=seq.entropy), "fetched": fetched}
        t = time.perf_counter()
        ensemble = run_ensemble(**spec, seed=seq)
        run_s = time.perf_counter() - t
        t = time.perf_counter()
        path = write_csv(ensemble, run, args.out, force=args.force)
        write_s = time.perf_counter() - t
        rows = int(ensemble.lat.size)
        record["runs"].append({"particles": n, "csv": str(path), "rows": rows, "run_s": run_s,
                               "write_s": write_s, "csv_mb": path.stat().st_size / 1e6})
        # Rewritten after every run, so a run that dies at a large N keeps the smaller ones.
        timings.write_text(json.dumps(record, indent=2, default=str))
        print(f"{n:>10,} {run_s:>9.2f} {write_s:>10.2f} {rows:>12,}")
    print(f"wrote {timings}")
    return record


if __name__ == "__main__":
    main()
