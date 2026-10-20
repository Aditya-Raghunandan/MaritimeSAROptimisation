"""ensemble_c2.py: sar.pipeline.ensemble's Monte Carlo through DriftPipelineC2, the current doubled."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sar.model.position import DEFAULT_SIGMA, DEFAULT_SIGMA_U, MEMORY_TIME_S
from sar.pipeline.ensemble import (
    Ensemble,
    _stack_forcing,
    add_run_arguments,
    as_seed_sequence,
    check_cloud,
    estimate_rows,
    forcing_from,
    parse_span,
    random_term,
    resolve_sigma_u,
    run_name,
    save_stride,
    spawn_positions,
    write_csv,
)
from sar.pipeline.track_c2 import DriftPipelineC2
from sar.utils.geo import to_display_longitude


def run_ensemble(forcing, particles, start, lat, lon, duration, timestep, datum_sigma_km,
                 sigma=DEFAULT_SIGMA, seed=None, save_every=None, forcing_log=None,
                 sigma_u=DEFAULT_SIGMA_U, memory_time_s=MEMORY_TIME_S) -> Ensemble:
    """sar.pipeline.ensemble.run_ensemble with DriftPipelineC2 in place of DriftPipeline."""
    datum_seq, kick_seq = as_seed_sequence(seed).spawn(2)
    starts = spawn_positions(lat, lon, particles, datum_sigma_km,
                             np.random.default_rng(datum_seq))
    pipeline = DriftPipelineC2(forcing, timestep, sigma=sigma, seed=kick_seq, sigma_u=sigma_u,
                               memory_time_s=memory_time_s)
    n_steps = pipeline.step_count(duration)
    stride = save_stride(save_every, pipeline.timestep)

    saved = [k for k in range(n_steps + 1) if k % stride == 0 or k == n_steps]
    n = int(particles)
    times = np.empty(len(saved), dtype="datetime64[us]")
    lats, lons = np.empty((len(saved), n)), np.empty((len(saved), n))
    beached = np.zeros((len(saved), n), dtype=bool)
    logged, row = [], 0
    stuck = np.zeros(n, dtype=bool)
    for state in pipeline.track(start, starts[:, 0], starts[:, 1], duration):
        if state.drift is not None:
            stuck = ~np.isfinite(state.drift).all(axis=1)
        if state.step != saved[row]:
            continue
        times[row] = state.time
        lats[row], lons[row], beached[row] = state.positions[:, 0], state.positions[:, 1], stuck
        if forcing_log is not None:
            logged.append(state)
        row = min(row + 1, len(saved) - 1)

    if forcing_log is not None:
        forcing_log.update(_stack_forcing(logged, n))
    return Ensemble(times, lats, lons, np.full(n, 1.0 / n), beached)


def describe_run(forcing, particles, start, lat, lon, duration, timestep, datum_sigma_km,
                 sigma=DEFAULT_SIGMA, seed=None, save_every=None, entropy=None,
                 sigma_u=DEFAULT_SIGMA_U, memory_time_s=MEMORY_TIME_S) -> dict:
    """sar.pipeline.ensemble.describe_run, its pipeline block recording current_factor."""
    pipeline = DriftPipelineC2(forcing, timestep, sigma=sigma, sigma_u=sigma_u,
                               memory_time_s=memory_time_s).describe(start, lat, lon, duration)
    for key in ("seed", "particles", "start_lat", "start_lon"):
        pipeline.pop(key)
    return {"lat": round(float(lat), 2),
            "lon": round(float(to_display_longitude(lon)), 2),
            "start": str(np.datetime64(start, "m")),
            "particles": int(particles),
            "timestep": float(timestep),
            "duration": float(duration),
            "seed": seed if seed is None else int(seed),
            "datum_lat": float(lat), "datum_lon": float(lon),
            "datum_sigma_km": float(datum_sigma_km),
            "save_every_s": None if save_every is None else float(save_every),
            "seed_entropy": as_seed_sequence(seed).entropy if entropy is None else entropy,
            "weight": "uniform, 1/N",
            "pipeline": pipeline}


def _cli(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_run_arguments(parser)
    parser.add_argument("--particles", type=int, required=True, help="N, the ensemble size")
    parser.add_argument("--save-every", type=parse_span,
                        help="keep a state this often, such as 15m; default every step")
    parser.add_argument("--with-forcing", action="store_true",
                        help="also write current, wind and drift per row, about triple the size")
    parser.add_argument("--force", action="store_true",
                        help="write even over the size threshold, or over an existing file")
    parser.add_argument("--out", required=True, help="data root; the CSV goes in <out>/derived")
    args = parser.parse_args(argv)

    try:
        estimate_rows(args.particles, args.duration, args.timestep, args.save_every,
                      args.with_forcing, args.force)
        check_cloud(args.particles, args.datum_sigma_km)
        forcing = forcing_from(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    noise = random_term(args)
    try:
        # sigma_u is sized by HYCOM's own current (D033), not the doubled one.
        noise["sigma_u"], speed = resolve_sigma_u(noise["sigma_u"], forcing, args.lat, args.lon,
                                                  args.start)
    except ValueError as error:
        parser.error(str(error))
    spec = {"forcing": forcing, "particles": args.particles, "start": args.start,
            "lat": args.lat, "lon": args.lon, "duration": args.duration,
            "timestep": args.timestep, "datum_sigma_km": args.datum_sigma_km,
            **noise, "seed": args.seed, "save_every": args.save_every}
    log = {} if args.with_forcing else None
    seq = np.random.SeedSequence(args.seed)
    run = describe_run(**spec, entropy=seq.entropy)
    if speed is not None:
        run["sigma_u_by_current"] = {"start_current_ms": speed, "sigma_u": noise["sigma_u"],
                                     "rule": "D033, sar.model.position.SIGMA_U_RULE"}
    if (Path(args.out) / "derived" / f"{run_name(run)}.csv").exists() and not args.force:
        parser.error(f"{run_name(run)}.csv already exists in {args.out}/derived; pass --seed "
                     "to keep both runs, or --force to replace it")
    ensemble = run_ensemble(**{**spec, "seed": seq}, forcing_log=log)
    path = write_csv(ensemble, run, args.out, log, args.force)
    print(f"wrote {path}\nwrote {path.with_suffix('.json')}")
    return path


if __name__ == "__main__":
    _cli()
