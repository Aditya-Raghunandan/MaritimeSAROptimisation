"""ensemble.py: N particles from a perturbed datum through one DriftPipeline (R1e, R1f)."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sar.model.position import DEFAULT_SIGMA, INTEGRATION_STEP_SECONDS, calculate_position
from sar.pipeline.forcing import ConstantForcing
from sar.pipeline.track import _STEP_TOLERANCE, DriftPipeline
from sar.utils.geo import to_display_longitude

# Measured on 48 h runs saved every 15 min: 161.8 MB per 1.93 M rows, 23.5 MB per 193 k forced.
BYTES_PER_ROW = 84
BYTES_PER_ROW_WITH_FORCING = 122

# A project choice: passes 10^4 particles saved every 15 min (162 MB), refuses every step (2.4 GB).
MAX_BYTES = 1_000_000_000

NAME_KEYS = ("lat", "lon", "start", "particles", "timestep", "duration", "seed")

_UNITS = (("h", 3600), ("m", 60), ("s", 1))
_NAME = re.compile(r"ensemble_(\d{4})([NS])(\d{5})([EW])_(\d{8}T\d{4})_N(\d+)"
                   r"_dt(\d+)s_T(\d+[hms])_seed(\d+|none)")


@dataclass(frozen=True)
class Ensemble:
    """#46's container: the cloud at T saved instants, N particles each."""

    times: np.ndarray       # (T,) datetime64[us]
    lat: np.ndarray         # (T, N) degrees north
    lon: np.ndarray         # (T, N) degrees east, 0 to 360
    weight: np.ndarray      # (N,) sums to 1
    beached: np.ndarray     # (T, N) bool, frozen because the forcing there was NaN (D016)

    def __post_init__(self) -> None:
        if np.ndim(self.times) != 1 or np.ndim(self.weight) != 1:
            raise ValueError("times and weight must be 1-D, over saved times and particles")
        if not np.isclose(np.sum(self.weight), 1.0):
            raise ValueError(f"weight must sum to 1, got {np.sum(self.weight)}")
        t, n = len(self.times), len(self.weight)
        for name in ("lat", "lon", "beached"):
            if getattr(self, name).shape != (t, n):
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, "
                                 f"expected ({t}, {n}) from times and weight")


def parse_span(text) -> float:
    """'48h', '15m', '90s' or a bare number of seconds, as seconds."""
    match = re.fullmatch(r"\s*([0-9.]+)\s*([hms]?)\s*", str(text))
    if not match:
        raise ValueError(f"{text!r} is not a span such as 48h, 15m, 90s or 3600")
    return float(match[1]) * dict(_UNITS).get(match[2] or "s")


def _whole_seconds(seconds) -> int:
    """A span as a positive whole number of seconds, which is all a run name can carry."""
    if float(seconds) <= 0 or float(seconds) != int(seconds):
        raise ValueError(f"a run name needs a positive whole number of seconds, got {seconds}")
    return int(seconds)


def format_span(seconds: float) -> str:
    """Seconds in the largest unit that holds them exactly, the inverse of parse_span."""
    seconds = _whole_seconds(seconds)
    unit, size = next((u, s) for u, s in _UNITS if seconds % s == 0)
    return f"{int(seconds) // size}{unit}"


def as_seed_sequence(seed) -> np.random.SeedSequence:
    """An int, None or a SeedSequence, as a fresh copy, since spawn() mutates the one it is given."""
    if isinstance(seed, np.random.SeedSequence):
        return np.random.SeedSequence(seed.entropy, spawn_key=seed.spawn_key,
                                      pool_size=seed.pool_size)
    return np.random.SeedSequence(seed)


def check_cloud(n, datum_sigma_km) -> None:
    """Refuse a particle count that is not a positive integer, or a negative datum spread."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"the particle count must be a positive integer, got {n!r}")
    if not np.isfinite(datum_sigma_km) or datum_sigma_km < 0.0:
        raise ValueError(f"datum_sigma_km must be a non-negative number, got {datum_sigma_km}")


def spawn_positions(lat, lon, n, datum_sigma_km, rng) -> np.ndarray:
    """R1e's start cloud: n positions, Gaussian with datum_sigma_km per axis about the datum."""
    check_cloud(n, datum_sigma_km)
    offsets_m = datum_sigma_km * 1000.0 * rng.standard_normal((int(n), 2))
    # https://www.investopedia.com/terms/c/central_limit_theorem.asp#toc-what-is-the-central-limit-theorem-clt
    # One second at offsets_m m/s is offsets_m metres, through the engine's own conversion.
    return calculate_position(np.tile([lat, lon], (int(n), 1)), offsets_m, 1.0)


def run_ensemble(forcing, particles, start, lat, lon, duration, timestep, datum_sigma_km,
                 sigma=DEFAULT_SIGMA, seed=None, save_every=None, forcing_log=None) -> Ensemble:
    """One DriftPipeline over the spawned cloud, keeping every save_every seconds and the end."""
    datum_seq, kick_seq = as_seed_sequence(seed).spawn(2)
    starts = spawn_positions(lat, lon, particles, datum_sigma_km,
                             np.random.default_rng(datum_seq))
    pipeline = DriftPipeline(forcing, timestep, sigma=sigma, seed=kick_seq)
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


def _stack_forcing(states, n) -> dict:
    """Current, wind and drift as (T, N, 2) arrays, NaN on the final state, which has none."""
    empty = np.full((n, 2), np.nan)
    return {name: np.array([getattr(s, name) if getattr(s, name) is not None else empty
                            for s in states])
            for name in ("current", "wind", "drift")}


def save_stride(save_every, timestep) -> int:
    """How many steps apart the saved states are; None saves every step."""
    if save_every is None:
        return 1
    steps = float(save_every) / float(timestep)
    if save_every <= 0 or abs(steps - round(steps)) > _STEP_TOLERANCE * max(1.0, steps):
        raise ValueError(f"save_every of {save_every:g} s is not a positive whole number of "
                         f"{timestep:g} s steps")
    return round(steps)


def synthetic_cloud(centre, spread_km, n, rng, time=None) -> Ensemble:
    """#46's stand-in: one instant, n particles Gaussian with spread_km per axis about centre."""
    positions = spawn_positions(centre[0], centre[1], n, spread_km, rng)
    return Ensemble(np.array([np.datetime64(time, "us")]), positions[None, :, 0],
                    positions[None, :, 1], np.full(n, 1.0 / n), np.zeros((1, n), dtype=bool))


def describe_run(forcing, particles, start, lat, lon, duration, timestep, datum_sigma_km,
                 sigma=DEFAULT_SIGMA, seed=None, save_every=None, entropy=None) -> dict:
    """The sidecar: DriftPipeline.describe() at the datum, plus the datum, N and the seed."""
    pipeline = DriftPipeline(forcing, timestep, sigma=sigma).describe(start, lat, lon, duration)
    # The datum, N and the seed are recorded above; one-particle copies of them would mislead.
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


def run_name(params) -> str:
    """ensemble_<datum>_<start>_N<particles>_dt<timestep>_T<duration>_seed<seed>."""
    lat, lon = float(params["lat"]), float(to_display_longitude(params["lon"]))
    start = np.datetime64(params["start"], "m").astype(object).strftime("%Y%m%dT%H%M")
    seed = "none" if params["seed"] is None else str(int(params["seed"]))
    return (f"ensemble_{abs(lat) * 100:04.0f}{'N' if lat >= 0 else 'S'}"
            f"{abs(lon) * 100:05.0f}{'E' if lon >= 0 else 'W'}_{start}"
            f"_N{int(params['particles'])}_dt{_whole_seconds(params['timestep'])}s"
            f"_T{format_span(params['duration'])}_seed{seed}")


def parse_run_name(name) -> dict:
    """The run parameters a file name encodes; the inverse of run_name."""
    match = _NAME.fullmatch(Path(name).stem)
    if not match:
        raise ValueError(f"{name!r} does not follow the ensemble naming convention")
    lat, ns, lon, ew, start, n, dt, dur, seed = match.groups()
    stamp = f"{start[:4]}-{start[4:6]}-{start[6:8]}T{start[9:11]}:{start[11:]}"
    return {"lat": int(lat) / 100 * (1 if ns == "N" else -1),
            "lon": int(lon) / 100 * (1 if ew == "E" else -1),
            "start": stamp,
            "particles": int(n),
            "timestep": float(dt),
            "duration": parse_span(dur),
            "seed": None if seed == "none" else int(seed)}


def check_size(rows, with_forcing=False, force=False) -> int:
    """Refuse a file over MAX_BYTES unless forced; returns the estimated bytes."""
    size = rows * (BYTES_PER_ROW_WITH_FORCING if with_forcing else BYTES_PER_ROW)
    if size > MAX_BYTES and not force:
        raise ValueError(f"this run writes {rows:,} rows, an estimated {size / 1e9:.1f} GB, "
                         f"over the {MAX_BYTES / 1e9:.1f} GB threshold; save less often "
                         "with --save-every, or pass --force")
    return size


def estimate_rows(particles, duration, timestep, save_every, with_forcing=False,
                  force=False) -> int:
    """Rows a run would write, refusing one over MAX_BYTES unless forced."""
    steps = DriftPipeline(ConstantForcing(), timestep).step_count(duration)
    stride = save_stride(save_every, timestep)
    rows = int(particles) * (steps // stride + 1 + (1 if steps % stride else 0))
    check_size(rows, with_forcing, force)
    return rows


def write_csv(ensemble, run, out, forcing=None, force=False) -> Path:
    """The long-format CSV under <out>/derived, with the run's JSON sidecar beside it."""
    t, n = ensemble.lat.shape
    check_size(t * n, bool(forcing), force)
    if np.isnat(ensemble.times).any():
        raise ValueError("every saved time must be a real instant to be written; "
                         "give synthetic_cloud a time")
    path = Path(out) / "derived" / f"{run_name(run)}.csv"
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; an unseeded run with the same parameters "
                              "has the same name, so pass --seed to keep both, or --force")

    start = ensemble.times[0]
    frame = pd.DataFrame({
        "step": np.repeat(_steps(ensemble.times, run), n),
        "seconds": np.repeat((ensemble.times - start) / np.timedelta64(1, "s"), n),
        "time": np.repeat(ensemble.times.astype(str), n),
        "particle": np.tile(np.arange(n), t),
        "lat": ensemble.lat.ravel(),
        "lon": ensemble.lon.ravel(),
        "beached": ensemble.beached.ravel().astype(int),
    })
    if forcing:
        for name in ("current", "wind", "drift"):
            frame[f"{name}_u"] = forcing[name][..., 0].ravel()
            frame[f"{name}_v"] = forcing[name][..., 1].ravel()

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(run, indent=2, default=str))
    return path


def _steps(times, run) -> np.ndarray:
    """The integration step each saved instant is, from the run's timestep."""
    elapsed = (times - times[0]) / np.timedelta64(1, "s")
    return np.rint(elapsed / float(run["timestep"])).astype(int)


def read_csv(path) -> Ensemble:
    """A written CSV back into an Ensemble, positions exact to the last bit."""
    frame = pd.read_csv(path, float_precision="round_trip").sort_values(["step", "particle"])
    t, n = frame["step"].nunique(), frame["particle"].nunique()
    if t * n != len(frame):
        raise ValueError(f"{path} has {len(frame)} rows, not {t} saved times x {n} particles")
    times = frame["time"].to_numpy()[::n].astype("datetime64[us]")
    return Ensemble(times, frame["lat"].to_numpy().reshape(t, n),
                    frame["lon"].to_numpy().reshape(t, n), np.full(n, 1.0 / n),
                    frame["beached"].to_numpy().reshape(t, n).astype(bool))


def add_run_arguments(parser) -> None:
    """The flags that say what a run is, shared with scripts/converge_ensemble.py."""
    parser.add_argument("--lat", type=float, required=True, help="datum latitude, degrees north")
    parser.add_argument("--lon", type=float, required=True, help="datum longitude, degrees east")
    parser.add_argument("--datum-sigma-km", type=float, required=True,
                        help="R1e's datum spread per axis in km, required")
    parser.add_argument("--start", required=True, help="start time, ISO 8601")
    parser.add_argument("--timestep", type=float, required=True,
                        help=f"step in seconds, required; D009 fixes {INTEGRATION_STEP_SECONDS:g}")
    parser.add_argument("--duration", type=parse_span, required=True,
                        help="how long to track, such as 48h, 90m or 3600")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help=f"D009's sigma in m/s^0.5, default {DEFAULT_SIGMA:g} (unmeasured)")
    parser.add_argument("--seed", type=int, help="base seed; recorded either way")
    parser.add_argument("--constant-current", nargs=2, type=float, required=True,
                        metavar=("U", "V"), help="uniform steady current in m/s")
    parser.add_argument("--constant-wind", nargs=2, type=float, metavar=("U", "V"),
                        help="uniform steady 10 m wind in m/s, default calm")


def forcing_from(args) -> ConstantForcing:
    """The one backend there is, from the parsed flags."""
    return ConstantForcing(args.constant_current, args.constant_wind or (0.0, 0.0))


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
    except ValueError as error:
        parser.error(str(error))
    forcing = forcing_from(args)
    spec = {"forcing": forcing, "particles": args.particles, "start": args.start,
            "lat": args.lat, "lon": args.lon, "duration": args.duration,
            "timestep": args.timestep, "datum_sigma_km": args.datum_sigma_km,
            "sigma": args.sigma, "seed": args.seed, "save_every": args.save_every}
    log = {} if args.with_forcing else None
    # Drawn once, so a run without --seed still records the entropy it actually used.
    seq = np.random.SeedSequence(args.seed)
    run = describe_run(**spec, entropy=seq.entropy)
    if (Path(args.out) / "derived" / f"{run_name(run)}.csv").exists() and not args.force:
        parser.error(f"{run_name(run)}.csv already exists in {args.out}/derived; pass --seed "
                     "to keep both runs, or --force to replace it")
    ensemble = run_ensemble(**{**spec, "seed": seq}, forcing_log=log)
    path = write_csv(ensemble, run, args.out, log, args.force)
    print(f"wrote {path}\nwrote {path.with_suffix('.json')}")
    return path


if __name__ == "__main__":
    _cli()
