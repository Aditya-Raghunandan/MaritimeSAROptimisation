"""shaped_windows.py: practice windows again, the same runs with the cloud-shape correction (L41).

The agents train on practice clouds (`scripts/training_montes.sh` then `sar.rl.windows`: five
places, a start every week for five years, 10,000 particles, rvc). This remakes each one from
the windows file itself (its last known position, start, seed, length and arrivals) with the
current and wind scaled and the random velocity stretched along the drift, as the noise model
rvs does in the benchmark. Same seeds, so each new cloud is the old one's twin, and the same
names, so the training split and the validation weeks do not move.

    python scripts/shaped_windows.py --src DATA/derived/rl/windows_arr30-60-120-180 \
        --out DATA/derived/rl/windows_rvs_arr30-60-120-180 --forcing-dir DATA \
        --shape-json shape.json [--part K --parts P]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from sar.model.position import BY_CURRENT
from sar.pipeline.ensemble import resolve_sigma_u, run_ensemble
from sar.rl.windows import build, load, parse_windows_name, save, shaped
from sar.search.platform import STEP_S


def remake(path: Path, forcing_dir, out: Path, shape: dict) -> Path | None:
    from sar.pipeline.gridded import GriddedForcing

    target = out / path.name
    if target.exists():
        return None
    old = load(path)
    p = parse_windows_name(path.name)
    lat, lon = old.lkp
    start = old.start
    duration = float(p["duration"])
    arrivals = [float(a) for a in old.arrivals_s]
    end = start + np.timedelta64(int(duration), "s")
    with GriddedForcing.from_dir(forcing_dir, start, end) as raw:
        sigma_u, _ = resolve_sigma_u(BY_CURRENT, raw, lat, lon, start)
        forcing, factors = shaped(raw, shape)
        ensemble = run_ensemble(forcing, int(p["particles"]), start, lat, lon, duration, STEP_S,
                                datum_sigma_km=0.0, sigma=0.0, seed=old.seed, save_every=300.0,
                                sigma_u=sigma_u, shape=factors)
        run = {"lat": float(p["lat"]), "lon": float(p["lon"]), "start": str(p["start"]),
               "particles": int(p["particles"]), "timestep": STEP_S, "duration": duration,
               "seed": old.seed, "datum_lat": lat, "datum_lon": lon}
        new = build(ensemble, run, forcing, arrivals)
    if new.name != old.name:
        raise ValueError(f"{path.name}: remade as {new.name}, not the same window")
    return save(new, out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--forcing-dir", required=True)
    parser.add_argument("--shape-json", required=True)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    args = parser.parse_args(argv)
    shape = json.loads(Path(args.shape_json).read_text())
    print(f"part {args.part}: shape {shape}", flush=True)      # what was read, not what was meant
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.src).glob("windows_*.npz"))[args.part::args.parts]
    done = 0
    for path in files:
        try:
            if remake(path, args.forcing_dir, out, shape):
                done += 1
        except (ValueError, FileNotFoundError) as error:
            print(f"{path.name}: skipped, {error}", flush=True)
    print(f"part {args.part}: {done} of {len(files)} remade into {out}", flush=True)
    os._exit(0)      # as cloud_shape.py: written, so never left waiting on the readers' threads


if __name__ == "__main__":
    main()
