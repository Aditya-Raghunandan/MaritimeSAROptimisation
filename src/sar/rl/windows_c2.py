"""windows_c2.py: sar.rl.windows for c2 runs, each marker and drift bearing under drift_c2's doubled current."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sar.pipeline.ensemble import read_csv
from sar.pipeline.gridded import GriddedForcing
from sar.rl.windows import WINDOW_S, SearchWindows, cloud_window, save, windows_name
from sar.search.datum_c2 import drift_bearing, marker_track
from sar.search.episode import STEPS
from sar.search.platform import STEP_S
from sar.search.scenario import centroid
from sar.utils.geo import to_store_longitude


def build(ensemble, run: dict, forcing, arrivals_s) -> SearchWindows:
    """sar.rl.windows.build with the c2 marker and drift bearing."""
    arrivals = np.asarray(sorted(float(a) for a in arrivals_s))
    t_s = (ensemble.times - ensemble.times[0]) / np.timedelta64(1, "s")
    if arrivals.size == 0 or np.any(np.abs(arrivals / STEP_S - np.round(arrivals / STEP_S)) > 1e-9):
        raise ValueError(f"arrivals must be whole minutes after the call, got {list(arrivals)}")
    if arrivals[0] < t_s[0] or arrivals[-1] + WINDOW_S > t_s[-1] + 1e-6:
        raise ValueError(f"the run covers {t_s[-1]:g} s; a window from {arrivals[-1]:g} s "
                         f"needs {arrivals[-1] + WINDOW_S:g} s")
    first = int(np.searchsorted(t_s, arrivals[0], side="right") - 1)
    last = int(np.searchsorted(t_s, arrivals[-1] + WINDOW_S - 1e-6, side="left"))
    keep = slice(first, last + 1)
    start = ensemble.times[0]
    lkp = (float(run["datum_lat"]), float(to_store_longitude(run["datum_lon"])))
    partial = SearchWindows(windows_name(run, arrivals), start, lkp, int(run["seed"]),
                            t_s[keep], ensemble.lat[keep], ensemble.lon[keep],
                            ensemble.beached[keep], ensemble.weight, arrivals,
                            np.empty((0, STEPS + 1)), np.empty((0, STEPS + 1)), np.empty(0))
    m_lat, m_lon, bearings = [], [], []
    for a in arrivals:
        window = cloud_window(partial, a)
        datum = centroid(window.lat[0], window.lon[0], window.weight)
        marker = marker_track(*datum, window.times[0], WINDOW_S, forcing)
        if marker.t_s.shape != (STEPS + 1,):
            raise ValueError(f"the marker track has {marker.t_s.size} points, not {STEPS + 1}")
        m_lat.append(marker.lat)
        m_lon.append(marker.lon)
        b = drift_bearing(forcing, *datum, window.times[0])
        bearings.append(np.nan if b is None else b)
    return SearchWindows(partial.name, start, lkp, partial.seed, partial.t_s, partial.lat,
                         partial.lon, partial.beached, partial.weight, arrivals,
                         np.array(m_lat), np.array(m_lon), np.array(bearings))


def _cli(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="a c2 ensemble CSV; its JSON sidecar beside it")
    parser.add_argument("--forcing-dir", required=True,
                        help="data root holding raw/hycom_* and raw/era5_*, for the markers")
    parser.add_argument("--arrival-min", type=int, nargs="+", required=True,
                        help="minutes from the call to arriving on scene, such as 30 60 120 180")
    parser.add_argument("--out", required=True, help="directory for the windows file")
    parser.add_argument("--force", action="store_true", help="replace an existing windows file")
    args = parser.parse_args(argv)

    run = json.loads(Path(args.run).with_suffix(".json").read_text())
    if run["pipeline"].get("current_factor") is None:
        parser.error(f"{args.run} has no current_factor: it is not a c2 run (sar.pipeline.ensemble_c2)")
    arrivals = [60.0 * m for m in args.arrival_min]
    path = Path(args.out) / f"{windows_name(run, sorted(arrivals))}.npz"
    if path.exists() and not args.force:
        parser.error(f"{path} exists; pass --force to replace it")
    start = np.datetime64(run["start"], "us")
    begin = start + np.timedelta64(int(min(arrivals)), "s")
    end = start + np.timedelta64(int(max(arrivals) + WINDOW_S), "s")
    try:
        with GriddedForcing.from_dir(args.forcing_dir, begin, end) as forcing:
            windows = build(read_csv(args.run), run, forcing, arrivals)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    path = save(windows, args.out)
    print(path)
    return path


if __name__ == "__main__":
    _cli()
