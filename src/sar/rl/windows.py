"""windows.py: one Monte Carlo run cut into the search windows PPO trains on, one per transit time."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sar.pipeline.ensemble import Ensemble, parse_run_name, read_csv, run_name
from sar.pipeline.gridded import GriddedForcing
from sar.search.datum import marker_track
from sar.search.episode import STEPS
from sar.search.patterns import MarkerTrack
from sar.search.platform import STEP_S
from sar.search.scenario import SearchSetup, centroid, drift_bearing
from sar.utils.geo import to_store_longitude

WINDOW_S = STEPS * STEP_S
_NAME = re.compile(r"windows_(?P<run>.+)_arr(?P<arrivals>\d+(?:-\d+)*)m")


@dataclass(frozen=True)
class SearchWindows:
    """A run's saved cloud over every window it serves, and each window's marker."""

    name: str
    start: np.datetime64           # the call: when the cloud was released at the LKP
    lkp: tuple[float, float]       # (lat, lon), longitude 0 to 360
    seed: int
    t_s: np.ndarray                # (F,) seconds since the call of each saved frame kept
    lat: np.ndarray                # (F, N)
    lon: np.ndarray                # (F, N), 0 to 360
    beached: np.ndarray            # (F, N)
    weight: np.ndarray             # (N,)
    arrivals_s: np.ndarray         # (A,) seconds from the call to arriving on scene
    marker_lat: np.ndarray         # (A, STEPS + 1), every STEP_S from the drop
    marker_lon: np.ndarray
    drift_bearing_deg: np.ndarray  # (A,), NaN where the datum has no drift

    def setup(self, k: int) -> SearchSetup:
        """Window k as the referee takes it: the cloud every minute from arrival, and its marker."""
        arrival = float(self.arrivals_s[k])
        window = cloud_window(self, arrival)
        marker = MarkerTrack(np.arange(STEPS + 1) * STEP_S, self.marker_lat[k], self.marker_lon[k])
        bearing = float(self.drift_bearing_deg[k])
        datum = centroid(window.lat[0], window.lon[0], window.weight)
        return SearchSetup(window, marker, None, datum, arrival,
                           None if np.isnan(bearing) else bearing)


def frames_at(t_s, lat, lon, beached, at) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cloud at times `at`, straight between the saves either side: 0.8 m rms off, under D030."""
    at = np.asarray(at, dtype=float)
    if at.min() < t_s[0] - 1e-6 or at.max() > t_s[-1] + 1e-6:
        raise ValueError(f"times {at.min():g} to {at.max():g} s lie outside the saves, "
                         f"{t_s[0]:g} to {t_s[-1]:g} s")
    k = np.clip(np.searchsorted(t_s, at, side="right") - 1, 0, len(t_s) - 2)
    f = ((at - t_s[k]) / (t_s[k + 1] - t_s[k]))[:, None]
    dlon = (lon[k + 1] - lon[k] + 180.0) % 360.0 - 180.0
    return (lat[k] + f * (lat[k + 1] - lat[k]), (lon[k] + f * dlon) % 360.0,
            np.where(f >= 1.0, beached[k + 1], beached[k]))


def cloud_window(source, arrival_s: float) -> Ensemble:
    """The STEPS + 1 frames, STEP_S apart, from arrival: what `SearchEpisode` needs."""
    at = arrival_s + np.arange(STEPS + 1) * STEP_S
    lat, lon, beached = frames_at(source.t_s, source.lat, source.lon, source.beached, at)
    times = source.start + (at * 1e6).astype("timedelta64[us]")
    return Ensemble(times, lat, lon, np.asarray(source.weight, dtype=float), beached)


def windows_name(run: dict, arrivals_s) -> str:
    """windows_<the run's name without 'ensemble_'>_arr<transit minutes, joined by ->m."""
    minutes = "-".join(str(round(float(a) / 60.0)) for a in arrivals_s)
    return f"windows_{run_name(run).removeprefix('ensemble_')}_arr{minutes}m"


def parse_windows_name(name) -> dict:
    """The run parameters and transit times a windows file name encodes; the inverse of windows_name."""
    match = _NAME.fullmatch(Path(str(name)).name.removesuffix(".npz"))
    if not match:
        raise ValueError(f"{name!r} is not a windows file name")
    params = parse_run_name(f"ensemble_{match['run']}")
    return {**params, "arrivals_min": [int(m) for m in match["arrivals"].split("-")]}


def is_test(name) -> bool:
    """Whether a windows file is held out for testing: its call falls in the last 7 days of its month."""
    start = np.datetime64(parse_windows_name(name)["start"], "D")
    month_end = (start.astype("datetime64[M]") + np.timedelta64(1, "M")).astype("datetime64[D]")
    return int((month_end - start).astype(int)) <= 7


def build(ensemble: Ensemble, run: dict, forcing, arrivals_s) -> SearchWindows:
    """Cut a run into one window per arrival, each with its datum marker drifted by `forcing`."""
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


def save(windows: SearchWindows, out) -> Path:
    """Write <out>/<name>.npz, positions in float64 so no particle moves across the strip's edge."""
    path = Path(out) / f"{windows.name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, name=windows.name, start=str(windows.start), lkp=np.array(windows.lkp),
             seed=windows.seed, t_s=windows.t_s, lat=windows.lat, lon=windows.lon,
             beached=windows.beached, weight=windows.weight, arrivals_s=windows.arrivals_s,
             marker_lat=windows.marker_lat, marker_lon=windows.marker_lon,
             drift_bearing_deg=windows.drift_bearing_deg)
    return path


def load(path) -> SearchWindows:
    """A windows file back into SearchWindows."""
    with np.load(path) as f:
        return SearchWindows(str(f["name"]), np.datetime64(str(f["start"]), "us"),
                             (float(f["lkp"][0]), float(f["lkp"][1])), int(f["seed"]), f["t_s"],
                             f["lat"], f["lon"], f["beached"], f["weight"], f["arrivals_s"],
                             f["marker_lat"], f["marker_lon"], f["drift_bearing_deg"])


def _cli(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="an ensemble CSV; its JSON sidecar beside it")
    parser.add_argument("--forcing-dir", required=True,
                        help="data root holding raw/hycom_* and raw/era5_*, for the markers")
    parser.add_argument("--arrival-min", type=int, nargs="+", required=True,
                        help="minutes from the call to arriving on scene, such as 30 45 60")
    parser.add_argument("--out", required=True, help="directory for the windows file")
    parser.add_argument("--force", action="store_true", help="replace an existing windows file")
    args = parser.parse_args(argv)

    run = json.loads(Path(args.run).with_suffix(".json").read_text())
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
