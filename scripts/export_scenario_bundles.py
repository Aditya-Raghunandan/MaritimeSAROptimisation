"""export_scenario_bundles.py: the scenario table as files the site replays and re-scores.

The site is static (D021), so it cannot run the engine. Instead, for every row x noise model
x arrival (a "window"), this writes what the browser needs to show the search and score it
again with its own copy of the referee (frontend/src/referee.js), held to this one:

  <out>/index.json                 every scenario: start, water type, straightness, windows
  <out>/<S01>/rv_2h.json           one window: the marker, the datum, the datum line, the real
                                   buoy, and every searcher's flight (the Waypoints the referee
                                   was given, step by step) with the paper's numbers for it
  <out>/<S01>/rv_2h.f32            the cloud: 46 frames x N particles x (east, north), float32
                                   little-endian, metres from the marker at that frame
  <out>/<S01>/field_2h.f32         current and wind on a 21 x 21 grid, 2 km apart about the
                                   datum, every 5 minutes of the window: what a player sees in
                                   "field only" mode. (u, v current, u, v wind), float32

WHY METRES FROM THE MARKER. float32 degrees are good to about 0.4 m, which would move
particles across the strip's edge. Metres from the marker are good to millimetres, and
`sar.utils.geo.offset_position` undoes `relative_m` exactly, so the browser rebuilds every
particle to well under a millimetre (checked here for every window, against the float64
cloud) and its score agrees with the paper's to about 1e-6.

The flights are the benchmark's (`sar.search.benchmark`), recorded as they are flown, and the
numbers beside them are the referee's, computed here. Nothing about a window is stored that
cannot be rebuilt from its row and seed.

    python scripts/export_scenario_bundles.py export --csv scenarios.csv --forcing-dir DATA \\
        --rows 1-55 --out DATA/published/scenarios/v1
    python scripts/export_scenario_bundles.py upload --src DATA/published/scenarios/v1 \\
        --repo AdityaRugs/MaritimeSARoperations --path-in-repo scenarios/v1
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sar.pipeline.gridded import GriddedForcing
from sar.search.benchmark import GREEDY, NOISE, SEARCHERS, recorded, searcher
from sar.search.episode import STEPS, SearchEpisode, run
from sar.search.platform import STEP_S, SWEEP_WIDTH_M
from sar.search.scenario import datum_line, read_scenarios, scenario_search, straightness
from sar.search.sweep import relative_m
from sar.utils.geo import offset_position, to_display_longitude

FORMAT = 1
FIELD_HALF_KM, FIELD_STEP_KM, FIELD_EVERY_MIN = 20.0, 2.0, 5
MAX_REBUILD_ERROR_M = 0.01


def window_name(noise: str, hours: float) -> str:
    return f"{noise}_{hours:g}h"


def _floats(a, digits: int = 9) -> list:
    return [None if not np.isfinite(v) else round(float(v), digits) for v in np.ravel(a)]


def encode_cloud(window, marker) -> tuple[np.ndarray, float]:
    """Particles as float32 metres from the marker at each frame, and the worst rebuild error."""
    frames, n = window.lat.shape
    mlat, mlon = marker.at(np.arange(frames) * STEP_S)
    out = np.empty((frames, n, 2), dtype="<f4")
    worst = 0.0
    for k in range(frames):
        east, north = relative_m(window.lat[k], window.lon[k], mlat[k], mlon[k])
        out[k, :, 0], out[k, :, 1] = east, north
        lat, lon = offset_position(mlat[k], mlon[k], out[k, :, 0].astype(float),
                                   out[k, :, 1].astype(float))
        back_e, back_n = relative_m(lat, lon, window.lat[k], window.lon[k])
        worst = max(worst, float(np.nanmax(np.hypot(back_e, back_n))))
    return out, worst


def field_grid(forcing, datum, start) -> np.ndarray:
    """(u, v) current and wind about the datum, every FIELD_EVERY_MIN through the window."""
    k = int(round(FIELD_HALF_KM / FIELD_STEP_KM))
    offsets = np.arange(-k, k + 1) * FIELD_STEP_KM * 1000.0
    east, north = np.meshgrid(offsets, offsets)               # rows run south to north
    lat, lon = offset_position(datum[0], datum[1], east.ravel(), north.ravel())
    times = start + (np.arange(0, STEPS + 1, FIELD_EVERY_MIN) * STEP_S * 1e6).astype(
        "timedelta64[us]")
    out = np.empty((times.size, 2 * k + 1, 2 * k + 1, 4), dtype="<f4")
    for i, t in enumerate(times):
        current, wind = forcing.sample(lat, lon, t)
        out[i] = np.concatenate([current, wind], axis=1).reshape(2 * k + 1, 2 * k + 1, 4)
    return out


def flights(setup, row, hours: float, greedy: dict) -> dict:
    """Every searcher flown and recorded: its steps, and the referee's numbers for it."""
    out = {}
    for name in SEARCHERS:
        policy, first, layout = searcher(name, setup, row, hours, greedy)
        flying, steps = recorded(policy)
        m = run(flying, SearchEpisode(setup.window, setup.marker, target=setup.target))
        target = m.get("target") or {}
        out[name] = {
            "layout": layout, "first_bearing_deg": first,
            "steps": [{"t_s": _floats(s.t_s), "east_m": _floats(s.east_m),
                       "north_m": _floats(s.north_m)} for s in steps],
            "python": {"pos": m["pos"], "drain_rate": m["removed_per_step"],
                       "expected_ttd_s": m["expected_ttd_s"],
                       "found": target.get("found"), "found_s": target.get("found_s"),
                       "closest_m": target.get("closest_m"),
                       "closest_s": target.get("closest_s")}}
    return out


def export_row(row, forcing, noises, arrivals_h, particles: int, out: Path,
               greedy: dict) -> dict:
    """Every window of one row; returns its entry for index.json.

    `forcing` is a backend, or a data root, opened once for the latest arrival's window.
    """
    if isinstance(forcing, (str, Path)):
        start = np.datetime64(str(row["start"]), "us")
        longest = max(arrivals_h) * 3600.0 + STEPS * STEP_S
        forcing = GriddedForcing.from_dir(forcing, start,
                                          start + np.timedelta64(int(round(longest)), "s"))
    folder = out / row["scenario"]
    folder.mkdir(parents=True, exist_ok=True)
    windows = []
    for hours in arrivals_h:
        for noise in noises:
            setup = scenario_search(row, forcing, round(hours * 3600.0), particles,
                                    **NOISE[noise])
            weight = np.asarray(setup.window.weight, dtype=float)
            if not np.allclose(weight, weight[0]):
                raise ValueError(f"{row['scenario']}: particle weights are not uniform")
            cloud, worst = encode_cloud(setup.window, setup.marker)
            if worst > MAX_REBUILD_ERROR_M:
                raise ValueError(f"{row['scenario']} {noise} {hours:g} h: a particle rebuilds "
                                 f"{worst:.4f} m off")
            name = window_name(noise, hours)
            cloud.tofile(folder / f"{name}.f32")
            lkp = (float(row["lat"]), float(row["lon"]))
            line_bearing, line_m = datum_line(lkp, setup.datum)
            target = setup.target
            record = {
                "format": FORMAT, "scenario": row["scenario"], "noise": noise,
                "noise_settings": NOISE[noise], "arrival_h": hours,
                "arrival_s": setup.arrival_s,
                "start_utc": str(row["start"]), "window_start_utc":
                    str(np.datetime_as_string(setup.window.times[0], unit="s")),
                "frames": int(cloud.shape[0]), "particles": int(cloud.shape[1]),
                "step_s": STEP_S, "sweep_width_m": SWEEP_WIDTH_M, "pod": 1.0,
                "weight": float(weight[0]), "cloud": f"{name}.f32",
                "cloud_layout": "float32 LE [frame][particle][east_m, north_m] from the "
                                "marker at that frame; longitude in the store convention",
                "rebuild_error_m": worst,
                "marker": {"t_s": _floats(setup.marker.t_s), "lat": _floats(setup.marker.lat, 10),
                           "lon": _floats(setup.marker.lon, 10)},
                "lkp": {"lat": lkp[0], "lon": lkp[1]},
                "datum": {"lat": setup.datum[0],
                          "lon": float(to_display_longitude(setup.datum[1]))},
                "datum_line": {"bearing_deg": line_bearing, "length_m": line_m},
                "drift_bearing_deg": setup.drift_bearing_deg,
                "target": None if target is None else {
                    "t_s": _floats(target.t_s), "lat": _floats(target.lat, 10),
                    "lon": _floats(target.lon, 10)},
                "field": f"field_{hours:g}h.f32",
                "flights": flights(setup, row, hours, greedy),
            }
            (folder / f"{name}.json").write_text(json.dumps(record, separators=(",", ":")))
            windows.append(name)
        grid = field_grid(forcing, setup.datum, setup.window.times[0])
        grid.tofile(folder / f"field_{hours:g}h.f32")
    return {"scenario": row["scenario"], "set": row.get("set"), "water": row.get("stratum"),
            "group": None if pd.isna(row.get("group")) else int(row.get("group")),
            "start_utc": str(row["start"]), "lat": float(row["lat"]),
            "lon": float(to_display_longitude(row["lon"])),
            "speed_4h_ms": float(row["speed_4h_ms"]) if "speed_4h_ms" in row else None,
            "straightness_4h": straightness(row), "windows": windows,
            "buoy": {"t_h": [0] + [h for h in range(1, 7) if f"lat_{h}h" in row],
                     "lat": [float(row["lat"])] + [float(row[f"lat_{h}h"]) for h in range(1, 7)
                                                   if f"lat_{h}h" in row],
                     "lon": [float(to_display_longitude(row["lon"]))]
                     + [float(to_display_longitude(row[f"lon_{h}h"])) for h in range(1, 7)
                        if f"lon_{h}h" in row]}}


def write_index(out: Path, entries: list[dict] | None, settings: dict | None) -> dict:
    """index.json from every process's rows, each kept in its own .index.<host>.<pid>.json.

    Several processes write at once, so each merges what it finds when it ends, and the
    `index` command merges once more after the array has finished.
    """
    if entries is not None:
        part = out / f".index.{socket.gethostname()}.{os.getpid()}.json"
        part.write_text(json.dumps({"settings": settings, "rows": entries}))
    rows = {}
    for p in sorted(out.glob(".index.*.json")):
        held = json.loads(p.read_text())
        settings = settings or held["settings"]
        rows.update({e["scenario"]: e for e in held["rows"]})
    index = {"format": FORMAT, **(settings or {}),
             "field": {"half_km": FIELD_HALF_KM, "step_km": FIELD_STEP_KM,
                       "every_min": FIELD_EVERY_MIN,
                       "layout": "float32 LE [time][row south->north][col west->east]"
                                 "[current u, current v, wind u, wind v]"},
             "scenarios": sorted(rows.values(), key=lambda e: e["scenario"])}
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return index


def upload(src: Path, repo: str, path_in_repo: str) -> None:
    """Upload the bundles. The credential comes from the environment, never an argument."""
    from huggingface_hub import HfApi, get_token
    token = get_token()
    if not token:
        raise SystemExit("no Hugging Face credential found: run `huggingface-cli login` once")
    api = HfApi(token=token)
    print(f"authenticated as {api.whoami()['name']}; uploading {src} -> {repo}/{path_in_repo}")
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(src),
                      path_in_repo=path_in_repo, ignore_patterns=[".index.*"])
    print(f"done: https://huggingface.co/datasets/{repo}/tree/main/{path_in_repo}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("--csv", required=True)
    e.add_argument("--forcing-dir", required=True)
    e.add_argument("--rows", nargs="+", required=True, help="names or 1-based rows, e.g. 1-55")
    e.add_argument("--noise", nargs="+", choices=sorted(NOISE), default=["rv", "rw"])
    e.add_argument("--arrival-h", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    e.add_argument("--particles", type=int, default=10_000)
    e.add_argument("--out", required=True)
    i = sub.add_parser("index", help="merge every process's rows into index.json")
    i.add_argument("--out", required=True)
    u = sub.add_parser("upload")
    u.add_argument("--src", required=True)
    u.add_argument("--repo", required=True)
    u.add_argument("--path-in-repo", required=True)
    args = parser.parse_args(argv)

    if args.command == "upload":
        upload(Path(args.src), args.repo, args.path_in_repo)
        return
    if args.command == "index":
        index = write_index(Path(args.out), None, None)
        print(f"{len(index['scenarios'])} scenarios in {Path(args.out) / 'index.json'}")
        return

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "score_scenarios", Path(__file__).with_name("score_scenarios.py"))
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)

    table = read_scenarios(args.csv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in score.row_names(args.rows, table):
        row = table[table["scenario"] == name].iloc[0]
        t0 = time.perf_counter()
        entries.append(export_row(row, args.forcing_dir, args.noise, args.arrival_h,
                                  args.particles, out, GREEDY))
        print(f"{name}: {len(entries[-1]['windows'])} windows in "
              f"{time.perf_counter() - t0:.1f} s", flush=True)
    write_index(out, entries, {"particles": args.particles, "noise": NOISE, "greedy": GREEDY,
                               "arrival_h": args.arrival_h, "searchers": list(SEARCHERS),
                               "commit": score.git_commit()})


if __name__ == "__main__":
    main()
