"""render_showcase.py: the two 3D showcase loops, one real scenario each (D031, Stage 6).

The working view on the site is the flat probability map. These two are for the open-day
stand and the presentation: one example each, rendered once, from real data, not drawn by
hand.

  mountain   S01, new noise, helicopter 2 h after the call. The probability left in each
             250 m cell about the datum marker, as height, while the Coast Guard's Expanding
             Square flies its 45 minutes. The strip eats a spiral groove into the peak. The
             real buoy is the red pin. The heights are the referee's own weights, replayed
             minute by minute from the published bundle; nothing is smoothed.
  cube       S01 from the call to 4 h: the map is the floor and time goes up. Each slice is
             the cloud at that moment (a sample of its particles), so the stack shows one
             splash spreading and drifting with the Gulf Stream; the red line is where the
             real buoy went, threaded through it.

Writes <out>/mountain.webm, <out>/cube.webm (VP9, looping on the site's attract screen) and
a PNG still of each. Needs ffmpeg on the PATH.

    python scripts/render_showcase.py --bundle DATA/published/scenarios/v1/S01 \\
        --csv DATA/derived/scenarios.csv --forcing-dir DATA --out OUT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

from sar.model.position import CALIBRATED_SIGMA_U  # noqa: E402
from sar.pipeline.ensemble import Ensemble, run_ensemble  # noqa: E402
from sar.pipeline.gridded import GriddedForcing  # noqa: E402
from sar.search.episode import SearchEpisode, Waypoints  # noqa: E402
from sar.search.patterns import MarkerTrack  # noqa: E402
from sar.search.platform import STEP_S  # noqa: E402
from sar.search.scenario import read_scenarios  # noqa: E402
from sar.search.sweep import relative_m  # noqa: E402
from sar.utils.geo import offset_position  # noqa: E402

CELL_M = 250.0
BG = "#121211"
INK = "#c3c2b7"


def load_window(folder: Path, name: str):
    meta = json.loads((folder / f"{name}.json").read_text())
    cloud = np.fromfile(folder / f"{name}.f32", dtype="<f4").reshape(meta["frames"],
                                                                     meta["particles"], 2)
    marker = MarkerTrack(np.array(meta["marker"]["t_s"]), np.array(meta["marker"]["lat"]),
                         np.array(meta["marker"]["lon"]))
    mlat, mlon = marker.at(np.arange(meta["frames"]) * STEP_S)
    lat, lon = offset_position(mlat[:, None], mlon[:, None], cloud[..., 0].astype(float),
                               cloud[..., 1].astype(float))
    times = np.datetime64(meta["window_start_utc"], "us") + (
        np.arange(meta["frames"]) * STEP_S * 1e6).astype("timedelta64[us]")
    window = Ensemble(times, lat, lon, np.full(meta["particles"], meta["weight"]),
                      np.zeros(lat.shape, dtype=bool))
    target = MarkerTrack(np.array(meta["target"]["t_s"]), np.array(meta["target"]["lat"]),
                         np.array(meta["target"]["lon"]))
    return meta, window, marker, target


def style(ax, aspect=(1, 1, 0.55)):
    ax.set_box_aspect(aspect, zoom=1.2)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.07, 0.07, 1.0))
        axis.label.set_color(INK)
        axis.line.set_color("#3a3a37")
    ax.tick_params(colors="#8a8a80", labelsize=8)
    ax.grid(False)


def mountain(folder: Path, out: Path, fps: int = 12) -> Path:
    meta, window, marker, target = load_window(folder, "rv_2h")
    steps = [Waypoints(np.array(s["t_s"]), np.array(s["east_m"]), np.array(s["north_m"]),
                       s.get("heading_deg"))
             for s in meta["flights"]["expanding-square"]["steps"]]
    # The helicopter the bundle was flown with: format 2 turns at its rate (D032), format 1
    # turned at once.
    ep = SearchEpisode(window, marker, heading_deg=meta.get("arrival_heading_deg") or 0.0,
                       turn_rate_deg_s=meta.get("turn_rate_deg_s") or float("inf"))
    half = 14  # cells each way: a 7 km square about the marker
    edges = (np.arange(-half, half + 1)) * CELL_M
    centres = 0.5 * (edges[:-1] + edges[1:]) / 1000.0
    X, Y = np.meshgrid(centres, centres)

    def heights(k):
        lat, lon = ep.particles()
        e, n = relative_m(lat, lon, *ep.marker_position)
        h, _, _ = np.histogram2d(n, e, bins=[edges, edges], weights=ep.weight)
        return h

    first = heights(0)
    top = first.max()
    fig = plt.figure(figsize=(9.6, 5.4), dpi=120, facecolor=BG)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.93)
    ax = fig.add_subplot(projection="3d")
    writer = FFMpegWriter(fps=fps, codec="libvpx-vp9",
                          extra_args=["-b:v", "0", "-crf", "34", "-pix_fmt", "yuv420p"])
    path = out / "mountain.webm"
    with writer.saving(fig, str(path), dpi=120):
        for k in range(len(steps) + 1):
            h = heights(k)
            for spin in range(2):
                ax.clear()
                style(ax)
                ax.plot_surface(X, Y, 100 * h, cmap="viridis", vmin=0, vmax=100 * top,
                                rstride=1, cstride=1, linewidth=0, antialiased=False)
                tt, tlat, tlon = ep.track()
                if len(tt) > 1:
                    mlat, mlon = marker.at(np.clip(tt, 0, ep.duration_s))
                    te, tn = relative_m(tlat, tlon, mlat, mlon)
                    ax.plot(te / 1000, tn / 1000, np.zeros(len(te)), color="#ffffff", lw=1.2)
                t_now = min(k * STEP_S, ep.duration_s)
                blat, blon = target.at(np.clip(t_now, target.t_s[0], target.t_s[-1]))
                be, bn = relative_m(blat, blon, *marker.at(t_now))
                if abs(be) < half * CELL_M and abs(bn) < half * CELL_M:
                    ax.plot([be / 1000] * 2, [bn / 1000] * 2, [0, 100 * top * 1.1],
                            color="#ff4d6d", lw=3)
                ax.set_zlim(0, 100 * top * 1.15)
                ax.set_xlabel("km east of the marker")
                ax.set_ylabel("km north")
                ax.set_zlabel("% of the probability per cell")
                ax.view_init(elev=32, azim=-60 + 1.2 * (2 * k + spin))
                ax.set_title(f"The probability left, {k} min into the search: "
                             f"{100 * ep.metrics()['pos']:.0f} % found", color="#ffffff", fontsize=12)
                writer.grab_frame(facecolor=BG)
            if k == 20:
                fig.savefig(out / "mountain.png", facecolor=BG)
            if k < len(steps):
                ep.fly(steps[k])
    plt.close(fig)
    return path


def cube(csv: Path, forcing_dir: str, out: Path, scenario: str = "S01", fps: int = 15) -> Path:
    table = read_scenarios(csv)
    row = table[table["scenario"] == scenario].iloc[0]
    start = np.datetime64(str(row["start"]), "us")
    forcing = GriddedForcing.from_dir(forcing_dir, start, start + np.timedelta64(4 * 3600, "s"))
    run = run_ensemble(forcing, 10_000, start, float(row["lat"]), float(row["lon"]),
                       4 * 3600.0, STEP_S, datum_sigma_km=0.0, seed=int(row["seed"]),
                       sigma=0.0, sigma_u=CALIBRATED_SIGMA_U, save_every=15 * 60.0)
    hours = (run.times - run.times[0]) / np.timedelta64(1, "h")
    pick = np.random.default_rng(1).choice(run.lat.shape[1], 1500, replace=False)
    blat = [float(row["lat"])] + [float(row[f"lat_{h}h"]) for h in range(1, 5)]
    blon = [float(row["lon"])] + [float(row[f"lon_{h}h"]) for h in range(1, 5)]
    be, bn = relative_m(np.array(blat), np.array(blon), float(row["lat"]), float(row["lon"]))
    fig = plt.figure(figsize=(9.6, 5.4), dpi=120, facecolor=BG)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.93)
    ax = fig.add_subplot(projection="3d")
    cmap = plt.get_cmap("viridis")
    writer = FFMpegWriter(fps=fps, codec="libvpx-vp9",
                          extra_args=["-b:v", "0", "-crf", "34", "-pix_fmt", "yuv420p"])
    path = out / "cube.webm"
    total = 120
    with writer.saving(fig, str(path), dpi=120):
        for f in range(total):
            show = min(len(hours), 1 + int(len(hours) * min(1.0, f / (0.6 * total))))
            ax.clear()
            style(ax)
            for k in range(show):
                e, n = relative_m(run.lat[k, pick], run.lon[k, pick], float(row["lat"]),
                                  float(row["lon"]))
                ax.scatter(e / 1000, n / 1000, np.full(pick.size, hours[k]), s=1.2,
                           color=cmap(hours[k] / 4.0), alpha=0.35, depthshade=False)
            upto = min(5, int(np.floor(hours[show - 1])) + 1)
            ax.plot(be[:upto] / 1000, bn[:upto] / 1000, np.arange(upto), color="#ff4d6d", lw=3,
                    marker="o", ms=4)
            ax.set_xlabel("km east of the splash")
            ax.set_ylabel("km north")
            ax.set_zlabel("hours since the call")
            ax.set_zlim(0, 4)
            ax.view_init(elev=18, azim=-70 + 1.5 * f)
            ax.set_title("One splash becomes a cloud: 10,000 possible people, drifting for 4 h",
                         color="#ffffff", fontsize=12)
            writer.grab_frame(facecolor=BG)
            if f == total - 1:
                fig.savefig(out / "cube.png", facecolor=BG)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bundle", required=True, help="a scenario's bundle folder, e.g. .../v1/S01")
    p.add_argument("--csv", required=True, help="the scenario table, for the cube")
    p.add_argument("--forcing-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("wrote", mountain(Path(args.bundle), out))
    print("wrote", cube(Path(args.csv), args.forcing_dir, out))


if __name__ == "__main__":
    main()
