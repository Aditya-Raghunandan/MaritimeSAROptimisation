"""produce_maps.py: each trained run's search flown beside a comparison pattern, on fixed test days.

COMPARISON PATTERNS (--compare, default expanding-square; give several to draw them all)
    expanding-square   the Coast Guard's standard first search (D004), the baseline
    sector             sector search about the datum
    parallel           parallel sweep along the drift
    trackline          trackline along the drift
    greedy             the hand-written searcher that always heads for the most probability
    random             a random heading every minute

WHAT IT DRAWS BY DEFAULT
    Every run_* folder in --runs-dir, at all five places in scripts/rl_locations.txt, on the
    six days below, at all four arrival times: 9 models x 5 places x 6 days x 4 arrivals.
    The days are test weeks, so no model trained on them.
        summer   2019-07-30   2021-07-27   2023-07-25
        winter   2019-01-29   2021-01-26   2023-01-31

WHERE IT WRITES
    <out>/<model>/<place>/<summer|winter>/<date>/<30min|1h|2h|3h>/<model>_<place>_<date>_<arrival>.png
    such as figures/produced_maps/run_heading72/gulf-stream-off-jacksonville/summer/2021-07-27/2h/
            run_heading72_gulf-stream-off-jacksonville_2021-07-27_2h.png

EVERYTHING (what scripts/wrap_run.sbatch runs, one model at a time)
    PYTHONPATH=src python scripts/produce_maps.py --runs-dir /home/26p67/data/derived/rl/runs \
        --windows /home/26p67/data/derived/rl/windows_arr30-60-120-180

ONE MAP: one model, one place, one day, one arrival time, into a folder of your choice
    PYTHONPATH=src python scripts/produce_maps.py --runs-dir /home/26p67/data/derived/rl/runs \
        --windows /home/26p67/data/derived/rl/windows_arr30-60-120-180 \
        --models run_heading72 --locations gulf-stream-off-jacksonville --dates 2021-07-27 \
        --arrivals 120 --compare expanding-square greedy --cloud --out /tmp/check

    --models      run folders to draw            (default: every run_* in --runs-dir)
    --locations   places by name                 (default: all five)
    --dates       any of the six days above      (default: all six)
    --arrivals    30 60 120 180 minutes          (default: all four)
    --compare     patterns above                 (default: expanding-square)
    --cloud       shade the probability at the start of the search (default: off)
    --out         where the tree starts          (default: figures/produced_maps)

Each model flies its best_model.zip (model.zip if there is none) with the most likely heading
each minute, through the same referee as every pattern (sar.rl.fly.fly_window).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from sar.rl.features import Context, time_weighted_return
from sar.rl.fly import PPO, fly_window, load_policy
from sar.rl.windows import load, parse_windows_name
from sar.search.benchmark import SEARCHERS
from sar.search.sweep import relative_m

DATES = {"summer": ("2019-07-30", "2021-07-27", "2023-07-25"),
         "winter": ("2019-01-29", "2021-01-26", "2023-01-31")}
SEASON = {d: season for season, days in DATES.items() for d in days}
ARRIVALS = {30: "30min", 60: "1h", 120: "2h", 180: "3h"}
PLACES = {"florida-current": "Florida Current",
          "gulf-stream-off-jacksonville": "Gulf Stream off Jacksonville",
          "bahamas-off-eleuthera": "Bahamas, off Eleuthera",
          "ring-zone-south-of-the-stream": "Ring zone south of the Stream",
          "sargasso-eddy-field": "Sargasso eddy field"}
LOCATIONS_FILE = Path(__file__).with_name("rl_locations.txt")
DPI = 200
CLOUD_CELL_KM = 0.25   # D007's 250 m cell, the grid greedy and the near map read

# The dataviz reference palette, as scripts/plot_sigma_calibration.py uses it; the model is always orange.
ORANGE, BLUE, AQUA, YELLOW = "#eb6834", "#2a78d6", "#1baf7a", "#eda100"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PATTERN_COLOURS = (BLUE, AQUA, YELLOW, INK2, MUTED, "#8a5cd0")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "font.size": 10,
    "legend.frameon": False, "axes.titlesize": 11, "axes.titleweight": "semibold",
})


def read_locations(path=LOCATIONS_FILE) -> dict[str, tuple[float, float]]:
    """{name: (lat, lon)} from the lat;lon;name lines that stage 1 ran."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            lat, lon, name = line.strip().split(";")
            out[name] = (float(lat), float(lon))
    return out


def find_window(windows_dir, lat: float, lon: float, date: str) -> Path:
    """The windows file called on `date` at the place nearest (lat, lon); its name holds both."""
    stamp = date.replace("-", "")
    for path in sorted(Path(windows_dir).glob(f"windows_*_{stamp}T*.npz")):
        p = parse_windows_name(path)
        if abs(p["lat"] - lat) < 0.006 and abs(p["lon"] - lon) < 0.006:
            return path
    raise FileNotFoundError(f"no windows file for {lat:.2f}, {lon:.2f} on {date} in {windows_dir}")


def path_km(flight: dict) -> tuple[np.ndarray, np.ndarray]:
    """The flown path in km from the marker, from the start at the marker through every waypoint."""
    east = [0.0] + [e for s in flight["steps"] for e in s["east_m"]]
    north = [0.0] + [n for s in flight["steps"] for n in s["north_m"]]
    return np.asarray(east) / 1000.0, np.asarray(north) / 1000.0


def model_rows(run: Path, model) -> list[tuple[str, str]]:
    """The model's settings, one per line, read from the loaded model and the run's config.json."""
    config = json.loads((run / "config.json").read_text())
    args = config["arguments"]
    clip = model.clip_range(1.0) if callable(model.clip_range) else model.clip_range
    lr = model.learning_rate(1.0) if callable(model.learning_rate) else model.learning_rate
    arch = model.policy.net_arch
    net = arch.get("pi", arch) if isinstance(arch, dict) else arch
    rows = [("headings", str(args["headings"])), ("learning rate", f"{lr:g}"),
            ("entropy bonus", f"{model.ent_coef:g}"), ("clip range", f"{clip:g}"),
            ("gamma", f"{model.gamma:g}"), ("GAE lambda", f"{model.gae_lambda:g}"),
            ("n_steps", str(model.n_steps)), ("batch size", str(model.batch_size)),
            ("passes", str(model.n_epochs)), ("network", " × ".join(map(str, net))),
            ("arrival times", " ".join(map(str, args["arrival_min"]))), ("envs", str(args["envs"])),
            ("seed", str(args["seed"])), ("patience", str(args.get("patience", "—"))),
            ("improve-threshold", f"{args['improve_threshold']:g}" if "improve_threshold" in args else "—")]
    trained = config.get("trained_timesteps")
    rows.append(("trained steps", f"{trained:,}" if trained else "—"))
    evals = run / "eval" / "evaluations.npz"
    if evals.exists():
        with np.load(evals) as e:
            means = e["results"].mean(axis=1)
            best = int(np.argmax(means))
            rows.append(("best validation", f"{means[best]:.3f} @ {e['timesteps'][best] / 1e6:.1f}M"))
    return rows


def label(name: str, model_name: str) -> str:
    return model_name if name == PPO else name.replace("-", " ").title()


def draw_map(run: Path, model, headings: int, windows_path: Path, arrival_min: int, place: str,
             date: str, compare, cloud: bool, out: Path) -> Path:
    """Fly the model and the comparison patterns over one window and draw the approved map."""
    windows = load(windows_path)
    minutes = [round(a / 60.0) for a in windows.arrivals_s]
    if arrival_min not in minutes:
        raise ValueError(f"{windows_path.name} has arrival times {minutes}, not {arrival_min}")
    setup = windows.setup(minutes.index(arrival_min))
    context = Context(windows.lkp, setup.arrival_s, setup.drift_bearing_deg)
    row = {"lat": windows.lkp[0], "lon": windows.lkp[1], "seed": windows.seed}
    flights = fly_window(setup, context, row, [PPO, *compare], model, headings)
    names = [*compare, PPO]                       # the model last, so it is drawn on top
    colours = {PPO: ORANGE, **{c: PATTERN_COLOURS[i % len(PATTERN_COLOURS)] for i, c in enumerate(compare)}}

    fig = plt.figure(figsize=(13, 8.4))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.65, 1), height_ratios=(3, 1.05), wspace=0.08, hspace=0.32,
                            top=0.94)
    ax, drain, panel = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[:, 1])

    for name in names:
        east, north = path_km(flights[name])
        ax.plot(east, north, color=colours[name], linewidth=2.2 if name == PPO else 1.6,
                zorder=3 if name == PPO else 2, label=label(name, run.name))
    ax.plot(0, 0, marker="+", markersize=14, markeredgewidth=2.4, color=INK, linestyle="none",
            zorder=4, label="datum marker")
    ax.set_aspect("equal", adjustable="datalim")
    if cloud:
        east, north = relative_m(setup.window.lat[0], setup.window.lon[0], setup.marker.lat[0], setup.marker.lon[0])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        bins = [min(400, max(1, round((x1 - x0) / CLOUD_CELL_KM))), min(400, max(1, round((y1 - y0) / CLOUD_CELL_KM)))]
        heat, _, _ = np.histogram2d(np.asarray(east) / 1000.0, np.asarray(north) / 1000.0, bins=bins,
                                    range=[[x0, x1], [y0, y1]], weights=setup.window.weight)
        heat = np.ma.masked_equal(heat.T, 0)
        if heat.count():
            ax.imshow(heat, origin="lower", extent=(x0, x1, y0, y1), cmap="Greys", alpha=0.8, zorder=0,
                      aspect="auto", norm=LogNorm(vmin=heat.min(), vmax=heat.max()))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    ax.set(xlabel="km east of the marker", ylabel="km north of the marker")
    ax.legend(loc="upper right", fontsize=9)

    minute = np.arange(1, 46)
    for name in names:
        drain.plot(minute, flights[name]["python"]["drain_rate"], color=colours[name],
                   linewidth=1.8 if name == PPO else 1.3, zorder=3 if name == PPO else 2)
    drain.set(xlim=(1, 45), xlabel="minute of the search", ylabel="found")
    drain.set_title("Drain rate: probability found each minute", loc="left", fontsize=10)

    panel.axis("off")
    rows: list[tuple[str, str] | str] = ["RESULT", "POS"]
    pos = {n: flights[n]["python"]["pos"] for n in names}
    rows += [(f"  {label(n, run.name)}", f"{pos[n]:.3f}") for n in [PPO, *compare]]
    rows.append((f"  difference vs {label(compare[0], run.name)}", f"{pos[PPO] - pos[compare[0]]:+.3f}"))
    rows.append("Reward (time-weighted)")
    rows += [(f"  {label(n, run.name)}", f"{time_weighted(flights[n]):.3f}") for n in [PPO, *compare]]
    rows.append("Expected time to detection")
    rows += [(f"  {label(n, run.name)}", ttd(flights[n])) for n in [PPO, *compare]]
    rows += ["", "MODEL", *model_rows(run, model)]
    y = 0.99
    for r in rows:
        if isinstance(r, str):
            panel.text(0.0, y, r, fontsize=10.5 if r.isupper() else 9.5, weight="bold" if r.isupper() else "normal",
                       color=INK if r.isupper() else INK2, transform=panel.transAxes, va="top")
        else:
            panel.text(0.0, y, r[0], fontsize=9.5, family="monospace", color=INK2, transform=panel.transAxes, va="top")
            panel.text(1.0, y, r[1], fontsize=9.5, family="monospace", color=INK, ha="right",
                       transform=panel.transAxes, va="top")
        y -= 0.027 if r else 0.012

    tag = ARRIVALS.get(arrival_min, f"{arrival_min}min")
    day = np.datetime64(date).astype(object).strftime("%d %b %Y").lstrip("0")
    fig.suptitle(f"{run.name} · {PLACES.get(place, place)} · {day} · arrival {tag.replace('min', ' min')}",
                 fontsize=13, weight="semibold", x=0.06, ha="left")
    folder = out / run.name / place / SEASON.get(date, "other") / date / tag
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{run.name}_{place}_{date}_{tag}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def time_weighted(flight: dict) -> float:
    return time_weighted_return(flight["python"]["drain_rate"])


def ttd(flight: dict) -> str:
    t = flight["python"].get("expected_ttd_s")
    return "—" if t is None or not np.isfinite(t) else f"{t / 60.0:.1f} min"


def _cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--runs-dir", required=True, help="the folder holding the run_* folders")
    p.add_argument("--windows", required=True, help="the windows_*.npz folder the runs trained on")
    p.add_argument("--models", nargs="+", help="run folders to draw (default: every run_* in --runs-dir)")
    p.add_argument("--locations", nargs="+", choices=list(PLACES), default=list(PLACES))
    p.add_argument("--dates", nargs="+", choices=list(SEASON), default=list(SEASON))
    p.add_argument("--arrivals", nargs="+", type=int, choices=list(ARRIVALS), default=list(ARRIVALS))
    p.add_argument("--compare", nargs="+", choices=list(SEARCHERS), default=["expanding-square"])
    p.add_argument("--cloud", action="store_true", help="shade the probability at the start of the search")
    p.add_argument("--out", default="figures/produced_maps", help="where the folder tree starts")
    p.add_argument("--locations-file", default=str(LOCATIONS_FILE), help="lat;lon;name lines")
    args = p.parse_args(argv)

    runs = Path(args.runs_dir)
    names = args.models or sorted(d.name for d in runs.glob("run_*") if d.is_dir())
    if not names:
        p.error(f"no run_* folders in {runs}")
    places = read_locations(args.locations_file)
    out = Path(args.out)
    for name in names:
        run = runs / name
        best = run / "best_model.zip"
        model, headings = load_policy(best if best.exists() else run / "model.zip")
        for place in args.locations:
            for date in args.dates:
                windows_path = find_window(args.windows, *places[place], date)
                for arrival in args.arrivals:
                    print(draw_map(run, model, headings, windows_path, arrival, place, date,
                                   args.compare, args.cloud, out))


if __name__ == "__main__":
    _cli()
