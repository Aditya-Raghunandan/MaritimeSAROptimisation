"""poster_figures.py: the open-day poster's figures and numbers, from the published 55-buoy benchmark (scenarios/v4)."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE = "https://huggingface.co/datasets/AdityaRugs/MaritimeSARoperations/resolve/main/scenarios/v4"   # frontend/.env.example
CACHE = Path("data/processed/poster_cache")
OUT = Path("figures/poster")
CELLS_ACROSS = 150      # display grid across each cloud, a choice so it prints smooth at any scale
BLUR_CELLS = 2.5        # Gaussian smoothing in display cells, a choice for looks only
RING_SHARE = 99.0       # the traced outline encloses this percentage of the particles, a choice
DPI = 300               # print resolution, a choice for the A0 poster
ANNOTATE = False        # keys, scale bars and labels; off for the poster, where they print too small to read

# The poster's own palette, sampled from the poster; the series hues passed the dataviz validator (all pairs).
NAVY, ORANGE, BLUE, AQUA, INK2 = "#0b2545", "#f26b1d", "#2a78d6", "#1baf7a", "#52514e"
EMBER = LinearSegmentedColormap.from_list("ember", ["#fff1e6", "#fcc39b", "#f7904f", ORANGE, "#b5420b", "#6b2506"])
COLOURS = {"expanding-square": BLUE, "parallel": AQUA, "sector": NAVY, "greedy": INK2, "random": INK2,
           "trackline": INK2, "ppo": ORANGE}
NAMES = {"expanding-square": "Expanding Square", "parallel": "Parallel Track", "sector": "Sector search",
         "greedy": "Greedy", "random": "Random", "trackline": "Trackline", "ppo": "Learning searcher"}
ARRIVALS = ("0.25h", "0.5h", "0.75h", "1h", "2h", "3h")
MARKER_KEY = "datum marker (drifts with the water; the search is centred on it)"


def fetch(rel: str) -> Path:
    """A published file, downloaded once into the cache."""
    path = CACHE / rel
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{BASE}/{rel}", path)
    return path


def record_of(scenario: str, arrival: str) -> dict:
    return json.loads(fetch(f"{scenario}/rvc_{arrival}.json").read_text())


def cloud_of(scenario: str, arrival: str) -> np.ndarray:
    """A buoy's cloud, [frame][particle][east_m, north_m] from the marker."""
    record = record_of(scenario, arrival)
    cloud = np.fromfile(fetch(f"{scenario}/{record['cloud']}"), dtype="<f4")
    return cloud.reshape(int(record["frames"]), int(record["particles"]), 2)


def path_km(flight: dict) -> tuple[np.ndarray, np.ndarray]:
    """The flown path in km from the marker, as scripts/produce_maps.py draws it."""
    east = [0.0] + [e for s in flight["steps"] for e in s["east_m"] if e is not None]
    north = [0.0] + [n for s in flight["steps"] for n in s["north_m"] if n is not None]
    return np.asarray(east) / 1000.0, np.asarray(north) / 1000.0


def cross(ax, x: float, y: float, size: float = 18, z: int = 6) -> None:
    """The datum marker: a navy x on a white disc so it reads even where paths meet."""
    ax.plot(x, y, marker="o", markersize=size * 1.15, color="white", markeredgewidth=0, linestyle="none", zorder=z)
    ax.plot(x, y, marker="x", markersize=size, markeredgewidth=8, color="white", linestyle="none", zorder=z)
    ax.plot(x, y, marker="x", markersize=size, markeredgewidth=3.2, color=NAVY, linestyle="none", zorder=z + 1)


def marker_handle() -> Line2D:
    return Line2D([], [], marker="x", markersize=11, markeredgewidth=2.6, color=NAVY, linestyle="none")


def key(fig, handles, labels, ncol: int) -> None:
    """The symbol key under the map, so it never sits on the data."""
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=ncol, frameon=False,
               fontsize=11, labelcolor=NAVY, handlelength=2.2, columnspacing=1.6)


def scale_bar(fig, ax, extent) -> None:
    """A map scale under the map, flush with its left edge: four alternating segments marked 0, half and full."""
    x0, x1 = extent[0], extent[1]
    length = min((0.5, 1, 2, 5, 10, 20), key=lambda k: abs(k - (x1 - x0) / 4))
    ax.apply_aspect()
    pos = ax.get_position()
    height_in = fig.get_size_inches()[1]
    bar = fig.add_axes([pos.x0, pos.y0 - 0.22 / height_in, length / (x1 - x0) * pos.width, 0.075 / height_in])
    bar.set_xlim(0, length)
    bar.set_ylim(0, 1)
    bar.axis("off")
    for k in range(4):
        bar.add_patch(plt.Rectangle((k * length / 4, 0), length / 4, 1, facecolor=NAVY if k % 2 == 0 else "white",
                                    edgecolor=NAVY, linewidth=1.0, clip_on=False))
    for v, text in ((0, "0"), (length / 2, f"{length / 2:g}"), (length, f"{length:g} km")):
        bar.text(v, -0.7, text, ha="left" if v == length else "center", va="top", fontsize=10, color=NAVY,
                 clip_on=False)


def north_arrow(ax, extent) -> None:
    x0, x1, y0, y1 = extent
    x, y = x1 - 0.05 * (x1 - x0), y1 - 0.15 * (y1 - y0)
    ax.annotate("", xy=(x, y + 0.08 * (y1 - y0)), xytext=(x, y),
                arrowprops={"arrowstyle": "-|>", "color": NAVY, "linewidth": 2}, zorder=8)
    ax.text(x, y + 0.09 * (y1 - y0), "N", ha="center", va="bottom", fontsize=12, color=NAVY, weight="bold", zorder=8)


def bare(ax, extent) -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")


def square_extent(points_e, points_n, pad: float = 0.1) -> tuple[float, float, float, float]:
    """A square box around everything given, with a margin, centred on it."""
    e, n = np.concatenate(points_e), np.concatenate(points_n)
    cx, cy = (e.min() + e.max()) / 2, (n.min() + n.max()) / 2
    half = max(e.max() - e.min(), n.max() - n.min()) / 2 * (1 + pad)
    return cx - half, cx + half, cy - half, cy + half


def draw_flights(record: dict, searchers, extent, path: Path) -> Path:
    """Paths about the datum marker on white, the learner on top, with a scale bar and a symbol key."""
    fig, ax = plt.subplots(figsize=(6.4, 7.0) if ANNOTATE else (6.4, 6.4))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.85 / 7.0 if ANNOTATE else 0.01)
    order = [s for s in searchers if s != "ppo"] + [s for s in searchers if s == "ppo"]
    for s in order:
        e, n = path_km(record["flights"][s])
        ax.plot(e, n, color=COLOURS[s], linewidth=(2.6 if s == "ppo" else 2.0) * (1 if ANNOTATE else 1.8), solid_joinstyle="round",
                zorder=4 if s == "ppo" else 3)
    cross(ax, 0.0, 0.0, size=18 if ANNOTATE else 30)
    bare(ax, extent)
    if not ANNOTATE:
        return save(fig, path)
    north_arrow(ax, extent)
    scale_bar(fig, ax, extent)
    handles = [Line2D([], [], color=COLOURS[s], linewidth=3) for s in searchers] + [marker_handle()]
    key(fig, handles, [NAMES[s] for s in searchers] + ["datum marker"], ncol=len(searchers) + 1)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    return path


def patterns(scenario: str, arrival: str, searchers, out: Path) -> list[Path]:
    """One map per searcher over the same buoy and at the same scale, so they compare side by side."""
    record = record_of(scenario, arrival)
    paths = [path_km(record["flights"][s]) for s in searchers]
    extent = square_extent([p[0] for p in paths], [p[1] for p in paths])
    return [draw_flights(record, [s], extent, out / f"pattern_{s}_{scenario}_{arrival}.png") for s in searchers]


def flight(record_path: Path, searchers, out: Path) -> Path:
    """Searchers overlaid from one window JSON, such as <run>/test/flights/<window>_30m.json from sar.rl.fly."""
    record = json.loads(Path(record_path).read_text())
    missing = [s for s in searchers if s not in record["flights"]]
    if missing:
        raise SystemExit(f"{record_path} has no flights for {missing}; it has {list(record['flights'])}")
    paths = [path_km(record["flights"][s]) for s in searchers]
    extent = square_extent([p[0] for p in paths], [p[1] for p in paths])
    return draw_flights(record, searchers, extent, out / f"flight_{Path(record_path).stem}.png")


def blur(h: np.ndarray, sigma: float) -> np.ndarray:
    """A separable Gaussian blur, so the figure needs only numpy."""
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    h = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, h)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, h)


def heat(ax, east_km, north_km, extent) -> None:
    """The cloud as particle density, smoothed, light to dark orange; the thin edge left white."""
    x0, x1, y0, y1 = extent
    h, _, _ = np.histogram2d(east_km, north_km, bins=CELLS_ACROSS, range=[[x0, x1], [y0, y1]])
    h = np.sqrt(blur(h.T, BLUR_CELLS))
    h = np.ma.masked_less(h, 0.04 * h.max())
    ax.imshow(h, origin="lower", extent=extent, cmap=EMBER, interpolation="bilinear", zorder=1)


def outline(ax, east_km, north_km, extent) -> None:
    """A fine line traced around the cloud's edge, enclosing RING_SHARE percent of its particles."""
    x0, x1, y0, y1 = extent
    h, xe, ye = np.histogram2d(east_km, north_km, bins=CELLS_ACROSS, range=[[x0, x1], [y0, y1]])
    h = blur(h.T, BLUR_CELLS)
    ranked = np.sort(h.ravel())[::-1]
    level = ranked[np.searchsorted(np.cumsum(ranked), RING_SHARE / 100.0 * ranked.sum())]
    ax.contour((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, h, levels=[level], colors=NAVY,
               linewidths=0.9 if ANNOTATE else 2.2, zorder=5)


def monte_carlo(scenario: str, arrivals, out: Path) -> Path:
    """The same buoy's cloud at several times after the call, side by side at one scale, each outlined."""
    clouds = []
    for a in arrivals:
        cloud = cloud_of(scenario, a)
        e, n = cloud[0, :, 0] / 1000.0, cloud[0, :, 1] / 1000.0
        ce, cn = np.median(e), np.median(n)
        ring = np.percentile(np.hypot(e - ce, n - cn), RING_SHARE)
        clouds.append((e - ce, n - cn, ring, a))
    gap = 0.3 * max(c[2] for c in clouds)
    centres, x = [0.0], 0.0
    for prev, nxt in zip(clouds, clouds[1:]):
        x += prev[2] + gap + nxt[2]
        centres.append(x)
    tall = max(c[2] for c in clouds)
    extent = (-clouds[0][2] - gap, centres[-1] + clouds[-1][2] + gap, -tall * 1.35, tall * 1.12)

    height = 12 * (extent[3] - extent[2]) / (extent[1] - extent[0]) + 0.9
    if not ANNOTATE:
        extent = (extent[0], extent[1], -tall * 1.12, tall * 1.12)
        height = 12 * (extent[3] - extent[2]) / (extent[1] - extent[0])
    fig, ax = plt.subplots(figsize=(12, height))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.85 / height if ANNOTATE else 0.01)
    for (e, n, ring, a), cx in zip(clouds, centres):
        half = ring * 1.4
        grid = (cx - half, cx + half, -half, half)   # each cloud on its own grid, so a small one stays sharp
        heat(ax, e + cx, n, grid)
        outline(ax, e + cx, n, grid)
        cross(ax, cx, 0.0, size=(9 if ring < 0.08 * tall else 14) * (1 if ANNOTATE else 1.8))
        label = {"0.25h": "15 min", "0.5h": "30 min", "0.75h": "45 min"}.get(a, a.replace("h", " h"))
        if ANNOTATE:
            ax.text(cx, -tall * 1.2, f"after {label}", ha="center", va="center", fontsize=15, weight="bold",
                color=NAVY, zorder=8)
    bare(ax, extent)
    scale_bar(fig, ax, extent)
    handles = [Patch(facecolor=ORANGE, edgecolor="none"), Line2D([], [], color=NAVY, linewidth=1.1),
               marker_handle()]
    key(fig, handles, ["possible positions (darker = more likely)",
                       f"outline around {RING_SHARE:g}% of them", "datum marker"], ncol=3)
    path = out / f"monte_carlo_{scenario}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    return path


def benchmark() -> pd.DataFrame:
    return pd.DataFrame(json.loads(fetch("benchmark.json").read_text())["scenarios55"])


def scores(searchers, out: Path) -> Path:
    """Share of the 55 real buoys each searcher found, by how long the helicopter took to arrive."""
    b = benchmark()
    found = b.pivot_table(index="arrival_h", columns="searcher", values="found") * 100
    x = np.arange(len(found.index))
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in [s for s in searchers if s != "ppo"] + ["ppo"]:
        ax.plot(x, found[s], color=COLOURS[s], linewidth=3.2 if s == "ppo" else 2.2,
                marker="o", markersize=8, markeredgecolor="white", markeredgewidth=2, zorder=4 if s == "ppo" else 3)
    ends = {s: found[s].iloc[-1] for s in searchers}
    placed = {}
    for s in sorted(ends, key=ends.get):
        y = ends[s]
        while any(abs(y - p) < 5 for p in placed.values()):
            y += 5
        placed[s] = y
        ax.text(x[-1] + 0.15, y, NAMES[s], va="center", fontsize=11, color=NAVY,
                weight="bold" if s == "ppo" else "normal")
    ax.set_xticks(x, [{0.25: "15 min", 0.5: "30 min", 0.75: "45 min"}.get(a, f"{a:g} h") for a in found.index])
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.3, x[-1] + 1.6)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("helicopter arrives after", color=INK2, fontsize=11)
    ax.set_ylabel("buoys found (of 55)", color=INK2, fontsize=11)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, length=0, labelsize=11)
    path = out / "scores_found_by_arrival.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    return path


def numbers() -> None:
    """Every number the poster quotes, from the 55 buoys: found share, POS and the time to find."""
    b = benchmark()
    found = b.pivot_table(index="arrival_h", columns="searcher", values="found")
    pos = b.pivot_table(index="arrival_h", columns="searcher", values="pos")
    print("Share of the 55 buoys found (success rate):")
    print((found * 100).round(1).to_string())
    print("\nMean POS (share of the cloud searched):")
    print((pos * 100).round(1).to_string())
    print("\nLearning searcher against the Expanding Square, found:")
    for a in found.index:
        p, s = found.loc[a, "ppo"], found.loc[a, "expanding-square"]
        print(f"  {a:>5g} h  {p * 55:.0f} vs {s * 55:.0f} buoys   {100 * (p - s) / s:+.0f}% relative")
    rows = []
    for a in ARRIVALS:
        for k in range(1, 56):
            f = {n: v["python"] for n, v in record_of(f"S{k:02d}", a)["flights"].items()}
            if f["ppo"].get("found") and f["expanding-square"].get("found"):
                rows.append((a, (f["expanding-square"]["found_s"] - f["ppo"]["found_s"]) / 60.0))
    t = pd.DataFrame(rows, columns=["arrival", "square_minus_learner_min"]).groupby("arrival", sort=False)
    print("\nWhen both found the buoy, minutes the learner was sooner (negative: the square was sooner):")
    print(t["square_minus_learner_min"].agg(["mean", "median", "count"]).round(1).to_string())


def _cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("patterns", help="each searcher's path over one published buoy, one image each")
    a.add_argument("--scenario", default="S55", help="S01 to S55")
    a.add_argument("--arrival", default="1h", choices=ARRIVALS)
    a.add_argument("--searchers", nargs="+", default=["expanding-square", "sector", "parallel", "ppo"],
                   choices=list(NAMES))
    f = sub.add_parser("flight", help="searchers overlaid from a window JSON written by sar.rl.fly")
    f.add_argument("--record", required=True, help="such as <run>/test/flights/<window>_30m.json")
    f.add_argument("--searchers", nargs="+", default=["expanding-square", "ppo"], choices=list(NAMES))
    m = sub.add_parser("montecarlo", help="one buoy's cloud growing after the call")
    m.add_argument("--scenario", default="S55")
    m.add_argument("--arrivals", nargs="+", default=["0.25h", "1h", "3h"], choices=ARRIVALS)
    s = sub.add_parser("scores", help="buoys found by each searcher, by arrival time")
    s.add_argument("--searchers", nargs="+", default=["expanding-square", "sector", "parallel", "ppo"],
                   choices=list(NAMES))
    sub.add_parser("numbers", help="print the poster's numbers")
    for q in (a, f, m, s):
        q.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)

    if args.command == "numbers":
        numbers()
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "patterns":
        written = patterns(args.scenario, args.arrival, args.searchers, out)
    elif args.command == "flight":
        written = [flight(Path(args.record), args.searchers, out)]
    elif args.command == "montecarlo":
        written = [monte_carlo(args.scenario, args.arrivals, out)]
    else:
        written = [scores(args.searchers, out)]
    for w in written:
        print(w)


if __name__ == "__main__":
    _cli()
