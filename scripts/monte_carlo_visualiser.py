"""Heatmaps of every CSV a bench_monte_carlo.py run wrote, and a column graph of each run's time against N."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from sar.pipeline.ensemble import read_csv

_HEATMAP = importlib.util.spec_from_file_location(
    "ensemble_heatmap", Path(__file__).with_name("ensemble_heatmap.py"))
ensemble_heatmap = importlib.util.module_from_spec(_HEATMAP)
_HEATMAP.loader.exec_module(ensemble_heatmap)

# A benchmark choice: the first four hours of a search, on a 1 km by 1 km grid.
TIMES = ("0m", "30m", "1h", "2h", "4h")
CELL_M = 1000.0

# Categorical slot 1 of the dataviz skill's reference palette; one series needs one hue.
BAR = "#2a78d6"


def plot_times(runs, path: Path) -> Path:
    """One column per N, its height the seconds run_ensemble took, CSV writing excluded."""
    plt = ensemble_heatmap._plt()
    labels = [f"{r['particles']:,}" for r in runs]
    seconds = [r["run_s"] for r in runs]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, seconds, width=0.6, color=BAR, zorder=2)
    ax.bar_label(bars, labels=[f"{s:.2f} s" for s in seconds], padding=3, color="#333333")
    ax.set_xlabel("particles N")
    ax.set_ylabel("run time (s)")
    ax.set_title("Monte Carlo run time against ensemble size, CSV writing excluded")
    ax.grid(axis="y", color="#dddddd", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    return ensemble_heatmap._save(fig, path, plt)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timings", required=True, help="the bench_*.json bench_monte_carlo.py wrote")
    parser.add_argument("--at", nargs="+", default=list(TIMES),
                        help="saved times to map, default 0m 30m 1h 2h 4h")
    parser.add_argument("--cell-m", type=float, default=CELL_M, help="cell size in metres, default 1000")
    parser.add_argument("--out", required=True, help="directory for the PNGs and their JSONs")
    args = parser.parse_args(argv)

    bench = json.loads(Path(args.timings).read_text())
    runs = sorted(bench["runs"], key=lambda r: r["particles"])
    maps = []
    for r in runs:
        ensemble = read_csv(r["csv"])
        maps += [ensemble_heatmap.write_map(ensemble, r["csv"], at, args.cell_m, 0.0, args.out)
                 for at in args.at]
    figure = plot_times(runs, Path(args.out) / f"{Path(args.timings).stem}_run_time.png")
    return {"maps": maps, "figure": str(figure)}


if __name__ == "__main__":
    main()
