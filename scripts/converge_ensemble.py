"""R1f's ladder: separate full runs at each N, reduced as ensemble_heatmap.py does, against N."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from sar.pipeline.ensemble import add_run_arguments, forcing_from, random_term, run_ensemble
from sar.utils.geo import M_PER_DEG_LAT, metres_per_degree_lon

_HEATMAP = importlib.util.spec_from_file_location(
    "ensemble_heatmap", Path(__file__).with_name("ensemble_heatmap.py"))
ensemble_heatmap = importlib.util.module_from_spec(_HEATMAP)
_HEATMAP.loader.exec_module(ensemble_heatmap)

# D024's ladder, ten times per rung.
LADDER = (10**2, 10**3, 10**4, 10**5, 10**6)
STATISTICS = ("centroid_lat", "centroid_lon", "spread_km", "area90_km2")


def rung(args, forcing, n, seeds) -> dict:
    """One N, repeated over independent spawned seeds, reduced to the spread of each statistic."""
    runs = []
    for seq in seeds:
        ensemble = run_ensemble(forcing, n, args.start, args.lat, args.lon, args.duration,
                                args.timestep, args.datum_sigma_km, seed=seq,
                                save_every=args.duration, **random_term(args))
        reduced = ensemble_heatmap.reduce_cloud(ensemble.lat[-1], ensemble.lon[-1], args.cell_m,
                                                beached=ensemble.beached[-1])
        runs.append({key: reduced[key] for key in (*STATISTICS, "lost", "beached_mass")})

    table = {key: np.array([r[key] for r in runs]) for key in STATISTICS}
    north = (table["centroid_lat"] - table["centroid_lat"].mean()) * M_PER_DEG_LAT / 1000.0
    east = ((table["centroid_lon"] - table["centroid_lon"].mean())
            * metres_per_degree_lon(table["centroid_lat"].mean()) / 1000.0)
    return {"particles": n,
            "repeats": len(runs),
            "centroid_error_km": float(np.sqrt((north**2 + east**2).sum() / (len(runs) - 1))),
            "spread_km": float(table["spread_km"].mean()),
            "spread_error_km": float(table["spread_km"].std(ddof=1)),
            "area90_km2": float(table["area90_km2"].mean()),
            "area90_error_km2": float(table["area90_km2"].std(ddof=1)),
            "lost": int(sum(r["lost"] for r in runs)),
            "beached_mass": float(np.mean([r["beached_mass"] for r in runs]))}


def slope(rows, key) -> float:
    """The log-log slope of an error against N; 1/sqrt(N) is -0.5."""
    n = np.array([r["particles"] for r in rows], dtype=float)
    err = np.array([r[key] for r in rows])
    return float(np.polyfit(np.log10(n), np.log10(err), 1)[0])


def plot(rows, path: Path) -> Path:
    """Centroid and spread scatter against N beside a 1/sqrt(N) line; the 90 % area's mean."""
    plt = ensemble_heatmap._plt()
    n = np.array([r["particles"] for r in rows], dtype=float)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    for key, label in (("centroid_error_km", "centroid"), ("spread_error_km", "spread")):
        err = np.array([r[key] for r in rows])
        left.loglog(n, err, "o-", label=f"{label}, slope {slope(rows, key):.2f}")
        left.loglog(n, err[0] * np.sqrt(n[0] / n), ":", color="grey")
    left.set_xlabel("particles N")
    left.set_ylabel("standard deviation across repeats (km)")
    left.set_title("Scatter; dotted lines fall as $1/\\sqrt{N}$")
    left.legend()
    right.errorbar(n, [r["area90_km2"] for r in rows], [r["area90_error_km2"] for r in rows],
                   fmt="o-", capsize=3)
    right.set_xscale("log")
    right.set_xlabel("particles N")
    right.set_ylabel("90 % area (km$^2$), mean and sd across repeats")
    right.set_title("The 90 % area is biased low until N is large")
    return ensemble_heatmap._save(fig, path, plt)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_run_arguments(parser)
    parser.add_argument("--cell-m", type=float, required=True, help="map cell size in metres")
    parser.add_argument("--rungs", type=int, nargs="+", default=list(LADDER),
                        help="particle counts, default 1e2 to 1e6")
    parser.add_argument("--repeats", type=int, required=True,
                        help="independent runs per rung, at least 2, so each error is measured")
    parser.add_argument("--out", required=True, help="directory for the figure and the JSON")
    args = parser.parse_args(argv)
    if args.repeats < 2:
        parser.error("--repeats must be at least 2: a scatter needs two runs")

    forcing = forcing_from(args)
    base = np.random.SeedSequence(args.seed)
    rows = []
    for n, seq in zip(args.rungs, base.spawn(len(args.rungs))):
        rows.append(rung(args, forcing, n, seq.spawn(args.repeats)))
        print(json.dumps(rows[-1]))

    out = Path(args.out)
    figure = plot(rows, out / "ensemble_convergence.png")
    result = {"seed_entropy": base.entropy, "arguments": vars(args),
              "rungs": rows,
              "slopes": {key: slope(rows, key)
                         for key in ("centroid_error_km", "spread_error_km")},
              "figure": str(figure)}
    figure.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
