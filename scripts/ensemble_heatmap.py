"""Condense one ensemble CSV into a probability heatmap at one saved time, through sar.model.grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sar.model.grid import ProbabilityGrid, normalise
from sar.pipeline.ensemble import format_span, parse_span, read_csv
from sar.utils.geo import M_PER_DEG_LAT, metres_per_degree_lon, to_display_longitude
from sar.viz.fields import _display_axes, _plt, _save

# The share of mass the contour R2c validates against encloses (ADR002).
CONTOUR_MASS = 0.9


def select_time(ensemble, at) -> int:
    """The saved state at an offset such as '24h', or at an ISO instant."""
    try:
        wanted = ensemble.times[0] + np.timedelta64(round(parse_span(at) * 1e6), "us")
    except ValueError:
        wanted = np.datetime64(at, "us")
    hits = np.flatnonzero(ensemble.times == wanted)
    if hits.size == 0:
        raise ValueError(f"{at!r} is {wanted}, which is not a saved time; the run saved "
                         f"{ensemble.times[0]} to {ensemble.times[-1]} in {len(ensemble.times)} states")
    return int(hits[0])


def map_statistics(grid, p) -> dict:
    """Centroid, spread per axis and 90 % contour area of a normalised map."""
    lat_c = float((p.sum(axis=1) * grid.lats).sum())
    lon_c = float((p.sum(axis=0) * grid.lons).sum())
    sd_north = np.sqrt((p.sum(axis=1) * (grid.lats - lat_c) ** 2).sum()) * M_PER_DEG_LAT
    sd_east = (np.sqrt((p.sum(axis=0) * (grid.lons - lon_c) ** 2).sum())
               * float(metres_per_degree_lon(lat_c)))
    ranked = np.cumsum(np.sort(p.ravel())[::-1])
    cells = int(np.searchsorted(ranked, CONTOUR_MASS * ranked[-1]) + 1)
    ns, ew = grid.cell_size_m
    return {"centroid_lat": lat_c,
            "centroid_lon": float(to_display_longitude(lon_c)),
            "sd_north_km": float(sd_north / 1000.0),
            "sd_east_km": float(sd_east / 1000.0),
            "spread_km": float(np.sqrt((sd_north**2 + sd_east**2) / 2) / 1000.0),
            "area90_km2": cells * ns * ew / 1e6}


def reduce_cloud(lats, lons, cell_m, margin_km=0.0, beached=None) -> dict:
    """docs/probability-grid.md's four lines on one instant's cloud, with its statistics."""
    grid = ProbabilityGrid.from_envelope(lats, lons, cell_m=cell_m, margin_km=margin_km)
    counts, lost = grid.bin(lats, lons)
    beached_count = int(np.count_nonzero(beached)) if beached is not None else 0
    p = normalise(counts, beached_mass=beached_count)
    return {"grid": grid, "p": p, "spec": grid.to_spec(), "lost": lost,
            "beached_mass": beached_count / len(lats), **map_statistics(grid, p)}


def render(reduced, path: Path, title: str) -> Path:
    """The map as colour, following sar.viz.fields.speed_map."""
    plt = _plt()
    grid = reduced["grid"]
    lat, lon_d, p = _display_axes(grid.lats, grid.lons, reduced["p"])
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.pcolormesh(lon_d, lat, np.ma.masked_equal(p, 0.0), shading="auto", cmap="magma_r")
    ax.plot(reduced["centroid_lon"], reduced["centroid_lat"], "+", color="tab:blue", ms=14)
    ax.set_xlabel("longitude (deg E, negative west)")
    ax.set_ylabel("latitude (deg N)")
    ax.set_aspect(1 / np.cos(np.radians(grid.centre_lat)), adjustable="box")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="probability per cell")
    return _save(fig, path, plt)


def write_map(ensemble, csv, at, cell_m, margin_km, out) -> dict:
    """One saved time of an already read ensemble as a PNG and the JSON beside it."""
    k = select_time(ensemble, at)
    reduced = reduce_cloud(ensemble.lat[k], ensemble.lon[k], cell_m, margin_km,
                           ensemble.beached[k])
    offset = (ensemble.times[k] - ensemble.times[0]) / np.timedelta64(1, "s")
    stem = f"{Path(csv).stem}_at{format_span(offset) if offset else '0s'}_cell{cell_m:g}m"
    png = render(reduced, Path(out) / f"{stem}.png",
                 f"{Path(csv).stem}\n{ensemble.times[k]}, {cell_m:g} m cells")

    record = {k_: v for k_, v in reduced.items() if k_ not in ("grid", "p")}
    record.update(csv=str(csv), time=str(ensemble.times[k]), figure=str(png))
    Path(png).with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(f"lost {record['lost']} particles outside the box, beached mass "
          f"{record['beached_mass']:.4f}, spread {record['spread_km']:.3f} km")
    return record


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True, help="a CSV written by sar.pipeline.ensemble")
    parser.add_argument("--at", required=True, help="offset from the start, such as 24h, or ISO")
    parser.add_argument("--cell-m", type=float, required=True, help="cell size in metres (ADR002)")
    parser.add_argument("--margin-km", type=float, default=0.0,
                        help="widen the box beyond the cloud, default none")
    parser.add_argument("--out", required=True, help="directory for the PNG and the JSON")
    args = parser.parse_args(argv)
    return write_map(read_csv(args.csv), args.csv, args.at, args.cell_m, args.margin_km, args.out)


if __name__ == "__main__":
    main()
