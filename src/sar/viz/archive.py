"""
archive.py -- publish the raw NetCDF archive as Zarr the browser can read.

    python -m sar.viz.archive --data /home/26p67/data --out /home/26p67/published

Step 4 of D021. `sar.viz.export` writes ONE window as a flat float32 blob, which
is right for a 48 h bundle and hopeless for five years; this writes the whole
archive as Zarr, at three time resolutions, so a scrub is continuous at every
scale.

WHY ZARR AND NOT THE NetCDF ITSELF
----------------------------------
NetCDF is one self-describing file you open with a library. To read a single
cell out of a 600 MB file a browser would have to fetch the 600 MB; there is no
partial-read story for it that works in a browser without a server in front.

Zarr is the same array split into many small files. Fetching "January 2021 over
the box" is a handful of ordinary HTTP GETs for exactly the chunk files that
cover it -- no server, no index, no library on the host side. ERA5's own
upstream source is a Zarr store and `zarr==3.3.0` is already pinned, so this is
the grain of the project rather than something bolted onto it.

THE ARCHIVE STAYS WHERE IT IS. This writes a DERIVED, downsampled copy for
publication. `/home/26p67/data/raw/*.nc` remains the analysis tier that the
engine, the hindcast validation and the clustering all read directly. The two
are different tiers with different jobs, and nothing here moves the archive.

MEASURED 2026-09-17, on the real 8-day local slice, not assumed
---------------------------------------------------------------
Compression, 9.11 MB of float32 wind:

    none                9.11 MB   1.00x        --
    zstd-3              7.20 MB   1.26x   161 MB/s
    zstd-9              6.58 MB   1.39x   106 MB/s
    zstd-19             5.78 MB   1.58x    20 MB/s   <- chosen
    blosc+zstd+shuffle  7.11 MB   1.28x   -- and REJECTED

zstd-19 is chosen on two grounds. It is the best ratio by a clear margin, and
at 20 MB/s the whole 2.03 GB hourly tier costs **1.7 minutes** to write once --
which is nothing for a one-off export, and zstd decodes at roughly the same
speed whatever level it was written at, so the browser pays nothing for it.

Blosc with a shuffle filter is the usual recommendation for float arrays and is
rejected here for a reason that has nothing to do with its ratio: **zarrita.js
cannot decode it.** Plain zstd is a registered codec in the Zarr v3 core spec
and it can. A 1.28x that the client cannot read is 0x.

CHUNKING: 48 TIMESTEPS x THE WHOLE SPATIAL BOX
----------------------------------------------
One chunk is 48 timesteps of the full 77 x 77 box = **1.11 MB**, at every tier.
Uniform, because the tiers differ in what a timestep *means*, not in its size.

Big enough that per-request overhead is amortised, small enough to fetch
without a stall. And 48 steps is exactly the 48 h scenario window at hourly
cadence -- the organising unit D009 and D021 already use -- so a scenario is
one chunk rather than a straddle.

Chunking the whole box in space rather than tiling it is deliberate: the box is
small (77 x 77 for wind) and every layer in the frontend draws all of it at
once, so a spatial tile would only ever be fetched together with its neighbours.

LONGITUDE IS CONVERTED HERE. D020 stores 0-360 everywhere and converts at the
presentation boundary only. This store is read by a web map, so it is the
presentation boundary, exactly as `sar.viz.export` is.
"""

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from zarr.codecs import ZstdCodec

from sar.utils.data_io import open_forcing
from sar.utils.geo import to_display_longitude

# Timesteps per chunk, PER PRODUCT. A chunk is one HTTP GET, so this number is
# really "how much does the browser download to show one moment".
#
# It is per product because the two grids are nowhere near the same size. ERA5
# is 77 x 77 over the box; HYCOM is 476 x 238 -- NINETEEN TIMES the cells per
# timestep. The same 48-step chunk is therefore:
#
#     wind     48 steps x 77 x 77     ->  1.1 MB over the wire
#     current  48 steps x 476 x 238   -> 29.0 MB over the wire
#
# 29 MB to display a single frame is unusable, and that -- not disk space -- is
# the real constraint on the current archive. Eight timesteps brings it to
# 4.8 MB, comparable to wind, at FULL spatial resolution.
#
# An earlier version halved the current grid instead. That was the wrong lever:
# it threw away half the data to fix a problem that chunking fixes without
# losing anything. 8 steps at 3-hourly is a 24 h chunk, which is also a natural
# unit to scrub through.
TIME_CHUNK = {"wind": 48, "current": 8}
DEFAULT_TIME_CHUNK = 48

# Measured, not picked: 1.58x against 1.26x at level 3, and the 2.03 GB tier
# still writes in 1.7 minutes. Decode cost in the browser is level-independent.
ZSTD_LEVEL = 19

# Stride in source timesteps, per published tier. The source cadence is hourly
# for wind and 3-hourly for current, so the same stride means different things
# for each -- which is why the tier carries the stride and not a cadence.
WIND_TIERS = {"hourly": 1, "6-hourly": 6, "daily": 24}
CURRENT_TIERS = {"3-hourly": 1, "daily": 8}

PRODUCTS = {
    "wind": {"prefix": "era5_", "vars": ("u10", "v10"), "tiers": WIND_TIERS,
             "label": "10 m wind"},
    "current": {"prefix": "hycom_", "vars": ("water_u", "water_v"), "tiers": CURRENT_TIERS,
                "label": "surface current"},
}


def open_archive(data: Path, product: str = "wind") -> tuple[xr.Dataset, list[str]]:
    """Open every raw file for a product as one continuous time axis.

    Returns the dataset and the list of source filenames, in time order.

    Each file goes through `open_forcing`, so pre-D020 files are normalised on
    the way in rather than rejected -- the archive is genuinely mixed, because
    the local slices predate `geo.py` by five days.

    Duplicate timestamps are dropped, keeping the first, and **the overlapping
    files are named in a warning** rather than silently reconciled: two files
    covering the same hour is either a re-fetch or a boundary bug, and the
    caller should know which. `open_box_means` already does this for the
    derived Parquet; the reason is the same.
    """
    spec = PRODUCTS[product]
    chunk = TIME_CHUNK.get(product, DEFAULT_TIME_CHUNK)
    files = sorted((data / "raw").glob(f"{spec['prefix']}*.nc"))
    if not files:
        raise FileNotFoundError(f"no {spec['prefix']}*.nc in {data / 'raw'}")

    parts, names = [], []
    for f in files:
        ds = open_forcing(f)
        missing = [v for v in spec["vars"] if v not in ds]
        if missing:
            ds.close()
            raise KeyError(f"{f.name} has no {missing} -- is it a {product} file?")
        # CHUNK EACH FILE BEFORE IT IS CONCATENATED. This one line is the
        # difference between the current export running and being SIGKILLed.
        #
        # Measured on the cluster, 60 files, peak RSS at each stage:
        #
        #     after 60 x open_forcing     0.22 GB
        #     after xr.concat            26.09 GB   <-- here
        #
        # xr.concat on lazily-indexed numpy backends materialises everything it
        # is given. Chunking afterwards cannot help, because the memory is
        # already gone by then -- two attempts died that way before the stages
        # were actually measured rather than reasoned about. With dask in place
        # first, the concat is a graph and peak memory is a few chunks.
        parts.append(ds[list(spec["vars"])].chunk(
            {"time": chunk, "lat": -1, "lon": -1}))
        names.append(f.name)

    combined = xr.concat(parts, dim="time").sortby("time")

    # Already dask, because each part was chunked before the concat. Re-stated
    # here so a later edit that removes the per-part chunk still has a graph,
    # and because a no-op rechunk costs nothing.
    combined = combined.chunk({"time": chunk, "lat": -1, "lon": -1})

    times = combined["time"].values
    duplicated = np.zeros(times.size, dtype=bool)
    duplicated[1:] = times[1:] == times[:-1]
    if duplicated.any():
        overlapping = [
            n for n, p in zip(names, parts)
            if np.isin(p["time"].values, times[duplicated]).any()
        ]
        warnings.warn(
            f"{int(duplicated.sum())} duplicate timestamps across {len(overlapping)} "
            f"overlapping file(s): {', '.join(overlapping)}. Keeping the first of each. "
            "Two files covering the same hour is a re-fetch or a boundary bug -- "
            "check which before publishing.",
            stacklevel=2,
        )
        combined = combined.isel(time=~duplicated)

    return combined, names


def to_display_grid(ds: xr.Dataset) -> xr.Dataset:
    """Convert longitude to -180..180 and re-sort. The presentation boundary.

    D020 stores 0-360; every web map and GeoJSON (RFC 7946) require -180..180.
    This happens here and nowhere downstream, the same rule `sar.viz.export`
    follows.
    """
    return ds.assign_coords(lon=to_display_longitude(ds["lon"].values)).sortby("lon")


def regularise_time(ds: xr.Dataset) -> tuple[xr.Dataset, dict]:
    """Reindex onto a gap-free axis at the modal cadence, inserting NaN.

    HYCOM really is missing timesteps: measured against the live server, the
    study window has **five** irregular steps -- four of 6 h and one of 12 h
    against a 3-hourly cadence -- which D011 already records as 7 missing steps
    in 14,608 (0.048 %).

    There were three ways to publish that and only one is honest.

    Interpolating across them invents current fields that were never modelled,
    in a store whose whole purpose is to be the thing other results are checked
    against. Publishing the squashed axis (`--allow-gaps`) keeps every real
    value but makes the client read a `time` array to know what it is looking
    at, and leaves a 12 h jump silently rendered as a 3 h step.

    This does the third: put the missing timesteps back as NaN. The axis becomes
    regular, so `start + k * step` is valid again and the client stays simple;
    the absent hours are absent rather than interpolated; and every renderer
    already treats NaN as "no data here" because the land mask taught them to.
    Nothing is invented and nothing is hidden.

    Only gaps that are exact multiples of the modal step can be filled this way.
    Anything else means the cadence itself is wrong, and that is not a hole to
    paper over -- it raises.
    """
    times = ds["time"].values
    if times.size < 2:
        return ds, {"inserted": 0, "original": int(times.size)}

    steps = np.diff(times)
    modal = np.median(steps).astype(steps.dtype)
    ragged = [s for s in np.unique(steps) if s % modal != np.timedelta64(0, "ns")]
    if ragged:
        raise ValueError(
            f"time steps {[str(r) for r in ragged]} are not whole multiples of the "
            f"modal cadence {modal} -- the cadence is wrong, not merely gapped"
        )

    full = np.arange(times[0], times[-1] + modal, modal)
    inserted = int(full.size - times.size)
    if inserted == 0:
        return ds, {"inserted": 0, "original": int(times.size)}

    # reindex puts NaN in every variable at the inserted stamps.
    out = ds.reindex(time=full)
    return out, {"inserted": inserted, "original": int(times.size)}


def _iso(t) -> str:
    """A full ISO-8601 UTC stamp, whatever precision the array carries.

    `str(np.datetime64)` drops trailing components, so an hour-precision array
    yields "2021-01-01T23" and a nanosecond one "2021-01-01T23:00:00.000000000".
    The client parses these with `new Date(...)`, which is happy with both and
    would make the manifest's format depend on how the source happened to be
    written. Cast to seconds first so it never does.
    """
    return str(np.datetime64(t, "s")) + "Z"


def time_axis_spec(times: np.ndarray, *, allow_gaps: bool = False) -> dict:
    """Describe a time axis, refusing an irregular one unless asked.

    The client reconstructs every frame's timestamp as `start + k * step`,
    exactly as it reconstructs coordinates from `lat0 + j * dlat`. That is only
    valid on a regular axis, and a gapped axis does not fail -- it mislabels.

    Measured on the real local archive 2026-09-17: eight days of January plus
    two days of March concatenate to 264 frames with one **1225-hour** gap.
    Reconstructed from start and step, the last frame comes out at 2021-01-11
    instead of 2021-03-03 -- **51 days wrong**, on a map that renders perfectly.

    `grid_spec` in sar.viz.export already refuses an irregular lat/lon axis for
    this reason. This is the same guard on the third axis, which did not have
    one. With `allow_gaps`, the axis is published anyway and marked
    `regular: false`, and the client must then read the store's own `time`
    array instead of reconstructing.
    """
    if times.size < 2:
        return {"frames": int(times.size), "step_seconds": 0, "regular": True,
                "gaps": None, "start": _iso(times[0]), "end": _iso(times[-1])}

    steps = np.diff(times)
    step = steps[0]
    gaps = [
        {"after": _iso(times[k]), "gap_hours": float(steps[k] / np.timedelta64(1, "h"))}
        for k in np.flatnonzero(steps != step)
    ]

    if gaps and not allow_gaps:
        worst = max(gaps, key=lambda g: g["gap_hours"])
        raise ValueError(
            f"time axis is not regular: {len(gaps)} gap(s), the largest "
            f"{worst['gap_hours']:.0f} h after {worst['after']}. The client "
            f"reconstructs timestamps as start + k * step, so publishing this "
            f"would mislabel every frame after the gap rather than fail. Pull "
            f"the missing window, or pass allow_gaps=True to publish the axis "
            f"explicitly."
        )

    return {
        "frames": int(times.size),
        "step_seconds": int(step / np.timedelta64(1, "s")),
        "regular": not gaps,
        "gaps": gaps or None,
        "start": _iso(times[0]),
        "end": _iso(times[-1]),
    }


def write_zarr_tier(ds: xr.Dataset, path: Path, stride: int,
                    *, time_chunk: int = DEFAULT_TIME_CHUNK, level: int = ZSTD_LEVEL,
                    allow_gaps: bool = False) -> dict:
    """Write one downsampled tier as a Zarr v3 store. Returns its description.

    Subsampling is a plain stride, not an average. Averaging would be defensible
    for a climatology and is wrong here: the daily tier exists so somebody can
    scrub a year and see weather go past, and a 24 h mean of a rotating wind
    vector is close to zero -- it would show five years of calm. A stride shows
    a real hour, just fewer of them.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if ds.sizes["time"] == 0:
        raise ValueError("nothing to publish: the dataset has no timesteps")

    # A stride longer than the record is NOT an error -- numpy slicing always
    # keeps the first element, so it yields a single-frame tier, which is the
    # sensible reading of "daily" over an eight-hour archive. Guarding against
    # it would have been an unreachable branch: the only way to get zero frames
    # is to hand this an already-empty dataset, which is what the check above
    # is for.
    tier = ds.isel(time=slice(None, None, stride))

    tier = tier.astype("float32")
    chunks = (min(time_chunk, tier.sizes["time"]), tier.sizes["lat"], tier.sizes["lon"])

    # Re-chunk onto the store's own boundary. open_archive already chunked, but
    # striding for a coarser tier leaves ragged chunks; aligning them with what
    # is written means dask reads, converts and writes one chunk at a time.
    tier = tier.chunk({"time": chunks[0], "lat": -1, "lon": -1})
    encoding = {
        v: {"chunks": chunks, "compressors": [ZstdCodec(level=level)]}
        for v in tier.data_vars
    }

    # Checked BEFORE writing: a gapped axis should cost nothing to discover,
    # not a gigabyte of upload followed by a mislabelled map.
    axis = time_axis_spec(tier["time"].values, allow_gaps=allow_gaps)

    path.parent.mkdir(parents=True, exist_ok=True)
    tier.to_zarr(path, mode="w", zarr_format=3, encoding=encoding, consolidated=False)

    on_disk = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    uncompressed = sum(tier[v].size * 4 for v in tier.data_vars)

    return {
        "path": path.name,
        **axis,
        "chunks": {"time": chunks[0], "lat": chunks[1], "lon": chunks[2]},
        "chunk_bytes_uncompressed": int(np.prod(chunks) * 4),
        "bytes": int(on_disk),
        "compression": f"zstd-{level}",
        "compression_ratio": round(uncompressed / on_disk, 3) if on_disk else None,
        "files": sum(1 for f in path.rglob("*") if f.is_file()),
    }


def verify_tier(path: Path, source: xr.Dataset, var: str) -> dict:
    """Read a cell back out of the published store and compare it to the source.

    The chunking and the 0-360 -> -180..180 conversion are both things that can
    be wrong in a way that still renders -- a mirrored map, an off-by-one frame
    -- so the round trip is checked rather than assumed. D021's verification
    section asks for exactly this.
    """
    published = xr.open_zarr(path, consolidated=False)
    t = published.sizes["time"] // 2
    j = published.sizes["lat"] // 3
    i = published.sizes["lon"] // 3

    cell = published[var].isel(time=t, lat=j, lon=i)
    want = source[var].sel(
        time=cell["time"].values, lat=float(cell["lat"]), lon=float(cell["lon"]),
    )

    got_v, want_v = float(cell.values), float(want.values)
    published.close()
    return {
        "at": {"time": _iso(cell["time"].values),
               "lat": float(cell["lat"]), "lon": float(cell["lon"])},
        "published": got_v,
        "source": want_v,
        "match": bool(np.isclose(got_v, want_v, rtol=0, atol=1e-6)
                      or (np.isnan(got_v) and np.isnan(want_v))),
    }


def export_archive(data: Path, out: Path, product: str = "wind",
                   tiers: dict | None = None, *, level: int = ZSTD_LEVEL,
                   allow_gaps: bool = False, fill_gaps: bool = False) -> dict:
    """Publish every tier of one product, plus the manifest that indexes them."""
    if product not in PRODUCTS:
        raise ValueError(f"unknown product {product!r}; expected one of {list(PRODUCTS)}")
    spec = PRODUCTS[product]
    tiers = spec["tiers"] if tiers is None else tiers

    combined, sources = open_archive(data, product)

    filled = {"inserted": 0, "original": int(combined.sizes["time"])}
    if fill_gaps:
        combined, filled = regularise_time(combined)
        if filled["inserted"]:
            print(f"filled    {filled['inserted']} missing timestep(s) with NaN "
                  f"({filled['inserted'] / (filled['original'] + filled['inserted']):.3%} "
                  f"of the axis) -- the source really is missing them")

    display = to_display_grid(combined)

    lat, lon = display["lat"].values, display["lon"].values
    written = {}
    for name, stride in tiers.items():
        path = out / f"{product}_{name}.zarr"
        written[name] = write_zarr_tier(
            display, path, stride, level=level, allow_gaps=allow_gaps,
            time_chunk=TIME_CHUNK.get(product, DEFAULT_TIME_CHUNK))
        written[name]["verified"] = verify_tier(path, display, spec["vars"][0])

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "product": product,
        "label": spec["label"],
        "variables": list(spec["vars"]),
        "units": "m/s",
        "longitude_convention": "-180..180 (display; D020 stores 0-360)",
        "bbox": [float(lat[0]), float(lon[0]), float(lat[-1]), float(lon[-1])],
        "grid": {
            "lat0": float(lat[0]), "dlat": float(np.diff(lat)[0]), "nlat": int(lat.size),
            "lon0": float(lon[0]), "dlon": float(np.diff(lon)[0]), "nlon": int(lon.size),
        },
        "tiers": written,
        "provenance": {
            "source_files": sources,
            "source_frames": int(combined.sizes["time"]),
            "script": "sar.viz.archive",
            # Said out loud in the manifest, so a reader of the published store
            # knows some frames are NaN by construction rather than by accident.
            "gaps_filled_with_nan": filled["inserted"],
            "frames_from_source": filled["original"],
        },
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / f"{product}_archive.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    combined.close()
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(
        description="Publish the raw archive as multi-resolution Zarr for the frontend.")
    p.add_argument("--data", required=True, help="archive root: /home/26p67/data")
    # No default: this writes gigabytes, and a relative default is how raw data
    # ends up inside the synced vault. Same rule as the fetchers.
    p.add_argument("--out", required=True, help="where the .zarr stores go. No default.")
    p.add_argument("--product", default="wind", choices=sorted(PRODUCTS))
    p.add_argument("--tier", action="append",
                   help="publish only this tier; repeatable. Default: all of them.")
    p.add_argument("--level", type=int, default=ZSTD_LEVEL, help="zstd level")
    p.add_argument("--fill-gaps", action="store_true",
                   help="insert the source's missing timesteps as NaN so the axis is "
                        "regular; nothing is interpolated")
    p.add_argument("--allow-gaps", action="store_true",
                   help="publish an irregular time axis; the client must then read the "
                        "store's time array rather than reconstructing it")
    args = p.parse_args()

    spec = PRODUCTS[args.product]
    if args.tier:
        unknown = [t for t in args.tier if t not in spec["tiers"]]
        if unknown:
            p.error(f"unknown tier(s) {unknown} for {args.product}; "
                    f"expected {list(spec['tiers'])}")
        tiers = {t: spec["tiers"][t] for t in args.tier}
    else:
        tiers = None

    m = export_archive(Path(args.data), Path(args.out), args.product, tiers,
                       level=args.level, allow_gaps=args.allow_gaps,
                       fill_gaps=args.fill_gaps)

    print(f"published {args.product} from {len(m['provenance']['source_files'])} file(s), "
          f"{m['provenance']['source_frames']} source frames")
    print(f"grid      {m['grid']['nlat']} lat x {m['grid']['nlon']} lon   bbox {m['bbox']}")
    total = 0
    for name, t in m["tiers"].items():
        total += t["bytes"]
        ok = "verified" if t["verified"]["match"] else "MISMATCH"
        axis = "regular" if t["regular"] else f"IRREGULAR ({len(t['gaps'])} gap(s))"
        print(f"  {name:10s} {t['frames']:7d} frames  {t['bytes'] / 1e6:8.1f} MB  "
              f"{t['compression_ratio']:.2f}x  {t['files']:5d} files  {ok}  {axis}")
    print(f"total     {total / 1e6:.1f} MB")
    print(f"manifest  {args.out}/{args.product}_archive.json")


if __name__ == "__main__":
    main()
