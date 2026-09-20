"""
export.py -- write a window of forcing as a bundle the browser can read.

    python -m sar.viz.export --data C:/maritime-data \
        --start 2021-01-01 --end 2021-01-03 --out frontend/public/data

Writes, into `--out`:

    manifest.json   what is in the bundle: clock, layers, grids, provenance
    wind.f32        [time][lat][lon][u, v] as raw little-endian float32

WHY RAW FLOAT32 AND NOT JSON. Measured 2026-09-16: one hour of the 77 x 77 box
is 46.3 KB as float32 and a 48 h window is 2.17 MB, against roughly 4 MB of
text for the same numbers. The browser turns the binary into an array in one
call -- `new Float32Array(await res.arrayBuffer())` -- instead of walking a JSON
tree. Half the bytes and none of the parse.

The grid is regular, so coordinates are NOT shipped. The manifest carries
`lat0/dlat/nlat` and the client reconstructs them, which is another 47 KB saved
per layer and removes any chance of the axes disagreeing with the data.

THIS IS THE PRESENTATION BOUNDARY. D020 stores longitude 0-360 everywhere and
converts only on the way out; GeoJSON (RFC 7946) and every web map require
-180..180. So `to_display_longitude` is called here, in the manifest's grid and
bbox, and nowhere downstream of here.

THE MANIFEST DESCRIBES LAYERS GENERICALLY ON PURPOSE. Each entry carries its own
`type` and its own `grid`, because the probability map (D007) will arrive on a
200-500 m grid inside a box that follows the ensemble -- nothing like the 0.25
deg wind grid -- and the search tracks arrive as geometry with no grid at all.
A renderer that assumed one grid would have to be rewritten; one that reads the
grid per layer does not. Four types are planned: `field`, `raster`, `points`,
`track`. Only `field` is written today.

Closes nothing on its own. Feeds D021's frontend and, later, R8c.
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sar.utils.data_io import open_forcing, select_window
from sar.utils.geo import regular_axis_step, to_display_longitude

# Layer types the frontend knows how to draw. Listed here so the exporter and
# the client cannot drift apart silently -- an unknown type is a bug, not a
# layer the client quietly skips.
LAYER_TYPES = ("field", "raster", "points", "track")

# Wind is hourly, current 3-hourly; both are regular, so a step in seconds plus
# a count describes the axis exactly and the client needs no time array.
VARS_BY_PRODUCT = {"wind": ("u10", "v10"), "current": ("water_u", "water_v")}


def _git_sha(repo: Path) -> str | None:
    """The commit the bundle was built from, so a published map is traceable.

    Returns None outside a checkout rather than raising: the exporter must still
    run from a tarball or on the cluster.
    """
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def grid_spec(lat: np.ndarray, lon_store: np.ndarray) -> dict:
    """Describe a regular grid by its origin and step, in DISPLAY longitude.

    Raises on an irregular axis. The client reconstructs coordinates by
    `lat0 + i * dlat`, which is simply wrong if the spacing varies, and a
    silently wrong map is the failure mode this project keeps meeting.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.sort(to_display_longitude(np.asarray(lon_store, dtype=float)))

    # One fitted step per axis, and a refusal if the axis does not fit one. The
    # judgement is `sar.utils.geo.regular_axis_step` so that the client's
    # `lat0 + k * dlat` and the engine's bilinear weights agree on what regular means.
    # It replaced a comparison of consecutive gaps at atol=1e-9, which rejected the real
    # HYCOM archive: its coordinates are float32, so a true 0.08 degree longitude grid
    # arrives with gaps from 0.079956 to 0.080018 and no amount of it is an error.
    dlat = regular_axis_step(lat, "lat")
    dlon = regular_axis_step(lon, "lon")

    return {
        "lat0": float(lat[0]), "dlat": dlat, "nlat": int(lat.size),
        "lon0": float(lon[0]), "dlon": dlon, "nlon": int(lon.size),
    }


def write_field(ds, u_name: str, v_name: str, path: Path) -> dict:
    """Write [time][lat][lon][u, v] as raw float32 and return its encoding.

    Longitude is re-sorted into display order here so the client can index the
    buffer directly against the grid in the manifest. Doing it in the browser
    instead would put the one conversion D020 allows in two places.
    """
    lon_display = to_display_longitude(ds.lon.values)
    order = np.argsort(lon_display)

    u = np.asarray(ds[u_name].values, dtype=np.float32)[:, :, order]
    v = np.asarray(ds[v_name].values, dtype=np.float32)[:, :, order]

    # Interleave so one cell's u and v are adjacent: the client reads a vector
    # with one offset rather than two reads a whole plane apart.
    stacked = np.stack([u, v], axis=-1).astype("<f4", copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stacked.tobytes(order="C"))

    return {
        "dtype": "float32",
        "endian": "little",
        "components": 2,
        "order": ["u", "v"],
        "shape": ["time", "lat", "lon", "component"],
        "bytes": path.stat().st_size,
    }


def export_window(data: Path, start: str, end: str, out: Path,
                  product: str = "wind") -> dict:
    """Export one time window of one product as a bundle. Returns the manifest.

    `start` inclusive, `end` exclusive, as everywhere else in this project.
    """
    if product not in VARS_BY_PRODUCT:
        raise ValueError(f"unknown product {product!r}; expected one of {list(VARS_BY_PRODUCT)}")
    u_name, v_name = VARS_BY_PRODUCT[product]

    prefix = "era5_" if product == "wind" else "hycom_"
    files = sorted((data / "raw").glob(f"{prefix}*.nc"))
    if not files:
        raise FileNotFoundError(f"no {prefix}*.nc in {data / 'raw'}")

    # Try each file rather than parsing filenames for dates: the file that
    # covers the window is the one whose time axis contains it, and a name is
    # not evidence of contents.
    ds = None
    for f in files:
        candidate = open_forcing(f)
        try:
            ds = select_window(candidate, start, end)
            source = f
            break
        except ValueError:
            candidate.close()
    if ds is None:
        raise ValueError(
            f"no {prefix}*.nc covers [{start}, {end}) -- checked {len(files)} file(s)"
        )

    out.mkdir(parents=True, exist_ok=True)
    field_path = out / f"{product}.f32"
    encoding = write_field(ds, u_name, v_name, field_path)
    grid = grid_spec(ds.lat.values, ds.lon.values)

    times = ds.time.values
    step = int((times[1] - times[0]) / np.timedelta64(1, "s")) if times.size > 1 else 0
    speed = np.hypot(ds[u_name].values, ds[v_name].values)

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bbox": [grid["lat0"], grid["lon0"],
                 grid["lat0"] + grid["dlat"] * (grid["nlat"] - 1),
                 grid["lon0"] + grid["dlon"] * (grid["nlon"] - 1)],
        "clock": {
            "start": str(times[0])[:19] + "Z",
            "end": end + "T00:00:00Z" if len(end) == 10 else end,
            "step_seconds": step,
            "frames": int(times.size),
        },
        "layers": [{
            "id": product,
            "type": "field",
            "label": "10 m wind" if product == "wind" else "surface current",
            "source": field_path.name,
            "units": "m/s",
            "encoding": encoding,
            "grid": grid,
            # The client scales its colour ramp from this rather than reading
            # the whole buffer to find a maximum before it can draw anything.
            "value_range": [0.0, float(np.nanpercentile(speed, 99.5))],
            "nan_fraction": float(np.isnan(speed).mean()),
        }],
        "provenance": {
            "source_file": source.name,
            "arrival_convention": ds.attrs.get("sar_arrival_convention"),
            "script": "sar.viz.export",
            "git_sha": _git_sha(Path(__file__).resolve().parents[3]),
        },
    }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ds.close()
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Export a forcing window for the frontend.")
    p.add_argument("--data", required=True, help="C:/maritime-data or /home/26p67/data")
    p.add_argument("--start", required=True, help="inclusive, YYYY-MM-DD")
    p.add_argument("--end", required=True, help="EXCLUSIVE, YYYY-MM-DD")
    p.add_argument("--out", default="frontend/public/data")
    p.add_argument("--product", default="wind", choices=sorted(VARS_BY_PRODUCT))
    args = p.parse_args()

    m = export_window(Path(args.data), args.start, args.end, Path(args.out), args.product)
    layer = m["layers"][0]
    total = sum((Path(args.out) / f).stat().st_size
                for f in (layer["source"], "manifest.json"))

    print(f"wrote     {args.out}")
    print(f"source    {m['provenance']['source_file']} ({m['provenance']['arrival_convention']})")
    print(f"clock     {m['clock']['start']} .. {m['clock']['end']}  "
          f"{m['clock']['frames']} frames at {m['clock']['step_seconds']} s")
    print(f"grid      {layer['grid']['nlat']} lat x {layer['grid']['nlon']} lon")
    print(f"bbox      {m['bbox']}")
    print(f"bundle    {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
