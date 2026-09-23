# The published archive (`sar.viz`)

What the website reads, how it is laid out, and why each choice was made.
Companion to `docs/current-fetch.md`, which covers getting the data in the
first place. The architecture decision and its rejected alternatives are in the
vault at *D021 Visualisation and analysis architecture*.

**Live at** `https://huggingface.co/datasets/AdityaRugs/MaritimeSARoperations`

---

## Two tiers, and they do not compete

```
  CLUSTER  /home/26p67/data/raw/          ANALYSIS TIER — stays put
  era5_*.nc    2.97 GB  (5 years)         the engine, hindcast validation,
  hycom_*.nc  12.9  GB  (5 years)         clustering: all read these directly
  gdp_hourly_*.csv  109 MB
                 │
                 │  sar.viz.archive  /  sar.viz.drifters
                 ▼
  /home/26p67/data/published/             PUBLICATION TIER — a derived copy
                 │
                 │  scripts/publish_archive.py
                 ▼
  HUGGING FACE  ◄──── the browser fetches only the chunks it needs
```

**Nothing here moves the archive.** This is a downsampled, reformatted copy for
publication; the NetCDF remains the thing analysis reads. Confusing the two is
how a "derived" file quietly becomes the source of truth.

### Why Hugging Face and not Cloudflare R2

R2's free tier is real — 10 GB, 1 M Class A and 10 M Class B operations, zero
egress — and our operation count is nowhere near it. It was rejected because it
**requires a linked payment card**, and Cloudflare's billing policy places
preauthorisation holds against that card for usage-based services; a failed
hold makes buckets return errors, with deletion after 30 days. A site that must
stay up through assessment should not carry a billing failure mode.

Hugging Face needs no card. The honest caveat is that its free public storage
is documented as *"best-effort"* with no guaranteed figure. Measured: the wind
archive is 1.60 GB across 2,238 files, inside the "first few gigabytes" they
describe as unremarkable and well inside their limits of 100k files per repo
and 10k per folder.

**CORS verified 2026-09-18**, which was the last open risk: HF reflects the
request origin, the header survives the 307 redirect to the resolve-cache, and
`Accept-Ranges` is exposed. `zarrita.js` reads it from GitHub Pages.

---

## Gridded fields — Zarr (`sar.viz.archive`)

```
wind_archive.json        the manifest: grid, bbox, tiers, provenance
wind_hourly.zarr         43,824 frames   1,273.7 MB   1.63x
wind_6-hourly.zarr        7,304 frames     257.5 MB   1.34x
wind_daily.zarr           1,826 frames      71.8 MB   1.21x
```

### Why Zarr and not the NetCDF itself

NetCDF is one self-describing file you open with a library. To read a single
cell from a 600 MB file a browser must fetch the 600 MB; there is no
partial-read story that works in a browser without a server in front of it.

Zarr is the same array split into many small files, so fetching "January 2021
over the box" is a handful of ordinary HTTP GETs for exactly the chunks that
cover it. ERA5's own upstream source is Zarr and `zarr==3.3.0` was already
pinned, so this runs with the grain of the project.

### Compression: zstd level 19

Measured on 9.11 MB of real float32 wind, not chosen from documentation:

| Codec | Size | Ratio | Write speed |
|---|---|---|---|
| none | 9.11 MB | 1.00x | — |
| zstd-3 | 7.20 MB | 1.26x | 161 MB/s |
| zstd-9 | 6.58 MB | 1.39x | 106 MB/s |
| **zstd-19** | **5.78 MB** | **1.58x** | 20 MB/s — **chosen** |
| blosc + zstd + shuffle | 7.11 MB | 1.28x | **rejected** |

Blosc with a byte shuffle is the standard recommendation for float arrays, and
it is rejected for a reason unrelated to its ratio: **`zarrita.js` cannot decode
it.** A ratio the client cannot read is 1.00x. Plain zstd is a registered codec
in the Zarr v3 core spec, and it was verified against the real store — zarrita
fetches and decodes one of our chunks in **45 ms**, returning values identical
to Python to the last bit.

zstd-19 costs 1.7 minutes for the 2.03 GB hourly tier, once, and zstd decodes
at roughly the same speed whatever level wrote it, so the browser pays nothing
for the extra ratio.

### Chunking: 48 timesteps × the whole spatial box

About 1.11 MB per chunk, at every tier. Large enough to amortise per-request
overhead, small enough to fetch without a stall. 48 steps is exactly the 48 h
scenario window at hourly cadence — the unit D009 already organises around — so
a scenario is one chunk rather than a straddle.

Chunked whole in space rather than tiled because the box is small (77 × 77 for
wind) and every layer draws all of it at once, so a spatial tile would only
ever be fetched alongside its neighbours.

### Tiers are strided, not averaged

A 24 h mean of a rotating wind vector is close to zero. An averaged daily tier
would show five years of calm. A stride shows a real hour, just fewer of them.
There is a test pinning this.

**Coarser tiers compress *worse* per frame** — 29.5 → 36.1 → 39.7 KB — which is
the opposite of the intuition: striding throws away the temporal correlation
zstd was exploiting. One tier's cost cannot be scaled from another's ratio.

### The time axis, and the guard on it

The client reconstructs every frame's timestamp as `start + k × step`, exactly
as it reconstructs coordinates from `lat0 + j × dlat`. That is only valid on a
regular axis, and **a gapped axis does not fail, it mislabels**.

Measured: eight days of January plus two of March concatenate to 264 frames
with one 1,225-hour hole. Reconstructed, the last frame lands on 2021-01-11
instead of 2021-03-03 — **51 days wrong, on a map that renders perfectly**.

So `write_zarr_tier` **refuses** an irregular axis before writing anything,
naming the largest gap. `--allow-gaps` publishes anyway with `"regular": false`,
and the client refuses a tier marked that way rather than mislabelling it.
`publish_archive.py` checks again at upload, because the two steps can run days
apart and what reaches the browser is what matters.

---

## Trajectories — Parquet (`sar.viz.drifters`)

```
drifters_archive.json        manifest and counts
drifters.parquet             926,533 observations   20.3 MB
drifter_segments.parquet     825 segments           34 kB
```

**Parquet, not Zarr.** These are irregular positions at irregular times, one
row per observation per buoy — not a gridded array, and forcing them into one
throws away the thing that makes a trajectory a trajectory. Parquet is also
what Hugging Face's dataset viewer reads.

**The cleaned frame is published, not the raw CSV.** `load_drifters` applies
every rule `fetch/drifters.py` established: the ERDDAP units row wedged under
the header, longitude to 0–360, SST Kelvin to Celsius masked against fill
values, `undrogued` per **observation** rather than per buoy, and `segment_id`
split on gaps over 3 h. Republishing the raw file would hand every reader the
same four traps.

`drifter_segments.parquet` is one row per contiguous in-box run — the unit R2
validates against, and the answer to "how many independent validation cases do
we have". **825 segments, 640 lasting at least 48 h.**

> **A correction lives here.** D018 cited 642. That counts *observations*, and
> 48 hourly observations span 47 hours; two segments sit in that gap and cannot
> verify a 48 h forecast. For D018's own stated purpose the number is **640**.
> The manifest key says `segments_over_48h_duration` so the criterion is in the
> name.

### Reading it

```python
import pandas as pd
B = "https://huggingface.co/datasets/AdityaRugs/MaritimeSARoperations/resolve/main"
segs = pd.read_parquet(f"{B}/drifter_segments.parquet")
obs  = pd.read_parquet(f"{B}/drifters.parquet")
```

```python
import xarray as xr
ds = xr.open_zarr(f"{B}/wind_daily.zarr", consolidated=False)
```

## Drifter tracks for the map — one file per buoy (`sar.viz.drifters --tracks`)

The site's drifter layer (issue #50) cannot read Parquet, and it never needs every
buoy at once: it shows one buoy's path and the few buoys in the water at the current
moment. So the tracks are exported as JSON, **one small file per buoy**, beside a
single index:

```
python -m sar.viz.drifters --tracks --data C:/maritime-data --out C:/maritime-data/published
```

It reads both GDP products and their buoy tables (see [drifter-fetch.md](drifter-fetch.md))
and the validation split (`derived/validation_split.csv`, [validation-split.md](validation-split.md)),
so run `python -m sar.validate.split` first.

| File | What it holds | Measured 2026-09-23 |
|---|---|---|
| `drifter_index.json` | one row per buoy: first and last fix, where it was first seen, drogue summary (`drogued` / `undrogued` / `mixed` / `uncertain`), its validation splits and a `sealed` flag | 326 buoys, **89 kB** |
| `drifter_tracks/<ID>.json` | that buoy's every fix, as columns: whole hours since `t0`, lat and lon to 4 dp (11 m), segment number, and 0/1 flags for undrogued and "more than 3 h from a real fix" | **26.9 MB** in all; median **36 kB**, largest **520 kB** |

The index loads once. A track file is fetched only when its buoy is picked or is in
the water at the current moment, so a click costs one small download and the path
keeps its full hourly resolution. The export takes about 5 s.

**Sealed buoys stay visible.** Looking at a raw track is fine. What vault D025 rules out
is comparing the *engine* against a sealed track before the frozen evaluation run, so the
map badges sealed buoys rather than hiding them.

**Where the site looks for it.** `VITE_DRIFTER_BASE`, which defaults to
`VITE_DATA_BASE`, so once the files are published beside the archive nothing needs setting.
Until then, the layer is simply not offered: a missing index is not an error.

---

## Conventions, which are the same everywhere

| | |
|---|---|
| Longitude | **−180…180** in everything published here |
| Latitude | ascending |
| Arrays | `(time, lat, lon)`, `float32`, m/s |
| Time | UTC, start inclusive, **end exclusive** |

D020 stores 0–360 internally and converts **once**, at the presentation
boundary. `sar.viz.archive` and `sar.viz.drifters` *are* that boundary; nothing
downstream of them converts again. Every web map and GeoJSON (RFC 7946) require
−180…180, so a store in 0–360 would push the conversion into the client, where
it would be done twice or not at all.

---

## Republishing

```bash
# gridded fields, on a compute node — 13 min for five years of wind
sbatch scripts/publish_wind_archive.sbatch

# trajectories — about a minute
python -m sar.viz.drifters --data /home/26p67/data --out /home/26p67/data/published

# upload, on a compute node — 1 h 29 m for 1.6 GB across 2,239 files
sbatch scripts/upload_archive.sbatch
```

`publish_archive.py` takes **no `--token` flag**, on purpose: a token on a
command line lands in shell history, `ps` output, CI logs and any transcript of
the session. Authenticate once, out of band, with `huggingface-cli login`. See
the vault's *Credentials — what we hold and how each one is protected*.

---

## Reading it from the frontend

`frontend/src/sources.js` holds `ZarrSource`. Two things about it are worth
knowing before changing it:

- **The async seam is `ensure(frame)` and it is deliberately narrow.** Reading a
  value stays synchronous, because the render loop, the renderer and the click
  chart all want a number rather than a promise. What is async is making a
  frame *resident*, which happens once per chunk boundary, not once per read.
- **Reading a non-resident frame throws.** Returning zeros would draw a calm,
  plausible, wrong map.

`pickTier(tiers, spanDays)` swaps resolution by how much time is on screen —
1.32 GB hourly against 0.07 GB daily for the same five years — preferring the
next *coarser* tier when the wanted one is absent, because that is the
direction that keeps the download bounded.
