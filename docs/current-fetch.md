# Surface current fetch (`sar.fetch.current`)

Closes GitHub issue #4. Retrieves surface current data from HYCOM
`GLBy0.08/expt_93.0`, either as a single nearest-neighbour point or as a
box+period pull for offline analysis.

## User guide

### `water_u` and `water_v`

- `water_u`: **eastward** surface current velocity, m/s. Positive means
  flowing east, negative means flowing west.
- `water_v`: **northward** surface current velocity, m/s. Positive means
  flowing north, negative means flowing south.
- Both are the surface value (depth 0 m), not depth-averaged; see
  [Why not `_sur`](#why-not-_sur) below for why that distinction matters.
- There is no hard limit enforced by this code; values are whatever HYCOM
  reports. As a sanity check: most of the D014 box sits well under
  ±0.5 m/s, while the Gulf Stream core can reach **1.5-2.5 m/s**. A value
  far outside that range at a non-Stream location is worth double-checking
  rather than trusting outright.
- `NaN` in either field means a masked cell, almost always land. See
  `scripts/find_current_gaps.py` below to pull all of those out of a
  fetched file at once.

### What times can be chosen

HYCOM only publishes a value **every 3 hours, UTC**: `00:00, 03:00, 06:00,
09:00, 12:00, 15:00, 18:00, 21:00`. There is no 11:00 record, no 07:30
record, and so on.

`fetch_current`/the CLI does **not** reject an off-grid `--time`; it
silently snaps to the *nearest* of those eight steps instead. For example,
`--time 11:00` is 2 hours from `09:00` and 1 hour from `12:00`, so it
returns the `12:00` record. **Always read the `time` field of the result**,
since it is the timestamp actually used, and can be up to 1.5 hours away
from what was requested.

The same applies to the *date*: the dataset's real coverage runs from
**2018-12-04 12:00 UTC to 2024-09-05 09:00 UTC** (measured against the live
server by Aditya, not the advertised range). A `--date`/`--time` outside
that window does not raise an error; nearest-neighbour has no concept of
"out of coverage" and will still return the closest timestamp the store
actually has, however far away that turns out to be. There is currently no
guard against this here (see [Known gaps](#known-gaps--open-questions)).

### The "nearest point" behaviour (space)

`--lat`/`--lon` are resolved the same way as time: **nearest grid point**,
not exact interpolation. "GLBy**0.08**" in the dataset name is the grid
spacing: **0.08° in both latitude and longitude** (confirmed against the
real coordinate arrays: consecutive longitude values are `278.000000,
278.080017, 278.160034, ...`). So the actual point used can be up to
~0.04° away from what was requested in each axis.

**How far apart is that in metres?** A degree of latitude is close to a
constant ~111.3 km everywhere; a degree of longitude shrinks toward the
poles by a factor of `cos(latitude)`, so the east-west spacing is not the
same at the top of the D014 box as at the bottom:

| | spacing in degrees | spacing in metres |
|---|---|---|
| North-south (any latitude) | 0.08° | **~8,900 m** (~8.9 km) |
| East-west at 17°N (south edge of box) | 0.08° | **~8,520 m** (~8.5 km) |
| East-west at ~26.5°N (mid-box) | 0.08° | **~7,970 m** (~8.0 km) |
| East-west at 36°N (north edge of box) | 0.08° | **~7,210 m** (~7.2 km) |

So a nearest-neighbour lookup can be up to roughly **4-4.5 km off** in
either direction from the point actually requested, which is worth keeping
in mind next to any current feature (like the Gulf Stream) that is itself
only tens of km wide.

Longitude is handled for you: pass ordinary `-180..180` values (e.g.
`-70.0` for 70°W) and `to_store_longitude` converts it to the store's
`0..360` convention internally; there is no need to do that conversion by
hand.

One current limitation worth knowing: `fetch_current`'s result only
reports the matched `time`, not the matched `lat`/`lon`, so there is no
built-in way to see exactly which grid cell was used in space, only in
time. If you need that, open the dataset yourself with
`open_current_dataset()` and inspect the selection before `.load()`ing it.

### How this data is actually produced

HYCOM `GLBy0.08/expt_93.0` is **not** a set of raw sensor readings at each
grid point; real current meters and drifting buoys are far too sparse to
cover a global 0.08° grid every 3 hours. It is the output of a numerical
ocean **model** (the HYbrid Coordinate Ocean Model) run by the US Navy,
which is periodically corrected ("data-assimilated") against whatever real
observations exist at that time: satellite sea-surface-height altimetry,
satellite and buoy sea-surface temperature, Argo float profiles, moored
buoys, and ship-based measurements. The assimilation system (NCODA) nudges
the model's internal ocean state toward those observations, and the
model's own physics (conservation of momentum, mass, and the effect of
wind and density on flow) fills in the rest of the grid in between, so
that every cell, including ones nowhere near an actual sensor, gets a
physically consistent value at every timestep.

Practically, this means `water_u`/`water_v` at any one grid point is a
**model estimate constrained by nearby real data**, not a direct
measurement. It will smooth over current features narrower than the model
can resolve, and its accuracy depends on how much real observational data
was available to correct it at that place and time, which is exactly what
the validation work in D018 (GDP drifter comparison) is for.

## Why not `_sur`

The `_sur` catalogue looks like the obvious source but does not contain
`water_u`/`water_v` at all: it holds **barotropic** velocity (depth-averaged
over the entire water column, ~5000 m), which is a small fraction of the
Gulf Stream's actual surface flow (D019 in the project vault). Using it would
quietly under-drive every particle in the drift model while still producing
plausible-looking tracks.

This module instead opens the full 3-D dataset and selects **depth level 0**,
which is verified against the live server (not assumed from documentation) to
be exactly 0 m:

| | |
|---|---|
| OPeNDAP endpoint | `https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0` |
| Variables | `water_u`, `water_v` (m/s, eastward/northward) |
| Depth | level 0 = 0 m (40 levels total, asserted at open time) |
| Cadence | **3-hourly**, not hourly |
| Latitude | ascending (`-80` to `90`), the opposite of ERA5 |
| Longitude | store convention is **0-360**, not -180-180 |

## `src/sar/fetch/current.py`

- `open_current_dataset()`: opens the dataset lazily (nothing downloaded
  yet) and asserts `depth.values[0] == 0.0` before any caller can rely on
  that assumption. Drops the `tau` coordinate, whose units (`"hours since
  analysis"`) are not a parseable reference date and otherwise crash
  xarray's CF time decoder before `water_u`/`water_v` are ever touched.
- `to_store_longitude(lon)`: converts a -180..180 longitude to the store's
  0..360 convention (`lon % 360`).
- `fetch_current(date, time, lat, lon)`: nearest-neighbour lookup in time,
  latitude and longitude. Returns `{"water_u": float, "water_v": float,
  "time": np.datetime64}`. The returned `time` is whichever 3-hourly step
  was actually selected, which will not equal the requested time unless it
  happens to land exactly on one.
- `fetch_current_box(start, end, lat_bounds, lon_bounds)`: loads every
  record (all lat/lon points at depth 0) for every 3-hourly timestep in
  `[start, end)`. Used by `scripts/fetch_current_range.py` and by
  `write_current_netcdf` below; asserts the latitude, longitude and time
  slices are non-empty rather than silently returning an empty dataset.
- `estimate_bytes(start, end)`: what a gridded pull over `[start, end)` will
  cost on disk, from the measured 885 KB per timestep. See the size table
  below; this is what the `--force` guard is computed from.
- `fetch_current_range(start, end, lat_bounds, lon_bounds, *, chunk_days=2)`:
  `fetch_current_box` over a long window, split into requests the server will
  actually serve, then concatenated. **Use this, not `fetch_current_box`, for
  anything longer than a couple of days** — see "The request-size ceiling"
  below. Prints progress per piece and refuses a result containing duplicate
  timestamps, which is what a mistaken piece boundary would produce.
- `write_current_netcdf(start, end, out, *, force=False)`: the **raw tier**
  of D020. Pulls the D014 box over `[start, end)` and writes it as gridded
  NetCDF to `<out>/raw/`, returning the path. Normalises to the D020
  convention and asserts it before writing, checks the land mask survived,
  stamps `units = "m/s"`, and refuses a range whose estimate exceeds 3 GB
  unless `force=True`.

  **It skips a window whose file already exists**, printing `skip` and
  returning the path. That makes a bulk pull resumable: throw the Slurm array
  at the problem repeatedly and it only fetches what is missing. Against an
  endpoint this erratic that matters more than getting the sizing right first
  time. `--force` re-fetches.
- `LAT_S, LAT_N, LON_W, LON_E`: the D014 study box (17-36 N, 82-63 W).
  Change these once, here, and nowhere else, matching the convention in the
  vault's `code/fetch_wind_arco.py`.

### CLI

Two modes. Point mode prints the resulting dict to stdout:

```
python -m sar.fetch.current --date 2019-01-01 --time 12:00 --lat 25.5 --lon -70.0
```

Box mode writes the raw archive tier and prints the path and its size:

```
python -m sar.fetch.current --start 2021-01-05 --end 2021-01-08 --out /home/26p67/data
# -> /home/26p67/data/raw/hycom_17-36N_82-63W_20210105-20210108.nc
```

Mixing the two sets of flags is an error rather than a precedence rule.

**`--out` has no default.** Every fetch script in this repo that defaulted it
to a relative `data` has written raw NetCDF into the synced OneDrive vault at
least once. On the cluster it is `/home/26p67/data` — **not** `/data1/26p67`
or `/data/26p67`, which are empty root-owned mount points with nothing mounted
on them (D017, measured 2026-09-14).

### Output: the raw tier

| | |
|---|---|
| Path | `<out>/raw/hycom_17-36N_82-63W_<start>-<end>.nc` |
| Dimensions | `(time, lat, lon)` — depth is *selected* at level 0, not left as a length-1 axis |
| Variables | `water_u`, `water_v`, both `units = "m/s"` |
| Longitude | 0-360, ascending (D020) |
| Latitude | ascending (D020) |
| `time` | 3-hourly, **start inclusive, end exclusive** — a 3-day request is 24 steps |

### Resolution, in metres as well as degrees

The grid is **0.08° longitude × 0.04° latitude**, which is **not** 1/12° and
**not** square — a figure that was wrong in three vault notes and in issue #7.
At the middle of the study box (26.5 N):

| Axis | Degrees | Metres |
|---|---|---|
| longitude | 0.08 | **8.07 km** (0.08 × 111.32 × cos 26.5°) |
| latitude | 0.04 | **4.45 km** |

### The request-size ceiling, and why it is not the same as throughput

**Measured 2026-09-18, by failing twice.** This is the single most useful thing
on this page if you are about to pull in bulk.

`tds.hycom.org` will not serve an arbitrarily large single request, and the
limit is *separate from* how fast it serves. A half-year asked for in one call
is 1,448 timesteps, and it times out after roughly 36 minutes:

```
500  java.net.SocketTimeoutException: Read timed out;
     water_u -- 8981:10428,0:0,2425:2900,3475:3712
```

Six Slurm array tasks died that way having written nothing.

**A throughput measurement taken beforehand did not predict this and could not
have.** 4.0 s per timestep, measured over a 4-day request, is a correct number
— but it describes the *rate*, and what failed was the *size of one request*.
Those are independent limits and only one of them had been measured.

| Request | Timesteps | Per variable | Result |
|---|---|---|---|
| 4 days | 32 | 14.5 MB | **works** — 128 s |
| 8 days | 64 | 29 MB | times out; xarray retries, so it lands eventually and burns the wall clock |
| half-year | 1,448 | 660 MB | times out, task dies |

`REQUEST_DAYS = 2` (16 timesteps, 7.3 MB per variable) — **half the largest
size seen to work**. That margin is deliberate: throughput here varies by
almost an order of magnitude between runs (4.0 s per timestep measured clean,
35 s on the very first 3-day pull), so a size that only just works on a good
day will not survive a bad one.

**The archive itself is complete.** A scan of 1,600 individual days found zero
unreadable timesteps, which is what establishes that the fault was entirely in
how it was being asked for rather than in the data.

Two consequences worth carrying:

- Any throughput figure quoted in the report must say **what request size it
  was measured at**, because the two do not compose.
- The five-year archive is about **640 requests**, not ten. Bulk pulls run as
  `scripts/pull_current_years.sbatch`: one month per array task, sixty tasks,
  three at a time, resumable.

### Size, and why there is a guard

One timestep over the box is **885 KB** — 476 lat × 238 lon cells × 2 variables ×
float32. That is **nineteen times** a wind timestep, which is the whole reason
this path refuses large ranges by default.

| Range | Timesteps | In memory (float32) | **On disk (int16)** |
|---|---|---|---|
| 3 days | 24 | ~21 MB | ~11 MB |
| 1 month | ~248 | ~220 MB | **112 MB** (measured) |
| 1 year | 2,920 | ~2.6 GB | ~1.3 GB |
| 5 years | 14,608 | 12.9 GB | **~6.6 GB** |

**The on-disk figure is half the in-memory one, and the difference is real.**
Measured 2026-09-18 on a written month: 112 MB for 247 timesteps = 443 KB each.
HYCOM serves `water_u`/`water_v` packed as **int16 with
`scale_factor = 0.001`**; xarray decodes to float32 on read and re-applies that
encoding on write, so the stored archive is int16 at 1 mm/s resolution — far
finer than the data is accurate to.

So **the five-year archive is about 6.6 GB, not the 12.9 GB quoted in D019,
D021 and issue #12.** `estimate_bytes` still uses the float32 figure and so
over-estimates by 2×; that is left deliberately, because what it guards is what
is held **in memory** during the pull, and that really is float32.

The threshold is **3 GB**, which passes one year and refuses five. In practice
the bulk pull uses **one month per task** — about 220 MB, comfortably under the
guard, so no task needs `--force`. Sizing the piece below the guard is the
point; switching the guard off would not be.

## `scripts/fetch_current_range.py`

Pulls every surface-current record in the D014 box across a date range and
writes it to `data/current/`.

```
python scripts/fetch_current_range.py --start 2019-01-01 --end 2019-01-03 \
    [--out data/current] [--format txt|csv|parquet]
```

- `--start` is inclusive, `--end` is exclusive, the same convention as
  `fetch_wind_arco.py`.
- Output columns: `time, lat, lon, water_u, water_v`, one row per grid
  point per timestep.
- Output path: `data/current/current_<start>_<end>.<ext>`. Defaults to
  `.txt` (whitespace-separated) if `--format` is not given.

**Size warning.** A single 2-day pull over the full box already produced
~160 MB of text (24 timesteps by 476 lat by 238 lon grid points). A multi-month or
multi-year range at this resolution will be very large; prefer
`--format parquet` and/or a short window, per Aditya's handoff note that a
full five-year, all-depths pull would be ~28 GB (this fetch is depth-0 only,
but the spatial grid is unchanged).

## Tests

`tests/fetch/test_current.py`: monkeypatches `open_current_dataset` (or
`xr.open_dataset` underneath it) with a small synthetic dataset shaped like
the real store, so tests never touch the network. Covers longitude
conversion, the depth-0 assertion, nearest-neighbour selection (both exact
and off-grid), and the box query's empty-slice guards.

Run with:

```
python -m pytest tests/fetch/test_current.py -q
```

## `scripts/find_current_gaps.py`

Pulls every NaN record (either `water_u` or `water_v` missing; HYCOM land
cells and any other masked value both decode to NaN) out of an already-
fetched `current_<period>` file, and writes a report for gap analysis and
mitigation decisions (e.g. feeding the vault's D011 "missing timestep
handling" question).

```
python scripts/find_current_gaps.py data/current/current_2019-01-01_2019-01-03.txt \
    [--out data/current]
```

- Reads `.txt`, `.csv` or `.parquet`, whichever `fetch_current_range.py`
  produced.
- Writes `<out>/gaps_<input stem>.txt` (default `<out>` = same directory as
  the input) containing: total/NaN record counts and percentage, the number
  of distinct `(lat, lon)` locations and timesteps affected, a per-timestep
  NaN count, and then every NaN row in full.
- On the 2019-01-01 to 2019-01-03 sample box pull, **11.8% of records were
  NaN** (321,720 of 2,718,912), almost all land cells inside the D014 box,
  since the box spans a lot of the Bahamas/Florida/East Coast coastline.

Tested in `tests/pipeline/test_find_current_gaps.py` against small
synthetic inputs, including a round-trip through the exact `.txt` format
`fetch_current_range.py` writes.

**Note on the `.txt` format.** The original `.txt` writer put the timestamp
as `"2019-01-01 00:00:00"`, a raw space inside a whitespace-delimited
format. Reading it back with `sep=r"\s+"` silently split the date and time
into two fields, swallowing the date into an implicit pandas index and
leaving only the time-of-day in the `time` column. Fixed by writing
timestamps as `"2019-01-01T00:00:00"` (no embedded space) before the
`.txt` write; `.csv` and `.parquet` were never affected, since they don't
rely on whitespace as the field separator.

## Known gaps / open questions

- No retry or timeout handling around the OPeNDAP request: a slow or
  dropped connection currently just hangs or raises whatever `xarray`/
  `netCDF4` raises natively.
- `fetch_current`'s nearest-neighbour selection can silently return a value
  far from the requested point/time if the request falls well outside the
  store's real coverage (2018-12-04 to 2024-09-05, per Aditya's handoff);
  there is no coverage guard here yet, unlike `fetch_wind_arco.py`'s
  `STUDY_START`/`STUDY_END` check.
- Land cells and any other masked HYCOM values decode to `NaN`. See
  `scripts/find_current_gaps.py` for pulling those out of an already-fetched
  file for review.
