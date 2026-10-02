# The real forcing (`sar.pipeline.gridded`)

Closes #88. Every 60 s step, `DriftPipeline` asks its forcing one question,
`sample(lats, lons, time)`: the current and the wind at every particle, as two (N, 2)
arrays in m/s. `ConstantForcing` answers it with one invented value. `GriddedForcing`
answers it from the stored HYCOM and ERA5 files, for every particle at once.

## Running it

```
python -m sar.pipeline.gridded --forcing-dir C:/maritime-data --time 2021-01-05T07:30 \
    --particles 100000 --seed 1 --check 1000
python -m sar.pipeline.track --forcing-dir C:/maritime-data --start 2021-01-05T01:00 \
    --lat 26.5 --lon -79.8 --duration 3600 --timestep 60
python -m sar.pipeline.ensemble --forcing-dir /home/26p67/data --lat 26.5 --lon -79.8 \
    --datum-sigma-km 0 --start 2019-01-31T12:00 --timestep 60 --duration 48h \
    --particles 10000 --sigma 0.1 --seed 20260930 --save-every 15m --out /home/26p67/data
sbatch scripts/bench_ensemble_step.sbatch --forcing-dir /home/26p67/data \
    --time 2019-06-01T06:00 --particles 1 10000 100000 1000000
```

The first samples random points in the box and prints how many are at sea, beached or
outside, the mean speeds and the time per call; `--check K` also compares K all-sea
points with `sar.model.interpolate.sample_field`. `--forcing-dir` on the track and ensemble
commands replaces `--constant-current`; exactly one of the two is required.

```python
from sar.pipeline.gridded import GriddedForcing
from sar.pipeline.track import DriftPipeline

with GriddedForcing.from_dir("C:/maritime-data", "2021-01-05T01:00", "2021-01-05T02:00") as forcing:
    current, wind = forcing.sample(lats, lons, "2021-01-05T01:30")   # each (N, 2), m/s
    states = list(DriftPipeline(forcing, 60.0).track("2021-01-05T01:00", 26.5, 280.2, 3600))
```

`from_dir(root, start, end)` finds the `raw/hycom_<box>_*.nc` and `raw/era5_<box>_*.nc` a
run needs, opens only those, and refuses the window before anything runs if the files do
not cover it or it crosses a gap D011 does not allow.

## What is returned

| | Meaning | Unit |
|---|---|---|
| `current` | HYCOM `water_u`, `water_v` at 0 m: eastward, northward | m/s |
| `wind` | ERA5 `u10`, `v10`, the 10 m wind: eastward, northward, the direction it blows *toward* | m/s |
| NaN current, finite wind | the particle is **beached**: its nearest HYCOM point is land | |
| NaN current and wind | the particle is **outside** the box; `outside(lats, lons)` says which | |

Arrays are new on every call. Longitude may be given as −180 to 180 or 0 to 360.

## The maths

**In space, bilinear.** A particle sits in a grid cell with four stored points at its
corners. Interpolate in a straight line along the bottom edge, then along the top edge, then
between those two:

$$u = (1-f_y)(1-f_x)\,u_{SW} + (1-f_y)f_x\,u_{SE} + f_y(1-f_x)\,u_{NW} + f_y f_x\,u_{NE}$$

where $f_x$ and $f_y$ are how far across the cell the particle is, 0 to 1. The four weights
always sum to 1. It is the same interpolant `sample_field` (#10) uses, and it is exact for a
field linear in latitude and longitude, which is the test that pins it. Components are
interpolated separately, never speed and direction (`docs/resultant-vector.md` says why).

**In time, linear** between the two stored steps either side:
$u = u_k + f_t\,(u_{k+1} - u_k)$ with $f_t = (t - t_k)/(t_{k+1} - t_k)$, computed from integer
nanoseconds so it is exact.

**Worked example.** At 26.53 N, 281.41 E: (26.53 − 17)/0.04 = 238.25, so row 238 and
$f_y$ = 0.25; (281.41 − 278)/0.08 = 42.625, so column 42 and $f_x$ = 0.625.

| Corner | Weight | u (m/s) | Weight × u |
|---|---|---|---|
| SW | 0.75 × 0.375 = 0.28125 | 1.20 | 0.3375 |
| SE | 0.75 × 0.625 = 0.46875 | 1.40 | 0.65625 |
| NW | 0.25 × 0.375 = 0.09375 | 1.00 | 0.09375 |
| NE | 0.25 × 0.625 = 0.15625 | 1.30 | 0.203125 |
| | 1 | | **1.2906** |

## Land

HYCOM has no value on land. The rule (Aditya, 29 Sep 2026; D016):

1. **A particle whose nearest HYCOM point is land has entered a land cell.** Its current
   is NaN and `DriftPipeline` freezes it with its mass.
2. **Otherwise only the wet corners are used**, their weights renormalised. With NE land in
   the example, the wet weights sum to 0.84375 and u = 1.0875 / 0.84375 = 1.2889. The
   nearest corner is at most half a cell away on each axis, so its weight is at least 0.25:
   this never divides by zero.
3. **The diagonal corner counts only through a wet edge neighbour.** HYCOM is a C-grid:
   water crosses cell edges, never corners. Without this, water on the far side of a
   one-point-thick barrier leaks into the average: at $f_x = f_y$ = 0.45 with +1 on this side
   and −1 beyond, plain renormalising gives 0.198 instead of 1.

Beaching on "any land corner" instead would stop particles up to a cell early: 2.2 % of sea
cells box-wide touch land, and 4.9 % around Florida and the Bahamas, where the Gulf Stream
runs (measured 29 Sep 2026 on the January 2021 file).

**What this leaves out.** Near the coast, sea velocities are carried right up to land
instead of tapering to zero. HYCOM's coastline is only as fine as its ~8 km grid, and D016's
GSHHG cross-check is still owed.

## Gaps, files and the box

- **Gaps (D011).** HYCOM is 3-hourly with a 6 h gap in January 2019 and a 12 h gap on
  2020-10-25 (cluster, 29 Sep 2026). 6 h is blended across and listed in `describe()`;
  longer is refused with `ForcingGapError`, both by `require(start, end)` for a whole window,
  since D011 excludes whole scenarios, and by `sample` as a backstop. ERA5 is hourly with no
  gaps.
- **Files.** Chosen by the dates in their names with a day's slack, because one older wind
  file's name gives an inclusive end, and then only the time axes inside the files are
  trusted. Overlapping files are refused by name. The files are never concatenated:
  `xr.concat` of lazily opened files loads them all into memory (measured, about 6.6 GB for
  the HYCOM archive). Each stays open lazily and one 2-D slice is read at a time.
- **The box** is where both grids are. HYCOM's last longitude is 296.96, not 297; ERA5 reaches
  297. A particle in that strip is outside, not beached.

## Resolution

| Field | Time step | Longitude step | Latitude step | At 26.5 N |
|---|---|---|---|---|
| Current (HYCOM GLBy0.08, expt 93.0, 0 m) | 3 h | 0.08° | 0.04° | 7.96 × 4.45 km |
| Wind (ERA5, 10 m) | 1 h | 0.25° | 0.25° | 24.9 × 27.8 km |

## Where the data come from

- **HYCOM** is an ocean model that assimilates observations (satellite altimetry and
  sea-surface temperature, floats, ships): a best estimate of the ocean, not a measurement.
  It resolves features of ~35–45 km and more (four to five grid cells), so interpolating
  between its points adds no detail it does not have.
- **ERA5** is ECMWF's reanalysis: a weather model run over the past with observations
  folded in. Its 10 m wind is smooth at ~30 km.

Both are models. The error they carry, 9 to 26 km of position a day
(`docs/position-update.md`), dwarfs anything the interpolation adds, which is why this module
matches `sample_field` rather than trying to be cleverer than the data.

## Accuracy against `sample_field`

On synthetic grids the two agree to 1e-9 m/s. On the real files they differ slightly,
because `sample_field` measures from the stored grid lines, which HYCOM keeps in float32, up
to 3.4e-5° off a regular grid; this module uses the regular grid, which the browser rebuilds
too. The difference is bounded by (that departure / the step) × the spread of the corners,
and `--check` reports both:

| File | Points | Current: largest difference | Bound | Wind |
|---|---|---|---|---|
| January 2021, laptop, 30 Sep 2026 | 500 | 0.00025 m/s | 0.00043 m/s | 2e-15 m/s |
| 1 June 2019, cluster (job 58738), 30 Sep 2026 | 1,000 | 0.00011 m/s | 0.00037 m/s | 2e-15 m/s |

## Cost

Where the work goes: the table of [u now, v now, u next, v next] for every grid point is
built only when time crosses a stored step (every 180 steps for HYCOM, 60 for ERA5); each
step then costs four lookups per particle and one blend. A slice takes 3.4 ms to read for
HYCOM and 1.0 ms for ERA5 (laptop), about 0.04 ms per step averaged.

| Where | N = 1 | 10⁴ | 10⁵ | 10⁶ |
|---|---|---|---|---|
| Compute node jaguar11 (i7-3770), job 58737, 30 Sep 2026 | 0.45 ms | 2.8 ms | 33 ms | **371 ms** |
| Laptop, 30 Sep 2026 | 0.14 ms | 2.1 ms | 42 ms | — |
| The 24 Sep stand-in on a node, for comparison | — | 3.1 ms | 29 ms | 364 ms |

**At scale the real backend costs what the budget assumed.** With the pipeline's own 112 ms, a
step at 10⁶ particles is 483 ms: 1,390 s, about 23 minutes, for a 48 h run on one node, against
1,334 s estimated from the stand-in on 24 Sep. A 48 h run at 10⁴ particles took 33 s including
the CSV (job 58738).

**At N = 1 it is 0.45 ms, not the ~50 µs hoped for.** That is Python overhead, around sixty
small NumPy calls per step on a 2012 CPU. The σ calibration (#89) calls it for one particle
about 38 million times, which would be about 5 hours; #89 should run many windows as one
ensemble, or add a one-particle fast path, rather than call this one particle at a time.

For comparison, `sample_field` in a loop costs 2.58 ms per point: 41 hours of sampling for a
48 h run at 10⁴ particles.

## First runs on the real forcing (cluster, 30 Sep 2026)

Smoke tests, not results: σ is a trial value until #89 calibrates it.

- **Across a file boundary** (job 58738). 10⁴ particles from 26.5 N 79.8 W at
  2019-01-31T12:00, datum spread 0 (D026), σ = 0.1, 48 h, bracketing January's last slice and
  February's first. The cloud drifted **85 km north and 36 km west** (0.54 m/s mean) up the
  Florida coast with the Florida Current; none beached. Its spread at 48 h was **153 m along
  the current and 33 m across**, against σ√t = 42 m per axis from the kicks alone: the
  current's shear stretched it about fourfold along-stream, which uniform forcing cannot do.
- **Across the 12 h gap.** The same command from 2020-10-24T12:00 was refused before it ran:
  "current has a 12 h gap from 2020-10-25T00:00 to 2020-10-25T12:00 ... D011 excludes a
  scenario that crosses it".

## Guards

- A time outside the files raises `OutOfCoverageError`; across a gap longer than D011 allows,
  `ForcingGapError`. The track and ensemble commands turn both into usage errors.
- No file for the window raises `FileNotFoundError`; overlapping files, files on different
  grids, times out of order, a descending axis and an end before the start raise
  `ValueError`. A NaT time raises.

## Not here

- **Telling beached from out of domain inside `Ensemble`.** Its `beached` flag marks any
  particle with NaN forcing, so it marks both; `outside()` gives it what it needs (D016).
- **The GSHHG coastline cross-check** (D016).
- **σ.** Calibrated in #89 at 54.4 m s⁻¹ᐟ²: `docs/sigma-calibration.md`.
