# Resultant vector method (`sar.model.interpolate`)

Closes #10. Gives the current and the wind at any position and time, by interpolating the
four surrounding grid points, and says how far the answer can be trusted.

Run it:

```
python -m sar.model.interpolate --current data/raw/hycom_....nc --wind data/raw/era5_....nc \
    --lat 26.53 --lon 281.4 --time 2021-01-05T07:30 [--with-wind | --no-wind] [--plot cell.png]

python -m sar.model.interpolate --diagram          # the explanatory figure, no data needed
python scripts/plot_resultant_diagram.py           # the same figure, with the other figures
```

This module reads forcing files; it never fetches them. The paths are real files the
fetchers wrote, and the position and time asked for must fall inside the box and window
those files cover, or the run stops with `OutOfCoverageError`.

### What counts as a forcing file

`open_forcing` takes either shape this project writes, chosen by the suffix:

| Suffix | Shape | Written by |
|---|---|---|
| `.nc`, `.nc4` | gridded NetCDF | `sar.fetch.current --out`, `sar.fetch.wind` |
| `.txt`, `.csv`, `.parquet` | a tidy table, one row per (time, lat, lon) | `scripts/fetch_current_range.py` |

The table form matters because it is the default of the current range fetcher, and so the
form the currents on hand are in. `data/current/current_2019-01-01_2019-01-03.txt` is
2,718,912 rows covering a 476 by 238 grid and 24 three-hourly steps, and
`open_forcing_table` pivots it back onto that grid in a few seconds. A record absent from
the table becomes NaN, which is what HYCOM means by land; 11.8 % of that file is already
NaN for exactly that reason. A table that is not a complete rectangle on a regular grid is
refused rather than pivoted into something misleading.

A table is a poor way to store a grid. Prefer `.parquet` to `.txt` for anything large,
and gridded NetCDF to either.

### What has to be installed

The JSON path needs numpy, pandas and xarray. `--plot` and `--diagram` additionally need
matplotlib, which is imported only when something is actually drawn, so the rest runs in
an environment without it.

matplotlib is declared in `pyproject.toml`, but a virtual environment created before it
was added will not have it, and neither will it have scipy. If drawing stops with a note
about matplotlib, bring the environment up to the declared set:

```
pip install -e .        # from the repository root, inside the venv
```

### Running it with no archive

```
python -m sar.model.interpolate --diagram        # the method drawn, no files at all
python scripts/make_demo_forcing.py --out data   # a small invented pair, about 57 KB
```

`make_demo_forcing.py` writes files with the right shape: both grids' real steps, the real
variable names and units, the D020 conventions, and a wedge of NaN where land would be, so
the `MissingCornerError` path can be tried too. Every value in them is invented and means
nothing; they are for checking that the code runs and that the JSON and the cell diagram
look right, never for judging a current or a wind. They land in the gitignored `data/` tree.

## User guide

### What is returned

`resultant_at(...)` returns a dictionary that serialises to JSON:

| Key | Meaning | Unit |
|---|---|---|
| `lat`, `lon` | the position asked for, longitude in the stored 0 to 360 convention | degrees |
| `time` | the time asked for | ISO 8601 |
| `current.u`, `current.v` | eastward and northward components of the interpolated current | m/s |
| `current.speed` | length of that vector | m/s |
| `current.direction_to_deg` | compass bearing the current flows toward (0 north, 90 east); `null` if the speed is zero | degrees |
| `current.uncertainty` | see "Quantifying the error" | mixed |
| `wind.*` | the same five keys for the 10 m wind (`u10`, `v10`); absent when wind is excluded | m/s |

Wind bearings are the direction the wind blows **toward**. Meteorologists quote the direction
it blows **from**; add 180 degrees and wrap to 0 to 360 to convert.

The wind is returned as the plain 10 m wind. It is **not** scaled to a drift velocity. A
floating object moves at a small percentage of the 10 m wind: the drift velocity is the
current plus alpha times the wind, and this project fixes alpha at 0.02, the midpoint of
the 1 to 4 percent range (D002 and D018, `ALPHA_MID` in `src/sar/fetch/wind.py` and
`ALPHA` in `frontend/src/drift.js`). Adding the raw vectors would therefore overstate the
wind term about fifty times. The integrator applies alpha.

### Inputs and what happens at the edges

| Situation | Behaviour |
|---|---|
| Longitude given as -180 to 180 | converted to 0 to 360 with `to_store_longitude` |
| Position outside the grid | `OutOfCoverageError`, naming the axis and its valid range |
| Time outside the file | `OutOfCoverageError` |
| Position exactly on a grid point or line | returns the grid value; the far-side weights are zero |
| Time exactly on a time step | the later time slice is not read |
| Any corner missing (land) | `MissingCornerError`, saying whether all 4 or k of 4 are missing |
| Cell crossing the 0 and 360 seam | not supported; the study box (278 to 297) never does |

### Resolution

| Field | Time step | Longitude step | Latitude step | At 26.5 N |
|---|---|---|---|---|
| Current (HYCOM GLBy0.08) | 3 hours | 0.08 deg | 0.04 deg | 8.07 km by 4.45 km |
| Wind (ERA5) | 1 hour | 0.25 deg | 0.25 deg | 24.9 km by 27.8 km |

The HYCOM cell is not square. `docs/current-fetch.md` records the measurement; every
function here takes the step from the grid axes rather than assuming a value.

### What counts as a regular grid

Every weight below assumes the grid is regular, and so do two other things: the
`lat0 + k * dlat` the browser reconstructs from `sar.viz.export`'s manifest, and the pivot
in `open_forcing_table`. All three ask `sar.utils.geo.regular_axis_step`, so they cannot
disagree about it.

The judgement is made against a **fitted line**, not against neighbouring gaps. Those look
equivalent and are not, because both archives store coordinates as float32. Measured on
2026-09-19 against `current_2019-01-01_2019-01-03.txt`, the real longitude axis has gaps
running 0.079956 to 0.080018 while being a perfectly regular 0.08 grid; its largest
departure from the fitted line is 5.5e-05 degrees, which is 0.07 % of a step and about 5 m
on the ground. A pairwise test rejects that file outright.

The tolerance is therefore taken from what float32 can represent at the axis's own
magnitude, which is the actual cause: eight units in the last place, capped at a twentieth
of a step so a coarse axis cannot hide a real irregularity. A stretched or curvilinear grid
departs by a sizeable fraction of a step and still fails by orders of magnitude.

## How the four points are found

The method uses the nearest grid point and one direction sign per axis. No distance search
and no haversine formula are involved. All quantities are in degrees.

| Symbol | Meaning | How it is calculated |
|---|---|---|
| `dlat`, `dlon` | grid step | `axis[1] - axis[0]`, after checking the axis is evenly spaced (`grid_step`) |
| `i`, `j` | row and column of the nearest grid point | `round((value - axis[0]) / step)` (`nearest_index`) |
| `dy`, `dx` | signed offset from that point to the particle | `lat - lat_axis[i]`, `lon - lon_axis[j]`; each within half a step |
| `sy`, `sx` | which side of the nearest point the particle is on | `+1` if the offset is zero or positive, else `-1` (`neighbour_direction`) |
| `fy`, `fx` | the offset as a fraction of a step | `abs(dy) / dlat`, `abs(dx) / dlon`; always 0 to 0.5 |

The four corners are the nearest point plus a step in x, a step in y, and both:

```python
offsets = [(0, 0), (0, sx), (sy, 0), (sy, sx)]      # (row, column) offsets
corners = [(i + di, j + dj) for di, dj in offsets]
```

Because the nearest point is at most half a step away on each axis, the particle always
lies between it and its neighbour in the direction of the sign, so the four corners bracket
the particle. If the particle is inside the grid that neighbour always exists, with one
exception: a particle exactly on the first or last grid line has an offset of zero, and a
sign of `+1` on the last line would point off the grid. `keep_inside` flips such a sign
inward, which changes nothing because that far corner has weight zero. Otherwise the
neighbour falls off the grid only when the particle is itself outside it.

## The physics of the resultant

### A vector at a grid point

Each grid point stores $(u, v)$ in m/s, eastward and northward. Speed and bearing are derived:

$$s = \sqrt{u^2 + v^2}, \qquad \theta_{to} = \mathrm{atan2}(u,\, v) \bmod 360^\circ$$

The bearing is clockwise from north. The argument order is `atan2(u, v)`, not `atan2(v, u)`.
To reverse it, $u = s\sin\theta_{to}$ and $v = s\cos\theta_{to}$. The meteorological "from"
direction is $(\theta_{to} + 180^\circ) \bmod 360^\circ$.

Speed and bearing are for **reporting only**: the JSON, the map label, a person reading the
answer. Nothing downstream computes with them. The drift equation takes the components,
for the reason set out in "What the drift equation takes" below.

### The weights

With $f_x$ and $f_y$ as defined above, the four weights are

$$w_{near} = (1-f_x)(1-f_y), \quad w_x = f_x(1-f_y), \quad w_y = (1-f_x)f_y, \quad w_{diag} = f_x f_y$$

Each is the area of the sub-rectangle opposite its corner. The ratios $f_x$ and $f_y$ are
each taken along one axis of a regular grid, so the $\cos\varphi$ factor that shortens a
degree of longitude cancels. Metres appear only when reporting error in real units.

### The resultant

$$u_p = \sum_{i=1}^{4} w_i u_i, \qquad v_p = \sum_{i=1}^{4} w_i v_i$$

Speed and bearing are derived from $(u_p, v_p)$ afterwards.

### Why not average speed and direction

- **Averaging angles fails at the 0 and 360 wrap.** Bearings of 350 and 10 degrees average
  arithmetically to 180 degrees (due south), although both point almost north. The vector
  sum gives $u = \sin 350^\circ + \sin 10^\circ = 0$ and
  $v = \cos 350^\circ + \cos 10^\circ = 1.97$, so the bearing is 0 degrees, which is correct.
- **Averaging speeds ignores cancellation.** With two corners of weight 0.5, one flowing
  east at 1 m/s and one flowing north at 1 m/s, averaging speeds gives 1.00 m/s. Averaging
  components gives $(0.5, 0.5)$, so 0.707 m/s toward 045 degrees, which is 29 percent
  slower. Two opposite 1 m/s corners cancel to exactly zero.
- **The lost speed is physical.** If the current curves or reverses inside the cell, the
  true mean over the cell really is smaller than the mean of the four speeds. The coherence
  number below measures that gap.

### Time

Space is interpolated first, then time, with one fraction between the two bracketing time
steps. Bilinear and linear-in-time interpolation are both linear, so doing time first
gives an identical answer; a test asserts this. `time_blend` is the single-instant form of
`interpolate_series` from #3: the piece at index `m` of `pieces` equals `time_blend` with
fraction `m / pieces`.

### What the drift equation takes, and why it is not speed

The integrator (D009) advances a particle with

$$\vec x_{k+1} = \vec x_k + \big(\vec v_c + \alpha\,\vec v_{w10}\big)\,\Delta t + \sigma\sqrt{\Delta t}\,\vec Z_k$$

Every term there is a **vector**. A particle's position has two components, so moving it
needs a velocity with two components; a speed on its own says how fast but not which way,
and cannot advance anything. This module therefore hands the integrator exactly what it
consumes, $\vec v_c = (u_p, v_p)$ for the current and $\vec v_{w10}$ for the wind, and the
integrator forms $\vec v_c + \alpha\,\vec v_{w10}$ component by component.

Reducing either input to a speed would repeat, one level up, the error the section above
warns about. The drift is a vector sum, and the length of a sum is not the sum of the
lengths unless the vectors happen to point the same way. With the real current from
`current_2019-01-01_2019-01-03.txt` at 27.173 N, 70.369 W on 2 January 2019 at 07:30,
$(u, v) = (-0.047, -0.096)$ m/s, which is 0.107 m/s toward 206 degrees, and a 10 m/s wind
blowing toward the north, so $\alpha\,\vec v_{w10} = (0, 0.2)$ m/s:

| | Result | Distance in one hour |
|---|---|---|
| Vectors added | $(-0.047, 0.104)$, so 0.114 m/s toward 336 degrees | 410 m, north north west |
| Speeds added | $0.107 + 0.2 = 0.307$ m/s, toward nowhere in particular | 1,105 m |

The speed sum is 2.7 times too fast and has no direction at all, because the current and
the leeway partly oppose each other and only the vector sum knows it.

The rest of this section's theory carries over unchanged, for one reason: the drift
equation is **linear** in the two velocities. Interpolating each field and then combining
them gives the same answer as combining at the corners and then interpolating, in the
same way that space and time can be interpolated in either order. So interpolating the
current and the wind separately, as this module does, loses nothing, and the weights,
the component rule and the uncertainty all apply to the inputs the integrator receives.

### The flat-cell approximation

The Earth is curved, and yes, inside one grid cell this method treats the surface as
flat. It does so in two places, and both are small enough to be exact in practice:

- **The cell is treated as a rectangle.** Lines of longitude converge towards the pole, so
  a real cell is very slightly narrower along its northern edge than its southern one. The
  relative difference is $\tan\varphi\,\Delta\varphi$ (latitude step in radians). At the
  box's middle, 26.5 N, that is 0.035 % for a HYCOM cell and 0.22 % for an ERA5 cell; at
  the northern edge, 36 N, it is 0.05 % and 0.32 %.
- **The four corner vectors are added as if their norths were parallel.** Each corner's
  $(u, v)$ is measured against its own local east and north, and on a sphere two points
  $\Delta\lambda$ apart in longitude have norths that differ by $\Delta\lambda\sin\varphi$.
  At 26.5 N that is 0.036 degrees across a HYCOM cell and 0.11 degrees across an ERA5
  cell, at most 0.15 degrees anywhere in the box. The corners' own directions typically
  differ by several degrees, so this is lost well below the signal.

Both follow from the cells being small: 8 km and 28 km on a planet of radius 6,371 km. A
flat tangent plane is the standard assumption for interpolating on a grid this fine.

Where curvature **does** matter is outside this module, in the integrator. Turning a
velocity in m/s into a change of position in degrees is not flat:

$$\Delta\varphi = \frac{v\,\Delta t}{R}, \qquad \Delta\lambda = \frac{u\,\Delta t}{R\cos\varphi}$$

The $\cos\varphi$ is not optional. Without it, east-west movement at 26.5 N would be
understated by 11 %, and the error would change by 18 % between the south and north edges
of the box (sec 36 / sec 17 = 1.18, the same figure `frontend/src/geo.js` quotes for map
distances). The integrator has to apply that factor at the particle's own latitude on
every step. It does, in `sar.model.position`; `docs/position-update.md` derives the step
and measures what the approximation costs.

Rotation of the Earth is not missing either. The Coriolis effect, the geostrophic balance
and the rest of the ocean dynamics are already inside HYCOM's velocities, which is why
D009's equation is kinematic: it moves a particle with a velocity that the ocean model has
already worked out, rather than solving for that velocity itself.

## Quantifying the error

`interpolation_uncertainty(corners, weights, product_sigma=0.0)` returns a dictionary:

1. **Corner scatter.** $\sigma_{spatial} = \sqrt{\sum_i w_i\,|\vec v_i - \vec v_p|^2}$ in
   m/s: the weighted spread of the four corners about the interpolated vector. It is zero
   when all four agree and large when the field changes sharply across the cell, which is
   when bilinear interpolation is least trustworthy.
2. **Coherence.** $R = |\vec v_p| \big/ \sum_i w_i |\vec v_i|$, between 0 and 1. It is 1
   when every corner points the same way and falls as they diverge; this is the number
   that captures the angle problem. `speed_loss_ms` is the same information as a speed
   difference: $\sum_i w_i|\vec v_i| - |\vec v_p|$.
3. **Direction spread.** $\sqrt{-2\ln R}$ radians, reported in degrees: the circular
   standard deviation of the corner directions. Reported as `null` when $R$ is below 0.1,
   because the direction is meaningless when the corners nearly cancel.
4. **Total.** $\sigma_{total} = \sqrt{\sigma_{spatial}^2 + \sigma_{product}^2}$, where
   `product_sigma` is the caller's estimate of the source model's own error (HYCOM or ERA5
   against drifters). That has **not been measured yet**, so the default is 0, not an
   invented figure.

Separately, `bilinear_error_bound(dlon, dlat, d2x, d2y)` returns the classical bound for
a field with known curvature,
$\frac{\Delta\lambda^2}{8}\max|f_{xx}| + \frac{\Delta\varphi^2}{8}\max|f_{yy}|$.
Four corners cannot estimate curvature, so it is
not used at run time. It is a test oracle: on an analytic field with known curvature the
measured interpolation error must stay below it.

**Limitation.** Items 1 to 3 are indicators of variability within the cell, computed from the
corners alone. They are not a guaranteed error bound, and they cannot see structure smaller
than the grid. What is checked is that they behave correctly on fields where the true
answer is known.

## Worked example: one current query, start to finish

Every number below comes from `data/current/current_2019-01-01_2019-01-03.txt`, the real
HYCOM table. Nothing is invented and no step is skipped. The wind is left out entirely;
this is the current only.

### The datum and the four points

Five points appear in this example, colour-coded here and used consistently throughout:

| | Point | Position | Role |
|---|---|---|---|
| $\textcolor{#c62828}{\blacksquare}$ | **$P$, the datum** (red) | 27.173 N, 289.631 E | the position asked for; not a grid point |
| $\textcolor{#1f77b4}{\blacksquare}$ | **$C_{near}$** (blue) | 27.160 N, 289.600 E | nearest grid point |
| $\textcolor{#e07b00}{\blacksquare}$ | **$C_x$** (orange) | 27.160 N, 289.680 E | one step east of it |
| $\textcolor{#2e7d32}{\blacksquare}$ | **$C_y$** (green) | 27.200 N, 289.600 E | one step north of it |
| $\textcolor{#7b3fa0}{\blacksquare}$ | **$C_{diag}$** (purple) | 27.200 N, 289.680 E | one step east and one north |

The datum is the query, not a measurement: nothing is stored at $\textcolor{#c62828}{P}$,
which is the whole reason the other four are needed.

Every number below carries its point's colour, so any value can be traced back to the
corner it came from. Quantities belonging to the datum, which means every intermediate and
final resultant, are in $\textcolor{#c62828}{\text{red}}$; a red number is therefore
always something derived rather than something stored.

The coordinate datum is WGS84, the datum both archives publish on, and no conversion
between reference ellipsoids happens anywhere in this module. "Datum" below always means
the queried point $\textcolor{#c62828}{P}$.

### Step 1: the query, in stored conventions

The position is quoted as 27.173 N, 70.369 W, and the time as 2 January 2019 at 07:30 UTC.
Longitude is converted to the stored 0 to 360 convention by `to_store_longitude`:

$$\lambda = 360^\circ - 70.369^\circ = 289.631^\circ$$

So the datum $\textcolor{#c62828}{P}$ is $(\varphi, \lambda) = (27.173, 289.631)$.

### Step 2: the grid steps

Taken from the axes themselves, not assumed:

$$\varphi_0 = 17.000^\circ, \quad \Delta\varphi = 0.04^\circ, \qquad \lambda_0 = 278.000^\circ, \quad \Delta\lambda = 0.08^\circ$$

The stored axis values are float32, so they print as 289.599976 and 27.200001 rather than
289.60 and 27.20. That is a display artefact of the storage type, not an irregular grid;
`regular_axis_step` accepts the axis against a fitted line for exactly this reason.

### Step 3: the nearest grid point

$$i = \mathrm{round}\!\left(\frac{27.173 - 17.000}{0.04}\right) = \mathrm{round}(254.325) = 254$$
$$j = \mathrm{round}\!\left(\frac{289.631 - 278.000}{0.08}\right) = \mathrm{round}(145.3875) = 145$$

$$\varphi_i = 17.000 + 254(0.04) = 27.160^\circ, \qquad \lambda_j = 278.000 + 145(0.08) = 289.600^\circ$$

That point is $\textcolor{#1f77b4}{C_{near}}$.

### Step 4: offsets, signs and fractions

$$dy = 27.173 - 27.160 = +0.013^\circ, \qquad dx = 289.631 - 289.600 = +0.031^\circ$$

Both are positive, so both signs point up and to the right:

$$s_y = +1, \qquad s_x = +1$$

$$f_y = \frac{|0.013|}{0.04} = 0.325, \qquad f_x = \frac{|0.031|}{0.08} = 0.3875$$

Both lie in 0 to 0.5, as they must: the nearest point is never more than half a step away
on either axis. The datum sits 32.5 % of the way north across the cell and 38.75 % of the
way east, so it is nearer $\textcolor{#1f77b4}{C_{near}}$ than any other corner.

### Step 5: the four corners

Applying the offsets $(0,0), (0,s_x), (s_y,0), (s_y,s_x)$ to $(i,j) = (254,145)$:

| Corner | $(i,j)$ | Latitude | Longitude |
|---|---|---|---|
| $\textcolor{#1f77b4}{C_{near}}$ | $\textcolor{#1f77b4}{(254, 145)}$ | $\textcolor{#1f77b4}{27.160}$ | $\textcolor{#1f77b4}{289.600}$ |
| $\textcolor{#e07b00}{C_x}$ | $\textcolor{#e07b00}{(254, 146)}$ | $\textcolor{#e07b00}{27.160}$ | $\textcolor{#e07b00}{289.680}$ |
| $\textcolor{#2e7d32}{C_y}$ | $\textcolor{#2e7d32}{(255, 145)}$ | $\textcolor{#2e7d32}{27.200}$ | $\textcolor{#2e7d32}{289.600}$ |
| $\textcolor{#7b3fa0}{C_{diag}}$ | $\textcolor{#7b3fa0}{(255, 146)}$ | $\textcolor{#7b3fa0}{27.200}$ | $\textcolor{#7b3fa0}{289.680}$ |

The datum lies inside the rectangle these four span, which is what makes the interpolation
an interpolation rather than an extrapolation.

### Step 6: the weights

$$\textcolor{#1f77b4}{w_{near} = (1 - 0.3875)(1 - 0.325) = (0.6125)(0.675) = 0.413438}$$
$$\textcolor{#e07b00}{w_x = (0.3875)(1 - 0.325) = (0.3875)(0.675) = 0.261562}$$
$$\textcolor{#2e7d32}{w_y = (1 - 0.3875)(0.325) = (0.6125)(0.325) = 0.199063}$$
$$\textcolor{#7b3fa0}{w_{diag} = (0.3875)(0.325) = 0.125937}$$

$$\sum w_i = \textcolor{#1f77b4}{0.413438} + \textcolor{#e07b00}{0.261562} + \textcolor{#2e7d32}{0.199063} + \textcolor{#7b3fa0}{0.125937} = 1.000000$$

The sum being exactly 1 is the check that the four sub-rectangles tile the cell. The
largest weight belongs to $\textcolor{#1f77b4}{C_{near}}$ and the smallest to
$\textcolor{#7b3fa0}{C_{diag}}$, the corner diagonally opposite the datum.

### Step 7: the stored values

The query time 07:30 falls between the 3-hourly steps 06:00 and 09:00, so both slices are
read. Values are $(u, v)$ in m/s:

| Corner | $(u, v)$ at 06:00 | $(u, v)$ at 09:00 |
|---|---|---|
| $\textcolor{#1f77b4}{C_{near}}$ | $\textcolor{#1f77b4}{(-0.039,\, -0.167)}$ | $\textcolor{#1f77b4}{(-0.053,\, -0.053)}$ |
| $\textcolor{#e07b00}{C_x}$ | $\textcolor{#e07b00}{(-0.024,\, -0.126)}$ | $\textcolor{#e07b00}{(-0.051,\, -0.041)}$ |
| $\textcolor{#2e7d32}{C_y}$ | $\textcolor{#2e7d32}{(-0.055,\, -0.159)}$ | $\textcolor{#2e7d32}{(-0.065,\, -0.043)}$ |
| $\textcolor{#7b3fa0}{C_{diag}}$ | $\textcolor{#7b3fa0}{(-0.039,\, -0.113)}$ | $\textcolor{#7b3fa0}{(-0.057,\, -0.026)}$ |

No corner is NaN, so no `MissingCornerError`: this cell is open ocean.

### Step 8: the weighted sum at 06:00

Each component is weighted separately. Eastward first:

| Corner | $w_i$ | $u_i$ | $w_i u_i$ |
|---|---|---|---|
| $\textcolor{#1f77b4}{C_{near}}$ | $\textcolor{#1f77b4}{0.413438}$ | $\textcolor{#1f77b4}{-0.039}$ | $\textcolor{#1f77b4}{-0.016124}$ |
| $\textcolor{#e07b00}{C_x}$ | $\textcolor{#e07b00}{0.261562}$ | $\textcolor{#e07b00}{-0.024}$ | $\textcolor{#e07b00}{-0.006277}$ |
| $\textcolor{#2e7d32}{C_y}$ | $\textcolor{#2e7d32}{0.199063}$ | $\textcolor{#2e7d32}{-0.055}$ | $\textcolor{#2e7d32}{-0.010948}$ |
| $\textcolor{#7b3fa0}{C_{diag}}$ | $\textcolor{#7b3fa0}{0.125937}$ | $\textcolor{#7b3fa0}{-0.039}$ | $\textcolor{#7b3fa0}{-0.004912}$ |
| | | $u_{06}$ | $\textcolor{#c62828}{\mathbf{-0.038262}}$ |

Northward:

| Corner | $w_i$ | $v_i$ | $w_i v_i$ |
|---|---|---|---|
| $\textcolor{#1f77b4}{C_{near}}$ | $\textcolor{#1f77b4}{0.413438}$ | $\textcolor{#1f77b4}{-0.167}$ | $\textcolor{#1f77b4}{-0.069044}$ |
| $\textcolor{#e07b00}{C_x}$ | $\textcolor{#e07b00}{0.261562}$ | $\textcolor{#e07b00}{-0.126}$ | $\textcolor{#e07b00}{-0.032957}$ |
| $\textcolor{#2e7d32}{C_y}$ | $\textcolor{#2e7d32}{0.199063}$ | $\textcolor{#2e7d32}{-0.159}$ | $\textcolor{#2e7d32}{-0.031651}$ |
| $\textcolor{#7b3fa0}{C_{diag}}$ | $\textcolor{#7b3fa0}{0.125937}$ | $\textcolor{#7b3fa0}{-0.113}$ | $\textcolor{#7b3fa0}{-0.014231}$ |
| | | $v_{06}$ | $\textcolor{#c62828}{\mathbf{-0.147883}}$ |

So the spatial resultant at 06:00 is
$\textcolor{#c62828}{(-0.038262,\, -0.147883)}$ m/s.

### Step 9: the weighted sum at 09:00

The same weights, because the position has not changed:

$$u_{09} = \textcolor{#1f77b4}{0.413438(-0.053)} + \textcolor{#e07b00}{0.261562(-0.051)} + \textcolor{#2e7d32}{0.199063(-0.065)} + \textcolor{#7b3fa0}{0.125937(-0.057)}$$
$$= \textcolor{#1f77b4}{-0.021912} \; \textcolor{#e07b00}{-0.013340} \; \textcolor{#2e7d32}{-0.012939} \; \textcolor{#7b3fa0}{-0.007178} = \textcolor{#c62828}{-0.055369}$$

$$v_{09} = \textcolor{#1f77b4}{0.413438(-0.053)} + \textcolor{#e07b00}{0.261562(-0.041)} + \textcolor{#2e7d32}{0.199063(-0.043)} + \textcolor{#7b3fa0}{0.125937(-0.026)}$$
$$= \textcolor{#1f77b4}{-0.021912} \; \textcolor{#e07b00}{-0.010724} \; \textcolor{#2e7d32}{-0.008560} \; \textcolor{#7b3fa0}{-0.003274} = \textcolor{#c62828}{-0.044470}$$

So the spatial resultant at 09:00 is
$\textcolor{#c62828}{(-0.055369,\, -0.044470)}$ m/s.

### Step 10: the time blend

$$f_t = \frac{07{:}30 - 06{:}00}{09{:}00 - 06{:}00} = \frac{90\ \text{min}}{180\ \text{min}} = 0.5$$

$$u_p = (1 - 0.5)\textcolor{#c62828}{(-0.038262)} + (0.5)\textcolor{#c62828}{(-0.055369)} = \textcolor{#c62828}{-0.046815}$$
$$v_p = (1 - 0.5)\textcolor{#c62828}{(-0.147883)} + (0.5)\textcolor{#c62828}{(-0.044470)} = \textcolor{#c62828}{-0.096177}$$

The current at the datum $\textcolor{#c62828}{P}$ is
$(u_p, v_p) = \textcolor{#c62828}{(-0.046815,\, -0.096177)}$ m/s: westward and southward,
so flowing into the third quadrant.

### Step 11: speed and bearing

$$s = \sqrt{\textcolor{#c62828}{(-0.046815)}^2 + \textcolor{#c62828}{(-0.096177)}^2} = \sqrt{0.002192 + 0.009250} = \sqrt{0.011442} = \textcolor{#c62828}{0.106966}\ \text{m/s}$$

$$\theta_{to} = \mathrm{atan2}\big(\textcolor{#c62828}{-0.046815},\, \textcolor{#c62828}{-0.096177}\big) \bmod 360^\circ = -154.045^\circ \bmod 360^\circ = \textcolor{#c62828}{205.955^\circ}$$

Both arguments are negative, which is the third quadrant, so `atan2` returns an angle in
$(-180^\circ, -90^\circ)$ and the modulo carries it into the south-westward part of the
compass. A single-argument $\arctan(u/v) = \arctan(0.4868) = 25.955^\circ$ would have
reported north north east, exactly reversed, because dividing two negatives discards both
signs.

Reversing the formula recovers the components, which is the check that the argument order
is right:

$$u = \textcolor{#c62828}{0.106966}\sin(\textcolor{#c62828}{205.955^\circ}) = 0.106966(-0.43766) = \textcolor{#c62828}{-0.046815} \ \checkmark$$
$$v = \textcolor{#c62828}{0.106966}\cos(\textcolor{#c62828}{205.955^\circ}) = 0.106966(-0.89914) = \textcolor{#c62828}{-0.096177} \ \checkmark$$

Rounded for the JSON: **0.107 m/s toward 206 degrees**. The meteorological "from" bearing
would be $(205.955 + 180) \bmod 360 = 25.955^\circ$, but currents are quoted "toward", so
206 degrees is what is reported.

### Step 12: the uncertainty

The four corners are blended in time first, which is permitted because both operations are
linear:

| Corner | $(u, v)$ at 07:30 | Speed | Bearing |
|---|---|---|---|
| $\textcolor{#1f77b4}{C_{near}}$ | $\textcolor{#1f77b4}{(-0.0460,\, -0.1100)}$ | $\textcolor{#1f77b4}{0.119231}$ | $\textcolor{#1f77b4}{202.69}$ |
| $\textcolor{#e07b00}{C_x}$ | $\textcolor{#e07b00}{(-0.0375,\, -0.0835)}$ | $\textcolor{#e07b00}{0.091534}$ | $\textcolor{#e07b00}{204.18}$ |
| $\textcolor{#2e7d32}{C_y}$ | $\textcolor{#2e7d32}{(-0.0600,\, -0.1010)}$ | $\textcolor{#2e7d32}{0.117478}$ | $\textcolor{#2e7d32}{210.71}$ |
| $\textcolor{#7b3fa0}{C_{diag}}$ | $\textcolor{#7b3fa0}{(-0.0480,\, -0.0695)}$ | $\textcolor{#7b3fa0}{0.084464}$ | $\textcolor{#7b3fa0}{214.63}$ |

**Corner scatter.** Each corner's squared distance from the resultant
$\textcolor{#c62828}{(-0.046815, -0.096177)}$, weighted and summed:

| Corner | $\vec v_i - \vec v_p$ | $\left\lvert \vec v_i - \vec v_p \right\rvert^2$ | $w_i\left\lvert \vec v_i - \vec v_p \right\rvert^2$ |
|---|---|---|---|
| $\textcolor{#1f77b4}{C_{near}}$ | $\textcolor{#1f77b4}{(+0.000815,\, -0.013823)}$ | $\textcolor{#1f77b4}{0.00019174}$ | $\textcolor{#1f77b4}{0.00007927}$ |
| $\textcolor{#e07b00}{C_x}$ | $\textcolor{#e07b00}{(+0.009315,\, +0.012677)}$ | $\textcolor{#e07b00}{0.00024748}$ | $\textcolor{#e07b00}{0.00006473}$ |
| $\textcolor{#2e7d32}{C_y}$ | $\textcolor{#2e7d32}{(-0.013185,\, -0.004823)}$ | $\textcolor{#2e7d32}{0.00019711}$ | $\textcolor{#2e7d32}{0.00003924}$ |
| $\textcolor{#7b3fa0}{C_{diag}}$ | $\textcolor{#7b3fa0}{(-0.001185,\, +0.026677)}$ | $\textcolor{#7b3fa0}{0.00071307}$ | $\textcolor{#7b3fa0}{0.00008980}$ |
| | | sum | $\textcolor{#c62828}{0.00027304}$ |

$$\sigma_{spatial} = \sqrt{\sum_i w_i \left|\vec v_i - \vec v_p\right|^2} = \sqrt{\textcolor{#c62828}{0.00027304}} = \textcolor{#c62828}{0.016524}\ \text{m/s}$$

which is 15.4 % of the speed itself. $\textcolor{#7b3fa0}{C_{diag}}$ is the
furthest from the resultant and so contributes most, although it carries the smallest weight.

**Coherence.** The weighted mean of the corner speeds is

$$\sum_i w_i |\vec v_i| = \textcolor{#1f77b4}{0.049295} + \textcolor{#e07b00}{0.023942} + \textcolor{#2e7d32}{0.023385} + \textcolor{#7b3fa0}{0.010637} = \textcolor{#c62828}{0.107259}\ \text{m/s}$$

$$R = \frac{\textcolor{#c62828}{0.106966}}{\textcolor{#c62828}{0.107259}} = 0.997263$$

$$\text{speed loss} = \textcolor{#c62828}{0.107259} - \textcolor{#c62828}{0.106966} = 0.000294\ \text{m/s}$$

**Direction spread.**

$$\sqrt{-2\ln(0.997263)} = 0.074045\ \text{rad} = 4.24^\circ$$

$R$ is very close to 1 and the spread is only 4.2 degrees, so this cell is coherent: the
four corners point within a few degrees of one another and almost nothing is lost to
cancellation. This is the ordinary case on open ocean, and it is why the right-hand panel
of the diagram nests its arcs at different radii rather than drawing them all at one.

### Step 13: what the wrong method would have given

Averaging the corner speeds and bearings instead of the components:

$$\bar s = \sum_i w_i |\vec v_i| = \textcolor{#c62828}{0.107259}\ \text{m/s}$$

$$\bar\theta = \sum_i w_i \theta_i = \textcolor{#1f77b4}{0.413438(202.69)} + \textcolor{#e07b00}{0.261562(204.18)} + \textcolor{#2e7d32}{0.199063(210.71)} + \textcolor{#7b3fa0}{0.125937(214.63)}$$
$$= \textcolor{#1f77b4}{83.799} + \textcolor{#e07b00}{53.406} + \textcolor{#2e7d32}{41.945} + \textcolor{#7b3fa0}{27.029} = \textcolor{#c62828}{206.180^\circ}$$

The speed is 0.27 % too fast and the bearing 0.22 degrees off, because $R$ is 0.997. The
agreement is a property of this cell, not of the method: it holds only while the corners
nearly align, and the bearing average survives here only because no corner is anywhere
near the 0 and 360 wrap.
The 2.7-times error quoted earlier in "What the drift equation takes" arises once a second
forcing pulls against the first, and no amount of coherence within one field protects
against that.

## Land corners: what happens now and what could be done

Any missing corner raises `MissingCornerError`. This is deliberate for now; a particle
within one cell of the coast cannot be advanced until one of these is chosen:

- **Renormalise over the valid corners.** Drop the missing corners and rescale the
  remaining weights to sum to 1. Simple, but biased: it extrapolates the ocean value onto
  the land side and ignores that the true field goes to zero at the coast.
- **Nearest ocean cell.** Use the value of the closest valid grid point. Continuous
  enough for drift, but discards the interpolation.
- **Treat as beached.** Stop the particle when it enters a cell with a land corner. This is
  a modelling decision for the integrator, not for the interpolator.
- **Partial versus total.** With one corner missing, renormalising is defensible; with
  three or four missing there is nothing to interpolate, so the choice is nearest ocean
  cell or beaching.

## Diagram

`python scripts/plot_resultant_diagram.py` writes
`figures/report/resultant_vector_diagram.png` from synthetic corner values, so it needs no
data. `--plot <path>` on the module's own command line draws the same three panels for a
real query instead, which is the way to see the cell a given position actually landed in.

- **Left, the cell to scale.** One HYCOM cell at 0.08 by 0.04 deg, visibly not square, with
  the particle, the four corner vectors and their weights, and the resultant.
- **Middle, velocity space.** The correct resultant beside the value from averaging speed
  and direction, so the speed lost to cancellation is visible.
- **Right, the bearings.** Lengths are dropped and only the angles remain. Each of the
  four corners gets its own arrow, its own arc swept clockwise from north, and its own
  theta written on that arc; the resultant's arc sits outside all four in red, with the
  corner spread from the uncertainty shaded around it. This is where the
  `direction_to_deg` quoted in the JSON can be read off the picture.

Three things about the right panel are deliberate.

Bearings are clockwise from north while matplotlib measures anticlockwise from east, so
every angle is drawn as `90 - bearing`. Reversing that gives a plausible-looking picture
with every angle mirrored, so a test renders a resultant in each of the four quadrants.

Each corner is drawn at its **own radius**, as a set of nested arcs, rather than all at
one. On real data the four corners usually agree within a few degrees, measured 4.2
degrees apart at 27.173 N, 70.369 W, so arrows and arcs at a common radius collapse into
a single stroke and the panel appears to show nothing at all. Nesting them is what makes
four separate angles visible exactly when the answer is most confident.

Each label sits at a different **fraction along its own arc** rather than all at the
midpoint, which separates the labels by angle as well as by radius, and the key is in the
top left, the one corner the arcs cannot reach until a bearing passes 270 degrees.

## The browser port, and what keeps it honest

The same resultant is available in the frontend, in `frontend/src/interpolate.js`. It is a
port of this module, not a call to it: the published site is a static bundle reading Zarr
from Hugging Face, so there is no server for the browser to ask.

Two copies of one formula drift apart silently, and a map that is subtly wrong still looks
reasonable, so the copies are tied together by a generated fixture:

```
python scripts/export_resultant_golden.py --from data/current/current_2019-01-01_2019-01-03.txt
cd frontend && npm test                        # replays every case through the JavaScript
```

Python is the reference. The script cuts a small window out of a stored forcing file, runs
the engine over it, and records the inputs and the answers: the weights, the uncertainty,
the time blending and the failure modes. `frontend/tests/interpolate.test.js` feeds the
JavaScript the identical inputs and asserts the answers match. Change one side alone and
that test fails, naming the case. Regenerate the fixture only when the Python is what
changed, and read the diff before committing it.

The window is chosen to **contain** land rather than to avoid it, because a coast is where
the browser is most likely to meet a missing corner, and an invented NaN would not prove
the two sides treat a real one alike. Cut without `--from` the grid is synthetic instead,
so the fixture can still be regenerated by someone with no archive; the cases are written
in fractional cell indices so the same set works either way.

Two conventions differ on the browser side, and both are deliberate:

- **Index names are swapped.** `layers.js` uses `j` for latitude and `i` for longitude;
  this module uses `i` for the latitude row and `j` for the longitude column. The port
  follows the frontend so it reads correctly beside its neighbours, and the fixture stores
  cells in frontend order.
- **Longitude is display longitude.** The manifest hands the browser -180 to 180, since
  D020 converts at the presentation boundary, so the port does not wrap to 0 to 360.

In `frontend/src/resultant.js` the port is used in two places:

- `sampleAt(frame, lat, lon)` is **always** bilinear. It is what a click on the map should
  show, and it reports the current and the leeway separately, with the uncertainty. Where
  the current is unavailable it returns `current: null` and says why in words rather than
  throwing, so a coastal click still shows the leeway term.
- The drawn field is behind the flag `BILINEAR_FIELD`, which is **off**. Every arrow would
  become four reads instead of one, and resampling the current onto the 28 km wind grid
  invents detail that picture cannot carry. Switch it on once the archive is published and
  the cost has been measured on a real frame.

## Where the data comes from

Both fields are model output, not sensor readings. HYCOM currents and ERA5 winds are
described in `docs/current-fetch.md` and `src/sar/fetch/wind.py`. Interpolation cannot add
accuracy the grid does not have: a feature smaller than one cell (a coastal jet, a wind
shadow behind an island) is invisible to it, which is why the uncertainty output exists.
