# Advancing a position (`sar.model.position`, `sar.pipeline.track`)

The step that turns a drift velocity into a new position, and the loop that repeats it.
`sar.model.drift` gives the velocity; this is what finally moves the particle.

## The step

The model is discrete in time. One step is D009's Euler-Maruyama update,

$$\vec x_{n+1} = \vec x_n + \vec v_d\,\Delta t + \sigma\sqrt{\Delta t}\,\vec Z_n, \qquad \vec Z_n\sim\mathcal N(0, I)$$

with $\vec x_n$ the position now, $\vec v_d$ the drift velocity from D002, $\Delta t$ the
step and $\sigma$ the diffusivity. All four are supplied by the caller;
`calculate_position` invents nothing and reads nothing.

$\sigma$ defaults to zero in the code, which makes the step deterministic and reduces it to
explicit Euler. A run with $\sigma = 0$ is the mean path, not a sample from the distribution,
and no spread can be read off it. The random walk was calibrated at $\sigma$ = 26.3
m s$^{-1/2}$, matched at 4 h (`docs/sigma-calibration.md` §0, vault D028); the first
calibration, at 24 h, gave 54.4. It is `CALIBRATED_SIGMA`, kept for comparisons.

### The random velocity (ADR005, vault D030, 6 Oct 2026)

**The engine's random term is now a velocity with memory, not a random walk.** Each
particle carries a velocity error $\vec u$ that forgets itself over $T_L$:

$$\vec x_{n+1} = \vec x_n + (\vec v_d + \vec u_n)\,\Delta t, \qquad
\vec u_{n+1} = a\,\vec u_n + \sqrt{1-a^2}\,\sigma_u \vec Z_n, \qquad a = e^{-\Delta t/T_L}$$

with $\vec u_0 \sim \mathcal N(0, \sigma_u^2 I)$, the process's steady state. The update of
$\vec u$ is exact for the Ornstein–Uhlenbeck process, so its variance stays $\sigma_u^2$ at any
step. **$T_L$ = 25.7 h** (`MEMORY_TIME_S`) is fitted to the dev drifters' spread at 1–48 h.
**$\sigma_u$ = 0.226 m/s per axis** (95 % CI 0.217–0.236, `CALIBRATED_SIGMA_U`) is matched at
4 h in the engine. At 60 s, $a$ = 0.99935, so a particle's velocity error 45 minutes on is
97 % the one it had: it drifts almost in a straight line, as a buoy does.

It is still D002's third term, $\eta$, and still integrated by Euler–Maruyama on the pair
$(\vec x, \vec u)$. `calculate_position` is unchanged: the pipeline adds $\vec u$ to the drift
it passes in. The functions are `start_velocity_error` and `evolve_velocity_error`.
`DriftPipeline` takes `sigma` or `sigma_u` (with `memory_time_s`), never both. The random walk
is unchanged bit for bit.

## Metres into degrees

A drift is in metres per second and a position is in degrees, and the conversion between
them is not one number:

$$\Delta\varphi = \frac{v\,\Delta t}{R}, \qquad \Delta\lambda = \frac{u\,\Delta t}{R\cos\varphi}$$

both then multiplied by $180/\pi$. $R$ is `EARTH_RADIUS_M`, 6,371,008.8 m, the GRS80 mean
radius $R_1 = (2a + b)/3$ from Moritz (2000), *Journal of Geodesy* 74(1), 128 to 133. It
is defined once, in `sar.utils.geo`, and every conversion between degrees and metres in the
project derives from it: this step, the grid, the search patterns, the sweep, and `R_EARTH_M`
in `frontend/src/geo.js`, which the search golden fixture holds to it. So the backend's track,
the patterns and the map's ruler cannot disagree about how far a particle moved.

The $\cos\varphi$ is the whole content of the flat-earth approximation. Lines of longitude
converge towards the pole, so a degree of longitude is shorter than a degree of latitude
everywhere except the equator: at 26.5 N, the middle of the D014 box, it is 89.5 percent
as long, and at 36 N it is 80.9 percent. Dropping the factor would understate every
eastward step by 11 percent at the box's middle, and by a different amount at every
latitude in it.

Two consequences look like bugs to a reader who has not met them:

- Latitude is evaluated at the position the step **starts** from, which is what makes this
  first order rather than a midpoint scheme. Section 2 below measures what that costs.
- Near a pole $1/\cos\varphi$ grows without bound. `POLAR_LIMIT_DEG` rejects a position
  past 89 degrees rather than returning that number. It is a project limit chosen for this
  reason, not a published one; the study box reaches 36 N, so nothing here comes near it.

## Conventions

| Quantity | Order | Units |
|---|---|---|
| Position | `[lat, lon]` | degrees, longitude 0 to 360 |
| Drift, current, wind | `[u, v]` | m/s, eastward then northward |
| $\sigma$ | scalar | m s$^{-1/2}$ |
| $\sigma_u$ | scalar | m/s, per axis |
| $T_L$ (`memory_time_s`) | scalar | s |
| Velocity error $\vec u$ | `[u, v]`, one per particle | m/s |

The two vector orders are crossed, deliberately: `lat` and `lon` name axes, in the order
D020 stores them and `sar.utils.geo` enforces, while `u` and `v` name components, in the
order `sar.model.drift` already uses. Swapping either pair produces a plausible wrong
answer rather than an error, which is why `test_position.py` has a test whose only job is
to confirm that $u$ moves longitude and $v$ moves latitude and neither leaks into the other.

Longitude comes back wrapped into 0 to 360, so a position handed in as $-78.6$ comes back
as $281.4$. Latitude is not wrapped: folding a pole crossing back over the top also flips
the longitude by 180 degrees, which this step does not model, so a step that large is
refused rather than quietly mangled.

NaN propagates, as it does through `calculate_drift`. HYCOM writes NaN for land, a particle
over land is a beaching question for D016, and a missing drift must produce a missing
position rather than a stationary one.

## The pipeline

`sar.pipeline.track.DriftPipeline` is the loop. Given a start time, a start position, a
step and a duration, it repeats four things:

1. sample the current and the wind where the particle is, at the time it is, through a
   `sar.pipeline.forcing` backend;
2. combine them into a drift with `calculate_drift`;
3. advance the position with `calculate_position`;
4. hand out the state it started the step from.

`track` is a generator. It yields the state at $t = 0$ before anything has been advanced,
then after one step, then after two, ending at $t =$ duration. **A track of $n$ steps
visits $n+1$ positions**, so an hour at a 60 s step yields 61 states and not 60.

Each `TrackState` carries the positions, and the current, wind and drift belonging to the
step that **leaves** that state. The final state has none of the three, because nothing
leaves it. With a random velocity it also carries each particle's `velocity_error`. `track`
keeps that error from step to step and hands it to `advance(positions, time,
velocity_error)`, so `advance` stays a function of what it is given.

A duration that is not a whole number of steps is refused rather than silently shortened,
because a track that stops early looks exactly like a track that drifted less far.

`timestep` is required on the pipeline and has no default, unlike the one on
`calculate_position`. The step is not an implementation detail of the loop; it is a
parameter of the run, every figure in the error budget below is linear in it, and a run
that inherited 60 s silently would be a run whose accuracy nobody stated. D009 fixes 60 s,
and `INTEGRATION_STEP_SECONDS` holds that value for a caller with no reason to choose
otherwise.

`seed` is likewise reported back by `describe`, so a run that used a non-zero $\sigma$ can
be reproduced exactly from what it recorded.

### Forcing

Every lookup goes through one `sample(lats, lons, time)` interface, which is D009's
`ForcingProvider`. One backend exists so far: `ConstantForcing`, a uniform steady field
whose answer can be worked out on paper, which is how the loop is tested without a file or
a network. Reading the stored HYCOM and ERA5 grids through `sar.model.interpolate` is a
separate ticket, and until it lands nothing here touches a grid.

### What the Monte Carlo ticket will change

Not the step, which already carries $\sigma$, and not the ensemble shape, which is already
an $(N, 2)$ array throughout. What is left is choosing $\sigma$, choosing $N$, spawning
independent streams per worker with `np.random.SeedSequence(base).spawn(n)` rather than
seeding once and forking, and reducing the $N$ endpoints into the probability map the
search pattern is drawn from.

## The error budget

Six things stand between this step and the truth. Figures are for 26.5 N over 24 hours at
the Gulf Stream's typical 1.8 m/s, which covers 155.5 km.

### 1. Euler truncation

The drift is held constant across the step, but the real drift changes continuously. The
local error is $\tfrac{1}{2}\left\lvert\dot{\vec v}\right\rvert\Delta t^2$ per step and
accumulates to first order in $\Delta t$.

Measured against a case with a closed-form answer: a velocity of constant speed turning at
the inertial period at 26.5 N, which is 26.8 hours and about as fast as a real ocean
velocity turns, integrated for 24 hours.

| $\Delta t$ | Endpoint error after 24 h |
|---|---|
| 1 s | 0.58 m |
| **60 s** | **35.0 m** |
| 300 s | 175 m |
| 900 s | 526 m |
| 3600 s | 2,104 m |

Exactly linear in $\Delta t$, as first order predicts. At 60 s that is 0.02 percent of the
track. At an hourly step it would be 2.1 km, which is the argument for the minute: the
forcing is served 3-hourly and hourly, but the integration step is not the cadence of
anything, it is the length at which the frozen-velocity assumption stops costing anything.

### 2. The frozen $\cos\varphi$

$\Delta\lambda$ uses the latitude the step starts from, while the exact answer uses the
latitude throughout. For constant $u$ and $v$ that exact answer is the Mercator relation,

$$\Delta\lambda = \frac{u}{v}\left[\ln\tan\left(\frac{\pi}{4}+\frac{\varphi}{2}\right)\right]_{\varphi_0}^{\varphi_1}$$

so this one can be checked rather than bounded. With 1 m/s east and 1 m/s north for 24
hours, moving the particle 86 km east and 0.78 degrees north:

| $\Delta t$ | Relative longitude error | On 86 km |
|---|---|---|
| **60 s** | $-2.4\times10^{-6}$ | **0.21 m** |
| 900 s | $-3.6\times10^{-5}$ | 3.1 m |
| 3600 s | $-1.4\times10^{-4}$ | 12.4 m |
| 86400 s | $-3.4\times10^{-3}$ | 295 m |

Again linear in $\Delta t$, matching the per-step prediction
$\tfrac{1}{2}\tan\varphi\cdot v\Delta t/R$, which is $2.3\times10^{-6}$ at 60 s. It is
negative because a particle moving north ends the step where a degree of longitude is
shorter, so freezing $\cos\varphi$ at the start understates how far east it went. At a one
minute step this is 21 cm a day. The table is here because it is the term people expect to
be large, and it is only large if the step is.

### 3. The sphere against the ellipsoid

The largest numerical term, and the only one that does not shrink with the step. Latitude
needs the meridional radius of curvature $M(\varphi)$ and longitude the prime vertical
radius $N(\varphi)$, both varying with latitude; a single mean radius is wrong by a fixed
fraction everywhere. Values from WGS84, $a = 6{,}378{,}137$ m and $e^2 = 0.00669438$.

| Latitude | $M$ | $N$ | Latitude step | Longitude step |
|---|---|---|---|---|
| 17 N | 6,340,881 m | 6,379,963 m | 0.48 percent short | 0.14 percent long |
| 26.5 N | 6,348,126 m | 6,382,392 m | 0.36 percent short | 0.18 percent long |
| 36 N | 6,357,482 m | 6,385,526 m | 0.21 percent short | 0.23 percent long |

Over a day's 155 km that is about 560 m of northward displacement at the box's middle and
about 280 m of eastward. It is a systematic bias, not scatter: every step errs in the same
direction, so it grows linearly with the track and no ensemble averages it away.

Fixing it costs two lines, replacing $R$ with $M(\varphi)$ and $N(\varphi)$. It is left
out because the ticket specifies the flat-earth spherical form, and because section 5
shows the number it would buy is three orders of magnitude below the error already in the
forcing. `EARTH_RADIUS_M` is defined once, in `sar.utils.geo`, so the change would be made in one place.

### 4. The stochastic term

**Since ADR005 it is a random velocity:** σ_u = 0.226 m/s per axis with T_L = 25.7 h. Its
spread is Taylor's, $\sqrt{2\sigma_u^2 T_L\,[t - T_L(1 - e^{-t/T_L})]}$:

| | One 60 s step | 1 h | 3 h | 4 h | 24 h | 48 h |
|---|---|---|---|---|---|---|
| Random velocity, σ_u = 0.226 | 14 m | 0.81 km | 2.39 km | 3.17 km | 16.9 km | 29.9 km |
| Random walk, σ = 26.3 | 204 m | 1.58 km | 2.73 km | 3.16 km | 7.7 km | 10.9 km |
| Dev drifters, 90 % spread per axis | | 0.78 km | 2.32 km | 3.07 km | 15.8 km | 29.4 km |

Both are matched at 4 h. Only the random velocity matches the drifters either side of it.
On the σ_u ladder, coverage of the 90 % region is flat from 1 to 6 h (89.4 to 90.0 % at
σ_u = 0.22), where the random walk ran from 98 % to 74 %. The rest of this section describes
the random walk, which is kept for comparisons.

A spread rather than a bias: it grows as $\sqrt{t}$, has zero mean, and is the only term
here that an ensemble is meant to expose rather than remove.

**Calibrated at $\sigma$ = 26.3 m s$^{-1/2}$**, chosen so the real buoy lands inside the
ensemble's 90 % region in 90 % of dev windows at **4 h**, the end of the longest search
(`docs/sigma-calibration.md` §0). The first calibration, at 24 h, gave 54.4.

| | One 60 s step | 1 h | 3 h | 4 h | 24 h |
|---|---|---|---|---|---|
| Spread per component, $\sigma\sqrt t$ | 204 m | 1.6 km | 2.7 km | 3.2 km | 7.7 km |

That is 130 to 530 times the trial values used before it was measured (0.05–0.2,
which gave 15–59 m in a day). The term no longer stands for minute-scale noise. It is the model's
account of the forcing's own error over a search's first hours, now measured: an eddy
diffusivity $K = \sigma^2/2 \approx 350$ m$^2$ s$^{-1}$.

Two things that follow:
- **One kick is 204 m**, about twice the drift's own ~108 m step at 1.8 m/s. It cannot jump
  a land cell, which is at least 4.45 km across.
- **The error it stands for is not a random walk.** The real gap grows as $t^{1.91}$ over
  the first 6 h, so a constant $\sigma$ is right at 4 h only: too wide before, too narrow
  after (vault L22). At 24 h this $\sigma$ under-covers badly, and any claim there needs the
  24 h value.

### 5. The forcing itself

This dominates everything above by three orders of magnitude, and is why none of the
numerical terms is worth chasing.

The drift is only as good as the current and wind fed to it. `sar.model.interpolate`
quantifies the interpolation part and `docs/resultant-vector.md` sets it out. On top of
that sits error in the fields themselves: HYCOM surface currents carry an uncertainty of
order 0.1 to 0.2 m/s in a strong-current region, and the leeway coefficient is known only
to 1 to 4 percent of wind speed against a mid value of 2 percent (D002, D018, after Allen
2000, DTIC ADA376479).

A 0.1 m/s error in the current, held for a day, is 8.6 km of position. At 10 m/s of wind
the leeway range spans 0.1 to 0.4 m/s, so the coefficient alone moves the endpoint over a
span of 26 km a day. Against that, the 35 m of section 1 and the 560 m of section 3 are
rounding.

### 6. Floating point

Nothing. Over the 1,440 additions of a day the accumulated departure from the exact sum is
about 1.5 micrometres, and the spacing of a double at a longitude of 281 degrees is
6 nanometres on the ground. Listed only so it is visibly ruled out.

### Summary

| Term | Over 24 h at 26.5 N | Behaviour |
|---|---|---|
| Forcing and leeway uncertainty | 9 to 26 km | dominates everything |
| Sphere against ellipsoid | about 600 m | systematic, independent of $\Delta t$ |
| Euler truncation | 35 m | linear in $\Delta t$ |
| Stochastic term | 16.9 km per component at the calibrated σ_u = 0.226 m/s (3.2 km at 4 h, where it is matched) | spread, grows as $t$ for hours and $\sqrt{t}$ after days; it is the forcing error above, held for T_L = 25.7 h (ADR005) |
| Frozen $\cos\varphi$ | 0.21 m | linear in $\Delta t$ |
| Floating point | micrometres | negligible |

The order of that table is the argument for the design. A second-order integrator on an
ellipsoid would improve a 155 km track by about half a kilometre while the forcing under
it is uncertain by nine. Effort spent on better forcing, a better leeway coefficient or a
measured $\sigma$ buys more than effort spent here.

## Running it

One step, by hand:

```
python -m sar.model.position --position 26.53 281.4 --drift 1.2 0.9
python -m sar.model.position --position 26.53 281.4 --drift 1.2 0.9 --sigma 0.1 --seed 7
```

A track with the random velocity: `python -m sar.pipeline.track ... --sigma-u 0.226
--memory-h 25.7 --seed 7`.

A whole track in a steady field, where the answer can be checked on paper. A 1 m/s
eastward current for an hour moves a particle 3,600 m east, which at 26.53 N is 0.03619
degrees of longitude and nothing else:

```
python -m sar.pipeline.track --constant-current 1.0 0.0 \
    --start 2021-01-05T07:30 --lat 26.53 --lon 281.4 \
    --timestep 60 --duration 3600 --every 10
```
