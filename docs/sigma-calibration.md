# The size of the random kick (`sar.validate.calibrate_sigma`, issue #89)

Every minute, every particle in the ensemble gets a random shove,
$\sigma\sqrt{\Delta t}\,\vec Z$ (`docs/position-update.md`). After a time $t$ the shoves
alone have spread the cloud by $\sigma\sqrt t$ metres per axis. D026 starts every particle
at one point, so this shove is the only thing that spreads the cloud: at $\sigma = 0$, the
default until now, every particle follows the same path and there is no probability map.

This document is how $\sigma$ was measured against real drifters, what came out, and what
it means. The decision it supports is vault D028.

> **The answer.** $\sigma$ = **54.4 m s⁻¹ᐟ²** (95 % CI 52.9–57.7). That is $\sigma^*$ = 53.6,
> the value at which 90 % of undrogued dev buoys land inside the ensemble's 90 % region at
> 24 h, plus a person's crosswind slide that no buoy can show. It spreads the cloud by
> 5.7 km per axis at 3 h and 16.0 km at 24 h, and the 90 % region at 24 h covers about
> 3,750 km². **It is a parameter of this model at a 24 h horizon, not a constant of the
> ocean.** The cloud is honest at 24 h, too wide before and too narrow after, because the
> real error grows as $t^{1.74}$.
>
> Along the way this answers **why a buoy drifts differently from the model** (§4).
> Drogued buoys ignore the wind the model gives a person, and that costs 4 km a day. The
> ~16 km a day that remains is HYCOM's current itself, and the buoy's own velocity, held
> fixed, beats it for the first two days.

## 1. What we are trying to get right, and the naive approach

**The goal.** The map says "the person is inside this region with 90 % probability". That
sentence should be true: across many cases, the real object should land inside the 90 %
region about 90 % of the time. Too narrow, and the helicopter searches a box the person is
not in. Too wide, and the search wastes hours on empty sea.

**The naive approach**, the first idea we had: start a simulated person at a buoy's
position, run it, and adjust $\sigma$ until it ends up *as close as possible* to where the
buoy went. **That cannot work.** The shove has zero mean. It widens the cloud and does not
move its centre.
- If you score one random particle's distance to the buoy, $\sigma = 0$ always wins,
  because noise only adds distance.
- If you score the cloud's centre, $\sigma$ hardly changes it.

How close the forecast gets is set by the current and the wind, not by $\sigma$. That is
requirement R2b.

**The idea that fixes it.** $\sigma$ controls width, so calibrate it on what width is for:
**coverage**, the fraction of buoys inside the cloud's 90 % region. Then $\sigma^*$ is the
value at which coverage is 90 % (R2c's level), 24 h after the start (P1, Aditya,
1 Oct 2026).

## 2. The data: 12,973 windows

`sar.validate.drift_windows` cuts the **dev** units of the sealed split
(`docs/validation-split.md`) into 48 h windows. A window is the stretch where the model
starts at the buoy's fix and both are followed for 48 hours.

| Rule | Why |
|---|---|
| Dev units only; a sealed or holdout-2023 unit raises `SealedUnitError`, with no override | D025: nothing that tunes the engine may see the tracks it is tested on |
| One drogue tier per window; tier-uncertain units are left out | D018 compares the tiers |
| Windows start at 00:00 UTC, every 48 h, never overlapping | The forcing is sampled at one time per call, so only windows that start together can share an engine run. That gives 1,394 runs instead of 12,973, and loses at most 23 h per unit. |
| Start fix within 3 h of a real GPS fix; truth only where `fix_gap_h <= 3` | D025's truth rule, so calibration and the sealed test agree on where the buoy was |

Built 2 Oct 2026 on the cluster (job 58740), identical to the laptop build:

| | Windows | Groups | Units |
|---|---|---|---|
| Undrogued | 10,049 | 109 | 301 |
| Drogued | 2,924 | 44 | 114 |

154 windows were dropped because their start fix was interpolated, and 99.1 % of the
hours count as truth. 28 windows on two start days cross HYCOM's 12 h gap on
2020-10-25 and were refused (D011).

**Groups, not windows, are the unit of independence.** Windows of one buoy, and of buoys
that shared water (D025's 10 km groups), are not independent samples. Every confidence
interval here comes from resampling whole groups. It is computed again by resampling
calendar months, because a badly placed HYCOM eddy affects every buoy near it for weeks,
and the wider of the two intervals is reported.

## 3. The method, in four stages

| Stage | What it does | Why it exists |
|---|---|---|
| **0** | Tests that the kick is the right size whatever the step, the same east as north, unchanged by the real forcing backend, and withheld from beached particles | Before trusting the experiment, prove the apparatus works (`tests/pipeline/test_track.py`, `test_gridded.py`) |
| **1** | One particle per window at $\sigma = 0$, once with $\alpha = 0.02$ (the engine) and once with $\alpha = 0$. The gap to the buoy, every hour, east/north and downwind/crosswind | How wrong the model is before any randomness. This is the drogued against undrogued comparison, the gate, the growth exponent, and a first estimate $\sigma_0$ |
| **1b** | The twin experiment: fake buoys with a **known** $\sigma$, driven by the real ocean, through stages 1 and 2 | The only test where the right answer is known and the ocean is real |
| **2** | Real ensembles, 1,000 particles per window, at $\sigma_0 \times \{0.5, 0.7, 1, 1.4, 2\}$, each scored on whether the buoy is inside the 90 % region | Picks $\sigma^*$ by coverage, with the engine exactly as it is used |
| **3** | Adds the person's crosswind slide, which a round buoy cannot show | The one part of a person's drift no buoy measures |

### The gap, and the wind frame

The gap is buoy minus model, in metres on a local flat earth at the midpoint latitude.
The **wind run** is the integral of the 10 m wind along the model's path so far,
$\vec R(t) = \int_0^t \vec v_{w10}\,dt'$ in metres. It is the vector the leeway term
multiplies, so its direction defines *downwind* and $90°$ clockwise from it defines
*crosswind* (positive to the right). A model that runs ahead of the buoy downwind gives a
negative downwind gap.

The leeway mismatch between buoy and model falls out of this frame. If the buoy feels a
fraction $\alpha_b$ of the wind and the model $\alpha$, then on average
$g_{dw} \approx (\alpha_b - \alpha)\,|\vec R|$, so a regression through the origin of
$g_{dw}$ on $|\vec R|$ gives $\alpha_b - \alpha$. **This is a property of the buoys,
compared with the literature. The model's $\alpha$ stays 0.02 and is never fitted (D018).**

### Three estimates of $\sigma$ at a horizon $T$

| Estimator | Formula | What it assumes |
|---|---|---|
| Raw | $\sigma^2 = \langle g_e^2 + g_n^2\rangle / 2T$ | The mean squared gap is all randomness |
| De-biased | $\sigma^2 = (\mathrm{var}\,g_{dw} + \mathrm{var}\,g_{cw}) / 2T$ | Removes the mean gap in the wind frame (the buoy-is-not-a-person bias) |
| Quantile | $\sigma = R_{90} / (2.146\sqrt T)$ | $R_{90}$ is the 90th-percentile gap. A round Gaussian cloud of $\sigma\sqrt T$ per axis holds 90 % inside $\sqrt{-2\ln 0.1} = 2.146$ of those, so this gives *exactly* 90 % coverage to the formula, and it ignores outliers |

### The growth exponent $\beta$

$\beta$ is the slope of log mean squared gap against log $t$, over 6–48 h, using windows
valid at every hour. A random walk grows its squared gap as $t$, so $\beta = 1$. A velocity
error that persists grows it as $t^2$. If $\beta$ is not 1, a constant $\sigma$ is right at
one horizon only: too wide before it, too narrow after.

## 4. Stage 1 results (cluster array 58741, 2 Oct 2026, 12 min)

### Table A: why a buoy drifts differently from the model

Median distance from the buoy, in km, with 95 % CIs. Figure:
`figures/report/sigma_separation_by_tier.png`.

| At 24 h | Drogued, $\alpha$ = 0 | Drogued, $\alpha$ = 0.02 | Undrogued, $\alpha$ = 0 | Undrogued, $\alpha$ = 0.02 |
|---|---|---|---|---|
| Model, median | **16.9** (15.6–19.1) | 21.0 (19.4–22.9) | 16.2 (15.7–16.8) | **16.4** (15.6–17.1) |
| Model, 90th percentile | 37.4 | 42.7 | 33.4 | 33.9 |
| "Buoy stays put" | 17.9 | 17.9 | 19.4 | 19.4 |
| "Buoy keeps its first velocity" | 11.5 | 11.5 | 12.8 | 12.8 |
| Model's lead downwind | 3.2 km | 12.9 km | −4.8 km | 5.3 km |
| Buoy's windage minus the model's | −0.71 % | −2.69 % | +0.90 % | −1.11 % |
| Windows / groups | 2,848 / 44 | 2,839 / 44 | 9,856 / 108 | 9,852 / 108 |

Read across, this is the answer to "why does a buoy drift differently from the equation":

1. **The wrong object.** A drogued buoy follows the water at 15 m and ignores the wind, so a
   person's 2 % leeway runs the model 12.9 km ahead of it in a day. Leaving the leeway term
   out brings drogued buoys from 21.0 km to 16.9 km.
2. **The buoys' own windage is measured, and it is consistent.** Undrogued buoys feel
   **0.89–0.90 %** of the wind *relative to HYCOM's surface current*. Drogued buoys feel
   **−0.69 to −0.71 %**: they lag the surface current downwind. That is what a 15 m drogue
   should do beneath a surface layer that HYCOM's 0 m current already pushes downwind
   (L11). The two runs agree, because the same number comes out whichever $\alpha$ the
   model used.
3. **Leeway doesn't help undrogued buoys, and that is expected rather than wrong.** Their
   windage sits halfway between 0 and 2 %. The model is 4.8 km behind without the term and
   5.3 km ahead with it, so the median distance does not move (16.2 against 16.4 km).
   D018's pattern half holds: the term hurts drogued buoys as predicted, and does not
   measurably help undrogued ones.
4. **What is left, about 16 km at 24 h, is the ocean data.** At each start, HYCOM's current
   and the buoy's own velocity agree in direction (mean rotation under 4°, vector
   correlation 0.55–0.59, equal mean speeds) but differ by **0.32–0.33 m/s RMS**. Held for a
   day, that is about 28 km of position, and no change to the equation removes it.

### The gate fails against persistence

| At 24 h, undrogued, $\alpha$ = 0.02 | Model | Baseline | Groups where the model wins | One-sided sign test |
|---|---|---|---|---|
| "Buoy stays put" | 16.4 km | 19.4 km | 87 of 108 | p = 5 × 10⁻¹¹: **the model wins** |
| "Buoy keeps its first velocity" | 16.4 km | 12.8 km | 24 of 108 | p = 1: **persistence wins** |

The buoy's own measured velocity, held constant, beats the HYCOM-driven model until about
48 h, where the two tie (30.1 against 30.2 km). The buoy's velocity carries the real eddy,
while HYCOM's eddy is somewhere nearby.

The plan's pre-registered stop rule (P6) was to stop if the model failed either baseline.
**The calibration was run anyway, and the reasons are in vault D028:**
- the forcing was checked for a bug and none was found (below);
- persistence needs the *velocity* of the drifting object at the datum, which a person
  reported in the water does not give;
- a width calibrated to this model's real error is honest whether or not the centre beats
  persistence.

Whether to accept that is Aditya's decision. **It also bears on D025's R2b**, which uses
persistence as a pass threshold.

**The forcing check** (`scripts/check_current_lag.py`, job 58773, 400 random start days,
3,684 windows). It compares HYCOM's current at the start with the buoy's velocity, with
HYCOM sampled from 24 h before the start to 24 h after.
- Agreement peaks at −1 to 0 h (vector correlation 0.567 undrogued, 0.538 drogued), so
  there is no clock error larger than an hour.
- The rotation is under 4°, so there is no swapped or flipped axis.
- HYCOM's speed is 0.91–0.98 of the buoys', so there is no unit error.
- The agreement dips at ±12 h and recovers at ±24 h. That is the inertial swing, roughly a
  day here, and HYCOM carries it in phase with the buoys.

### The exponent: the gap is not a random walk

| | $\beta$ (95 % CI) |
|---|---|
| Undrogued, $\alpha$ = 0.02 | **1.74** (1.65–1.81) |
| Drogued, $\alpha$ = 0.02 | 1.88 (1.78–1.96) |

Both are far from 1, much nearer the persistent-error value of 2. So a constant $\sigma$ is
right at **one horizon only**, which is why P1 fixed 24 h before anything was measured.
Figure: `sigma_msd_growth.png`.

### $\sigma$ by horizon

Undrogued, $\alpha$ = 0.02, in m s⁻¹ᐟ². $K = \sigma^2/2$ is the eddy diffusivity the kick
stands for, from the de-biased $\sigma$. Figure: `sigma_by_horizon.png`.

| $T$ | Quantile $\sigma$ (95 % CI) | De-biased $\sigma$ | $K$ (m² s⁻¹) |
|---|---|---|---|
| 6 h | 30.7 (29.4–32.1) | 30.5 | 464 |
| 12 h | 40.7 (39.1–42.9) | 46.5 | 1,083 |
| **24 h** | **53.7 (51.6–56.4)** | **54.5** | **1,483** |
| 48 h | 70.8 (68.0–74.4) | 69.4 | 2,409 |

$\sigma$ rises as $T^{(\beta - 1)/2} \approx T^{0.37}$: from 24 h to 48 h that predicts a
factor of 1.29, and the table gives 70.8 / 53.7 = 1.32.

**$\sigma_0 = 53.7$**: the quantile estimate at 24 h, which seeds the ladder.

At 24 h, downwind $\sigma$ is 57.0 (52.3–64.3) and crosswind 51.9 (47.9–57.1). The intervals
overlap, so under P4 **one $\sigma$** is used and `position.py` stays isotropic. The mean
gap is 2.5 km to the *left* of the wind: the model sits slightly to the right of the
buoys, where HYCOM's surface current carries its wind-driven, to-the-right-of-the-wind
flow. Figure: `sigma_wind_frame.png`.

### $\sigma$ depends on how strong the current is

Undrogued, at 24 h, split by HYCOM's current speed at the start:

| Current at start | Windows | Quantile $\sigma$ (95 % CI) | Median miss |
|---|---|---|---|
| < 0.3 m/s | 6,415 | 47.3 (45.2–49.4) | 14.9 km |
| 0.3–1 m/s | 3,396 | 62.5 (60.0–66.7) | 19.5 km |
| **> 1 m/s** | **41** | **104 (78–130)** | **32.9 km** |

**A pooled $\sigma$ will under-cover in the Gulf Stream itself**, the strong-current water
this project is about. Only 41 undrogued windows start in a current above 1 m/s: the
Stream is a narrow jet and buoys cross it fast. This goes in the limitations register.

## 5. Twin experiment (cluster array 58786, 2 Oct 2026)

**Can the method give back a $\sigma$ we know?** Fake buoys were made by running the
engine itself with a planted $\sigma$, one particle per window. They started at real buoys'
start points and were driven by the real HYCOM and ERA5 on 150 random start days, about
1,060 undrogued windows per case. Then they went through stages 1 and 2 exactly like real
buoys. The model is exactly right for these buoys, so the right answer is known and the
ocean, shear included, is real. Figure: `sigma_twin_recovery.png`.

| Planted | Stage 1 formula (95 % CI) | Stage 2 $\sigma^*$ (95 % CI) | Coverage at the planted $\sigma$ | $\beta$ |
|---|---|---|---|---|
| 20 | 20.9 (20.1–21.7) | **19.9** (19.3–21.1) | 90.4 % | 1.17 |
| 50 | 52.1 (49.9–54.5) | **50.1** (48.5–53.6) | 89.9 % | 1.15 |
| 100 | 104.2 (100.1–108.8) | **103.8** (98.5–110.6) | 89.1 % | 1.10 |
| 50, plus a person's crosswind slide | 51.9 (49.9–54.4) | **51.4** (49.2–55.0) | — | 1.15 |

What it shows:

- **Stage 2 recovers every planted value inside its interval.** This was the pass test, and
  it passes.
- **Stage 1's formula reads about 4 % high.** The planted shove is stretched again by the
  current's shear, so the one-track formula counts shear twice, but only slightly at 24 h.
  On the real buoys the formula gave 53.7 against the ladder's 53.6, consistent with this.
- **Shear alone makes $\beta \approx 1.15$.** The fake buoys are a pure random walk, yet in
  the real ocean their gap grows as $t^{1.15}$. The real buoys' $\beta$ = 1.74 is therefore
  mostly a forcing error that persists, not shear.
- **The slide case backs P5's rule.** Adding a 0.51 % crosswind slide with a random side
  raised $\sigma^*$ from 50.1 to 51.4. Matching on the crosswind axis predicts
  $\sqrt{50.1^2 + 9.55^2} = 51.0$; spreading it over both axes predicts 50.5. The rise
  follows the crosswind-axis rule, which is also the conservative one.
- **The energy-score optimum reads about 4 % low even when the truth is known** (19.2, 48.0
  and 96.1). That is a property of fitting a parabola to a five-point ladder in log
  $\sigma$. On the real buoys, the optimum of 49.5 is therefore about 4 % below $\sigma^*$
  once this is allowed for, not 8 %.

## 6. Stage 2: the ladder (cluster array 58774, 2 Oct 2026)

Real ensembles: 1,000 particles per undrogued window, released from one point (D026),
with $\alpha$ = 0.02, 60 s steps and beached particles frozen and kept (D016). That is the
engine exactly as it will be used. Each cloud is scored at 6, 12, 24 and 48 h against the
buoy (`sar.validate.region`). The 90 % region is found from the particles themselves: no
grid, and no circle. Figure: `sigma_coverage_ladder.png`.

| $\sigma$ | 26.9 | 37.6 | **53.7** | 75.2 | 107.4 |
|---|---|---|---|---|---|
| Buoys inside the 90 % region at 24 h | 53.3 % | 74.3 % | **90.1 %** | 96.7 % | 98.5 % |
| Mean energy score (km) | 14.20 | 13.54 | **13.30** | 13.88 | 15.76 |

9,972 undrogued windows. 12 tasks × 4 processes; 29 start days each took about 32 min
(66 s a day).

- **$\sigma^* = 53.6$** (95 % CI 52.1–56.9, wider of the group and month resamplings),
  read off by interpolating in log $\sigma$ where coverage crosses 90 %.
- **Stage 1's formula gave 53.7.** The worry that the formula double-counts shear did not
  show up at this horizon. The twin experiment (§5) tests it directly.
- **The energy score is lowest at $\sigma$ = 49.5** (a parabola in log $\sigma$). That is
  the "best" width with no percentage chosen, and it is 8 % below $\sigma^*$. So matching
  at 90 % costs only a little sharpness, and the cloud's shape is close to right.

### The confirmation run, at $\sigma$ = 54.4 (array 58798)

Every dev window of both tiers, 1,000 particles each, scored at seven lead times. Figure:
`sigma_coverage_by_lead.png`.

| Lead | Undrogued inside the 90 % region | Drogued | Median 90 % area | Median centroid error (undrogued) | Particles beached |
|---|---|---|---|---|---|
| 1 h | 99.3 % | 99.7 % | 154 km² | 0.8 km | 1.1 % |
| 3 h | 99.1 % | 99.2 % | 463 km² | 2.5 km | 1.4 % |
| 6 h | 98.4 % | 97.6 % | 925 km² | 4.7 km | 1.6 % |
| 12 h | 96.2 % | 92.8 % | 1,856 km² | 8.7 km | 2.1 % |
| **24 h** | **90.4 %** (88.7–91.6) | **81.5 %** (76.6–85.5) | **3,750 km²** | **15.8 km** | 2.9 % |
| 36 h | 85.3 % | 73.9 % | 5,701 km² | 22.5 km | 3.7 % |
| 48 h | 81.5 % | 68.3 % | 7,711 km² | 28.7 km | 4.4 % |

- **Honest at 24 h, by construction, and the $\beta$ = 1.74 shape either side.** Before
  24 h the cloud is too wide, which is safe; after, it is too narrow. That is what P1 traded,
  knowingly.
- **Drogued buoys are covered less (81.5 %)** because the engine gives them a person's 2 %
  leeway, which they do not feel (Table A). That is a fact about the test object, not about
  a person. It bears on D025, which scores R2c over both tiers.

**Reliability at 24 h** (undrogued): the fraction inside the region of each stated
probability.

| Stated | 50 % | 68 % | 80 % | 90 % | 95 % |
|---|---|---|---|---|---|
| Observed | 58.9 % | 74.1 % | 83.2 % | 90.4 % | 93.9 % |

This answers the outside critique's first question. Matching at 90 % leaves the **core too
wide** (the 50 % region holds 59 %) and the **far tail slightly too thin** (the 95 % region
holds 93.9 %). The real errors have heavier tails than the cloud: most buoys sit closer
than the cloud expects, and a few sit further out. The effect is modest. Every level lands
within 9 points of the diagonal, and the 80 % and 90 % levels the search uses most are
within 3.

### The check: particle count and kernel width (array 58799)

100 random start days, 732 undrogued windows, 4,000 particles, at 24 h:

| | 1,000 particles | 4,000, half width | 4,000, normal width | 4,000, double width |
|---|---|---|---|---|
| Inside the 90 % region | 89.1 % | 88.9 % | 88.8 % | 88.8 % |
| Inside the 50 % region | 55.1 % | 55.3 % | 55.7 % | 56.1 % |

The 90 % coverage moves by 0.3 points across all four, inside the plan's 1-point pass mark.
1,000 particles and the default kernel are enough, and the answer does not depend on the
kernel. (This subset reads 89 % rather than 90.4 % because it is a subset.)

## 7. Stage 3: the person's crosswind slide

A person also slides $a_c = 0.51\,\%$ of the wind speed *across* the wind (Allen 2005, in
D018), to a side nobody can know in advance. Over $T$ that is a displacement of
$\pm a_c |\vec R(T)|$ at right angles to the wind run. A random walk has variance
$\sigma^2 T$ on that axis, so matching the crosswind variance gives

$$\sigma_c = a_c\,\sqrt{\langle |\vec R(T)|^2\rangle} \big/ \sqrt T.$$

On the undrogued dev windows at 24 h, the RMS wind is 6.37 m/s, so **$\sigma_c = 9.55$**
(9.35–9.86). Spreading the same variance over both axes would give 6.75. The twin
experiment (§5) follows the crosswind-axis value, so that is the one used:

$$\sigma_{\rm final} = \sqrt{\sigma^{*2} + \sigma_c^2} = \sqrt{53.6^2 + 9.55^2} = \mathbf{54.4}
\quad (52.9\text{–}57.7).$$

**Why the variances simply add**, which the outside critique of 2 Oct challenged. Two
errors' variances add whenever their covariance is zero; they do not need to be
independent. The side of the slide is a fair coin, independent of everything else, so
$\mathrm{cov}(A, \pm S) = \mathbb E[\pm]\,\mathbb E[A S] = 0$ for any ocean error $A$, even
though $A$ and $S$ both depend on the wind. A buoy is round, so the slide is not already
inside $\sigma^*$.

## 8. A worked window

Buoy `300234066418750`, window 19, starting 2020-12-10 00:00 UTC at 25.6637 N, 64.8116 W,
undrogued, east of the Bahamas. Its 24 h miss is the median.

| | Value |
|---|---|
| HYCOM current at the start | (0.007, 0.069) m/s: gently north |
| The buoy's own velocity at the start | (0.046, −0.168) m/s: south at 0.17 m/s |
| Wind run over 24 h | 381.7 km (a mean 4.4 m/s) |
| Gap at 24 h, east and north | +4.45 km, −15.73 km |
| Distance | $\sqrt{4.45^2 + 15.73^2}$ = **16.34 km** (the median is 16.35) |
| Downwind and crosswind | +14.22 km, −8.06 km |
| "Stays put" / "keeps its first velocity" | 14.59 km / **5.67 km** |

HYCOM put this water moving north. The buoy was moving south, and kept doing so, so
persistence beat the model by 10 km. That is reason 1 in Table A, in a single window.

With $\sigma_0 = 53.7$, the formula's cloud at 24 h is $53.7 \times \sqrt{86{,}400} =
15.8$ km per axis, and its 90 % circle has radius $2.146 \times 15.8 = 33.9$ km. This buoy,
at 16.3 km, is inside. Across all 9,852 undrogued windows, the 90th-percentile miss is
33.88 km, which is how $\sigma_0 = 33{,}879 / (2.146 \times 293.94) = 53.7$ came out.

## 9. Running it

On the cluster, from a checkout of the branch:

```
D=/home/26p67/data/derived/sigma
sbatch scripts/calibrate_sigma.sbatch windows --out $D/windows
sbatch --array=0-31%24 scripts/calibrate_sigma.sbatch stage1 --windows $D/windows --out $D/stage1
python -m sar.validate.calibrate_sigma fit --windows $D/windows --stage1 $D/stage1 --out $D/fit.json
SIGMA_PROCS=4 sbatch --array=0-11 --cpus-per-task=4 scripts/calibrate_sigma.sbatch ladder \
    --windows $D/windows --sigmas 26.9 37.6 53.7 75.2 107.4 --out $D/ladder
SIGMA_PROCS=4 sbatch --array=0-11 --cpus-per-task=4 scripts/calibrate_sigma.sbatch twin \
    --windows $D/windows --days 150 --out $D/twin
python -m sar.validate.calibrate_sigma calibrate --windows $D/windows --fit $D/fit.json \
    --ladder $D/ladder --twin $D/twin --out $D/calibration.json
python scripts/plot_sigma_calibration.py --windows $D/windows --stage1 $D/stage1 \
    --fit $D/fit.json --calibration $D/calibration.json --out figures/report
```

The partition gives each array task a whole node, so the ensemble stages run four processes
per task (`SIGMA_PROCS`). The data stays under `/home/26p67/data/derived/sigma/` and never
enters the repo.

**Measured cost:**

| Stage | Size | Time |
|---|---|---|
| Windows | 12,973 windows from 12 MB of drifters | 17 s, one node |
| Stage 1 | 1,394 start days × 2 runs, one particle per window | 3.0 min per task of 44 days; 12 min wall for 32 tasks |
| Fit | 1.27 M residual rows, 1,000 bootstraps × 2 clusterings | 35 s, laptop |
| Ladder | 9,972 windows × 1,000 particles × 5 σ | 32 min per process, 48 processes on 12 nodes |
| Twin | 150 start days × 4 cases × (1 fake buoy + 5 σ ladder) | 8–10 min per process |
| Confirmation | 12,945 windows × 1,000 particles, 7 leads | 9–10 min per process |
| Check | 100 days × 4,000 particles, 3 kernel widths | 8–9 min per process, 12 processes |

## 10. Limitations this adds

- **$\sigma$ is right at 24 h only** ($\beta$ = 1.74): the cloud is too wide earlier and too
  narrow later.
- **A pooled $\sigma$ under-covers in currents above 1 m/s** by about a factor of two.
- **The buoys measure the ocean's error, not a person's.** Undrogued buoys are a lower bound
  on a person's wind drift (D018), not a measurement of one. Only the crosswind slide is
  added, from Allen 2005.
- **HYCOM's 0 m current runs downwind of the 15 m water** by about 0.7 % of the wind, so the
  leeway term partly double-counts wind-driven surface flow (L11, now measured).
- **All windows start at 00:00 UTC**, about 20:00 local time, so the diurnal phase is
  always the same. That is minor offshore.

## Not here

- **The sealed and holdout evaluation**, which is #52's, once per frozen engine.
- **Fitting $\alpha$**, which D018 forbids. The buoys' windage is measured and compared;
  the model's $\alpha$ stays 0.02.
- **A time-dependent $\sigma$, or a "random flight" kick with memory**, which would fit
  $\beta \ne 1$ properly. That would change D009 and is future work.
